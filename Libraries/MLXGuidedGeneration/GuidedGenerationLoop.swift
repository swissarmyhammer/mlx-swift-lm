// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import os

/// Runs grammar-constrained generation with fast-forward token support.
///
/// When the grammar forces deterministic tokens (e.g. JSON structural
/// characters `{`, `}`, `,`, `:`), they're fed through the model one at
/// a time to update the KV cache. Each pass uses the optimized T_q=1
/// Metal kernel.
///
/// The loop overlaps grammar mask computation (CPU) with the model forward
/// pass (GPU). After committing a token, the grammar state is ready for the
/// next mask computation. We compute it while the GPU processes the forward
/// pass, hiding the ~50us CPU cost behind the 10-100ms GPU latency.
///
/// This is a self-contained loop that has direct access to the grammar state
/// and can inject fast-forwarded tokens into both the output stream and the
/// KV cache.
public enum GuidedGenerationLoop {

    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX",
        category: "GuidedGenerationLoop"
    )

    // MARK: - Constants

    /// Logit penalty used to reject a token: applied at each EOS/stop
    /// position in the precomputed `eosPenalty` array (normal-zone
    /// premature-EOS suppression) and to every non-closing token in the
    /// hard zone's forced-closing bias. Large enough to always dominate
    /// any raw logit magnitude the model produces.
    private static let logitRejectionPenalty: Float32 = -10000.0

    /// Shared prefix for every diagnostic/progress log line `run()` emits.
    private static let logPrefix = "[GuidedGen]"

    /// Shared message-template fragment used by every "why did the loop
    /// stop" diagnostic log line (see `logStopReason`).
    private static let stopReasonFragment = " Stop reason: "

    /// Attoseconds per millisecond -- the divisor needed to fold
    /// `Duration.components.attoseconds` into the same millisecond count
    /// as `components.seconds * millisecondsPerSecond` for progress/final-stats
    /// logging (see `durationToMilliseconds`).
    private static let attosecondsPerMillisecond: Int64 = 1_000_000_000_000_000

    /// Milliseconds per second -- shared by `durationToMilliseconds`'s
    /// elapsed-time conversion.
    private static let millisecondsPerSecond: Int64 = 1000

    /// Runs the guided generation loop, yielding text deltas through `emit`.
    ///
    /// Overlaps grammar mask computation with GPU forward passes: after
    /// committing a token, the next mask is computed on the CPU while the
    /// model forward pass runs on the GPU.
    ///
    /// - Parameters:
    ///   - input: Prepared model input (prompt already tokenized)
    ///   - context: Model context (model, tokenizer, configuration)
    ///   - constraint: The xgrammar constraint (must have `fastForward: true`)
    ///   - maxTokens: Maximum tokens to generate
    ///   - completionReserve: Number of tokens before maxTokens at which closing
    ///     bias activates. When `tokenCount >= maxTokens - completionReserve`,
    ///     the bias nudges sampling toward JSON-closing tokens.
    ///   - hardReserve: Number of tokens before maxTokens at which a *hard*
    ///     closing zone activates: closing tokens are forced and all others
    ///     suppressed (plus an EOS penalty), more aggressively than
    ///     `completionReserve`'s soft nudge. Defaults to 0, which disables it.
    ///   - vocabSize: Number of tokens in the grammar's vocabulary. May differ
    ///     from the model's logit dimension (e.g. added special tokens beyond
    ///     the embedding size). Used to correctly interpret the grammar bitmask.
    ///   - kvBits: Bit width for KV-cache quantization, or nil to disable. When
    ///     set, the KV cache is quantized after each forward pass to reduce
    ///     memory use, mirroring the unconstrained `TokenIterator`. nil is a
    ///     no-op, so models that don't quantize are unaffected.
    ///   - kvGroupSize: Group size for KV-cache quantization (default 64, matching
    ///     `GenerateParameters`). Only used when `kvBits` is non-nil.
    ///   - quantizedKvStart: Token offset at which quantization begins (default 0).
    ///     Only used when `kvBits` is non-nil.
    ///   - closingBias: Pre-computed logit bias array favoring closing tokens
    ///     (from `ClosingTokenBias.compute`). Nil disables forced completion.
    ///   - whitespaceBias: Pre-computed negative logit bias array penalizing
    ///     whitespace-only tokens (from `WhitespaceTokenBias.compute`). Nil
    ///     disables whitespace suppression.
    ///   - whitespaceTokenIDs: Set of token IDs classified as whitespace-only.
    ///     Used by the run tracker to detect consecutive whitespace runs.
    ///   - diagnosticLog: When true, flush the grammar constraint's diagnostic
    ///     logs after the run completes. Defaults to false.
    ///   - cache: An existing `[KVCache]` to continue generating from
    ///     (e.g. one holding a prior round's prefilled prompt), or `nil`
    ///     to build a fresh cache internally. `KVCache` instances (reference
    ///     types) are mutated in place by this call, so the caller's own
    ///     array reference stays valid for in-place state changes -- BUT
    ///     `maybeQuantizeKVCache` (active when `kvBits` is set) replaces
    ///     array *elements* with new `QuantizedKVCache` instances inside
    ///     this call's local `[KVCache]` copy, and array value semantics
    ///     mean that replacement never reaches back into the caller's own
    ///     array variable. Use `RunResult.cache` from this call's return
    ///     value as the authoritative post-call array -- never assume the
    ///     `cache:` argument the caller passed in still reflects it,
    ///     particularly once `kvBits` triggers a mid-run replacement.
    ///   - onTokenCommitted: Called, in order, with the ID of every token
    ///     actually fed through the model this call (sampled tokens and
    ///     fast-forward tokens alike) -- i.e. exactly the tokens
    ///     `cache`'s `offset` advances over. Never called for a token that
    ///     is sampled/detokenized/emitted but never fed back (the terminal
    ///     token on a `commitResult.isTerminated` stop, or any
    ///     fast-forward token abandoned by a mid-batch stop), so a caller
    ///     accumulating these IDs for `PromptCache` storage never needs to
    ///     reconcile them against `cache.offset` by re-encoding emitted
    ///     text -- the count always matches by construction. `nil` is a
    ///     no-op, so existing callers are unaffected.
    ///   - emit: Callback for each text delta. Return `false` to stop.
    /// - Returns: A `RunResult` bundling the total token count (including
    ///   FF tokens, matching the previous bare `Int` return) and the final
    ///   `[KVCache]` array (see the `cache:` parameter doc above).
    /// - Throws: `GuidedGenerationError.incompleteOutput` if maxTokens is
    ///   exhausted before the grammar reaches a stop state.
    ///   `GuidedGenerationError.prematureEOS` if the model emits EOS
    ///   before the grammar accepts. On either throw, no `RunResult` is
    ///   produced -- NEITHER field, not just `cache`, is available from a
    ///   throwing exit: there is no authoritative `tokenCount` attached to
    ///   the thrown error either. `onTokenCommitted` has already reported
    ///   every token fed so far, though, so a caller that needs the count
    ///   on this path must tally the callback's calls itself (its running
    ///   count is exactly `RunResult.tokenCount` would have been) -- the
    ///   same pattern already used to report usage on an incomplete guided-
    ///   generation round. The possibly-quantization-updated `[KVCache]`
    ///   array is likewise only reachable by tallying, or -- since
    ///   `kvBits` is not set by any caller in this codebase today -- simply
    ///   unsupported in practice; wiring quantized KV caches through a
    ///   throwing exit is not yet supported.
    @discardableResult
    public static func run(
        input: LMInput,
        context: ModelContext,
        constraint: GrammarConstraint,
        maxTokens: Int,
        vocabSize: Int,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKvStart: Int = 0,
        completionReserve: Int = 64,
        hardReserve: Int = 0,
        closingBias: MLXArray? = nil,
        whitespaceBias: MLXArray? = nil,
        whitespaceTokenIDs: Set<Int> = [],
        diagnosticLog: Bool = false,
        cache: [KVCache]? = nil,
        onTokenCommitted: ((Int) -> Void)? = nil,
        emit: (String) -> Bool
    ) throws -> RunResult {
        let model = context.model
        let initialCache = cache ?? model.newCache(parameters: nil)

        // Build EOS token set
        let stopTokenIDs = Self.buildStopTokenIDs(
            tokenizer: context.tokenizer,
            configuration: context.configuration
        )

        // Prefill prompt and get first set of logits
        let prefill = try prefillAndGetLogits(input: input, model: model, cache: initialCache)

        let detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        var grammarStopped = false
        let whitespaceTracker = WhitespaceRunTracker(whitespaceTokenIDs: whitespaceTokenIDs)

        // Pre-compute bias arrays used in the zone policy.
        //
        // eosPenalty: -10000 at each EOS/stop position. Used in the normal
        // zone to prevent premature EOS at structurally incomplete states,
        // and in the hard zone alongside closing token penalties.
        //
        // The EOS penalty is NOT applied in the soft zone. The grammar mask
        // ensures structural validity (EOS only appears when JSON is
        // structurally complete). Removing the penalty lets the model stop
        // when output is structurally valid but semantically short, which
        // is acceptable near the budget limit.
        let eosPenalty: MLXArray? =
            if let bias = closingBias {
                {
                    let biasLen = bias.shape[0]
                    var penalty = [Float32](repeating: 0.0, count: biasLen)
                    for eos in stopTokenIDs where eos >= 0 && eos < biasLen {
                        penalty[eos] = logitRejectionPenalty
                    }
                    return MLXArray(penalty)
                }()
            } else {
                nil
            }

        let clock = ContinuousClock()
        let startInstant = clock.now

        // Logit dimension is constant across the generation; capture once so the
        // grammar-mask array can be built outside applyMaskAndSample.
        let logitDim = prefill.logits.shape[prefill.logits.ndim - 1]

        // Pre-compute the first mask + its sample array (no overlap possible for
        // the first iteration). Subsequent arrays are built in the overlap window.
        let (initialMask, initialMaskArray) = try computeMaskAndArray(
            constraint: constraint, vocabSize: vocabSize, logitDim: logitDim)

        var state = LoopState(
            cache: initialCache,
            modelState: prefill.modelState,
            logits: prefill.logits,
            mask: initialMask,
            maskArray: initialMaskArray,
            detokenizer: detokenizer,
            tokenCount: 0,
            accumulatedText: "",
            whitespaceTracker: whitespaceTracker
        )

        while state.tokenCount < maxTokens {
            // Cooperative cancellation: exit promptly when the enclosing Task
            // is cancelled (e.g. test timeout or user-initiated cancellation).
            try Task.checkCancellation()

            // Diagnostic: capture mask state before sampling
            if diagnosticLog {
                logMaskSnapshot(state: state, vocabSize: vocabSize)
            }

            // Check stop from the pre-computed mask
            if state.mask.isTerminated {
                logStopReason(
                    diagnosticLog: diagnosticLog, "mask.isTerminated", tokenCount: state.tokenCount)
                grammarStopped = true
                break
            }

            let tokenID = applyBiasAndSample(
                state: &state,
                closingBias: closingBias,
                whitespaceBias: whitespaceBias,
                eosPenalty: eosPenalty,
                maxTokens: maxTokens,
                completionReserve: completionReserve,
                hardReserve: hardReserve
            )

            // Check EOS only when the grammar exposed a real mask
            // (`needsApply == true`). When `false` the grammar is in an
            // unconditional splice: the sampled value is irrelevant
            // because commitToken will surface the forced tokens.
            // Checking for EOS here would cause a spurious stop -- the
            // model's raw logits might have EOS as the highest value
            // even though the grammar has NOT accepted the output.
            //
            // When `needsApply` IS true: if the grammar mask allowed
            // EOS (bit = 1), the grammar considers the output
            // acceptable. If the mask did NOT allow EOS,
            // `applyMaskAndSample` set it to -inf, so argmax would not
            // have selected it.
            if state.mask.needsApply,
                tokenID == context.tokenizer.unknownTokenId || stopTokenIDs.contains(tokenID)
            {
                logStopReason(
                    diagnosticLog: diagnosticLog, "EOS/unk tokenID=\(tokenID)",
                    tokenCount: state.tokenCount)
                grammarStopped = true
                break
            }

            // Commit to grammar
            let commitResult = try constraint.commitToken(Int32(tokenID))

            // Yield the sampled token
            if emitToken(tokenID, state: &state, emit: emit) { break }

            // Periodic progress logging (once per main loop iteration, not per FF token)
            if state.tokenCount % 50 == 0 {
                logProgress(state: state, clock: clock, startInstant: startInstant)
            }

            if commitResult.isTerminated {
                logStopReason(
                    diagnosticLog: diagnosticLog, "commitResult.isTerminated",
                    tokenCount: state.tokenCount)
                grammarStopped = true
                break
            }

            // Handle fast-forward tokens. CommitResult.tokens carries
            // ONLY the jump-forward ids (the sampled token is not echoed
            // back by xgrammar), so use the array directly.
            let ffTokens: [Int32] = commitResult.tokens

            let shouldStop: Bool
            if !ffTokens.isEmpty {
                shouldStop = try processFastForwardTokens(
                    tokenID,
                    ffTokens,
                    state: &state,
                    maxTokens: maxTokens,
                    model: model,
                    onTokenCommitted: onTokenCommitted,
                    kvBits: kvBits,
                    kvGroupSize: kvGroupSize,
                    quantizedKvStart: quantizedKvStart,
                    constraint: constraint,
                    vocabSize: vocabSize,
                    logitDim: logitDim,
                    emit: emit
                )
            } else {
                try advanceSingleSampledToken(
                    tokenID,
                    state: &state,
                    model: model,
                    onTokenCommitted: onTokenCommitted,
                    kvBits: kvBits,
                    kvGroupSize: kvGroupSize,
                    quantizedKvStart: quantizedKvStart,
                    constraint: constraint,
                    vocabSize: vocabSize,
                    logitDim: logitDim
                )
                shouldStop = false
            }
            if shouldStop { break }
        }

        // Log final generation stats
        let totalElapsed = clock.now - startInstant
        let totalMs = durationToMilliseconds(totalElapsed)
        logger.info("\(logPrefix) done tokens=\(state.tokenCount) elapsed=\(totalMs)ms")

        // Flush any xgrammar warnings (limit exceedances, parser state)
        if diagnosticLog, let logs = constraint.flushLogs() {
            logger.warning("\(logPrefix) xgrammar logs:\n\(logs)")
        }

        // If we exhausted maxTokens without the grammar reaching a stop state,
        // the output is structurally incomplete (e.g., truncated JSON).
        if !grammarStopped && state.tokenCount >= maxTokens {
            throw GuidedGenerationError.incompleteOutput
        }

        return RunResult(tokenCount: state.tokenCount, cache: state.cache)
    }

    // MARK: - Loop state

    /// Mutable per-generation state threaded through `run()`'s helper
    /// functions: the model's running KV cache and forward-pass state, the
    /// freshly computed grammar mask and its sample array, the streaming
    /// detokenizer and accumulated output text, the committed-token count,
    /// and the whitespace-run tracker. Bundling these avoids an
    /// ever-growing parameter list on `applyBiasAndSample`,
    /// `processFastForwardTokens`, and `advanceSingleSampledToken` as
    /// `run()`'s loop advances.
    private struct LoopState {
        var cache: [KVCache]
        var modelState: LMOutput.State?
        var logits: MLXArray
        var mask: MaskResult
        var maskArray: MLXArray?
        var detokenizer: NaiveStreamingDetokenizer
        var tokenCount: Int
        var accumulatedText: String
        var whitespaceTracker: WhitespaceRunTracker
    }

    /// Prepares the cache for `input` and runs the initial forward pass,
    /// returning the first logits and model state `run()`'s loop needs
    /// before entering the mask/sample cycle. Mirrors `model.prepare`'s own
    /// `PrepareResult` cases: `.tokens` means the cache didn't already hold
    /// a matching prefix and the tokens must be run through the model;
    /// `.logits` means `prepare` already produced the initial step (e.g.
    /// reusing a previous round's cache).
    ///
    /// - Parameters:
    ///   - input: Prepared model input (prompt already tokenized).
    ///   - model: The language model to prepare and, if needed, run the
    ///     initial forward pass on.
    ///   - cache: The KV cache to prepare against; mutated in place by
    ///     `model.prepare`/`model.callAsFunction` as usual for `KVCache`'s
    ///     reference semantics.
    /// - Returns: The first logits and model state for `run()`'s loop to
    ///   begin sampling from.
    /// - Throws: Rethrows any error `model.prepare(_:cache:state:windowSize:)`
    ///   throws while preparing `input` against `cache`.
    private static func prefillAndGetLogits(
        input: LMInput, model: any LanguageModel, cache: [KVCache]
    ) throws -> (logits: MLXArray, modelState: LMOutput.State?) {
        switch try model.prepare(input, cache: cache, state: nil, windowSize: 512) {
        case .tokens(let tokens):
            let result = model(tokens[text: .newAxis], cache: cache, state: nil)
            return (result.logits, result.state)

        case .logits(let result):
            return (result.logits, result.state)
        }
    }

    /// Computes the zone-policy bias (normal/soft/hard budget zones plus
    /// optional whitespace suppression), samples the next token against the
    /// current mask, and records it in the whitespace-run tracker.
    /// Extracting this out of `run()`'s main loop collapses the hard-zone
    /// branch -- previously nested 4-5 levels deep -- into a single
    /// top-level call.
    ///
    /// - Parameters:
    ///   - state: Loop state; reads `logits`/`maskArray`/`mask` and updates
    ///     `whitespaceTracker` with the sampled token.
    ///   - closingBias: Pre-computed logit bias array favoring closing
    ///     tokens; `nil` disables soft/hard zone biasing entirely.
    ///   - whitespaceBias: Pre-computed negative logit bias array penalizing
    ///     whitespace-only tokens; `nil` disables whitespace suppression.
    ///   - eosPenalty: Precomputed EOS-position penalty applied on top of
    ///     the forced-closing bias in the hard zone; `nil` when
    ///     `closingBias` is `nil`.
    ///   - maxTokens: Generation budget; combined with `completionReserve`/
    ///     `hardReserve` to determine the current zone.
    ///   - completionReserve: Number of tokens before `maxTokens` at which
    ///     the soft closing zone activates.
    ///   - hardReserve: Number of tokens before `maxTokens` at which the
    ///     hard closing zone activates; `0` disables it.
    /// - Returns: The sampled token ID.
    private static func applyBiasAndSample(
        state: inout LoopState,
        closingBias: MLXArray?,
        whitespaceBias: MLXArray?,
        eosPenalty: MLXArray?,
        maxTokens: Int,
        completionReserve: Int,
        hardReserve: Int
    ) -> Int {
        // Zone policy for budget management:
        //
        //   Normal zone (tokenCount < maxTokens - completionReserve):
        //     No bias. The grammar mask already gates EOS on
        //     structural validity, so primitive schemas (e.g.
        //     `{"type": "integer"}`, where the grammar allows EOS
        //     after one digit) can stop naturally after one token,
        //     without a bias layer on top.
        //
        //   Soft zone (completionReserve .. hardReserve tokens left):
        //     Closing bias only (+200 EOS, +100 closing tokens). No EOS
        //     penalty. The grammar mask ensures EOS only appears when JSON
        //     is structurally valid, so removing the penalty lets the model
        //     stop naturally. May produce shorter output for unbounded
        //     schemas, which is acceptable this close to the budget.
        //
        //   Hard zone (hardReserve tokens left):
        //     Penalize all non-closing tokens (-10000) AND EOS (-10000).
        //     Forces the model to select closing tokens (}, ], ", digits)
        //     that build up JSON structure. The grammar reaches a natural
        //     stop state when JSON is complete. EOS is penalized because
        //     the grammar may allow it at intermediate valid states before
        //     all required fields are present.
        //
        // Only applied when the grammar's mask carries exclusions
        // (`needsApply == true`). When false, the grammar is in an
        // unconditional splice (all tokens forced by FF). Applying
        // bias without a grammar mask can cause EOS selection before
        // the grammar has accepted the output.
        var activeBias: MLXArray? = nil
        if state.mask.needsApply {
            if let bias = closingBias {
                if hardReserve > 0 && state.tokenCount >= maxTokens - hardReserve {
                    // Hard zone: force closing tokens, suppress everything else.
                    activeBias = hardZoneBias(bias: bias, eosPenalty: eosPenalty)
                } else if state.tokenCount >= maxTokens - completionReserve {
                    // Soft zone: nudge toward closing tokens, no EOS penalty.
                    activeBias = bias
                }
                // Normal zone: no bias. Grammar mask + natural EOS handle
                // termination. Intentionally leaves `activeBias == nil`.
            }
            if let wsBias = whitespaceBias, state.whitespaceTracker.isActive {
                activeBias = activeBias.map { $0 + wsBias } ?? wsBias
            }
        }

        let token = applyMaskAndSample(
            logits: state.logits,
            maskArray: state.maskArray,
            closingBias: activeBias
        )
        let tokenID = Int(token)

        // Track the sampled token for whitespace run detection.
        // Fast-forward tokens are NOT tracked (they are grammar-forced).
        if whitespaceBias != nil {
            _ = state.whitespaceTracker.record(tokenID: tokenID)
        }

        return tokenID
    }

    /// Computes the hard-zone logit bias: forces closing tokens (zeroing
    /// their bias) and suppresses every other token with
    /// `logitRejectionPenalty`, additionally penalizing EOS positions when
    /// `eosPenalty` is available. Extracted from `applyBiasAndSample` to
    /// keep the hard-zone branch from nesting a 4th level deep.
    ///
    /// - Parameters:
    ///   - bias: The closing-token bias array (positive at closing tokens).
    ///   - eosPenalty: Precomputed EOS-position penalty; added on top of the
    ///     forced-closing bias when present.
    /// - Returns: The hard-zone bias array to apply to logits.
    private static func hardZoneBias(bias: MLXArray, eosPenalty: MLXArray?) -> MLXArray {
        var hardBias = which(bias .> 0, Float32(0.0), logitRejectionPenalty)
        if let eosPenalty {
            hardBias = hardBias + eosPenalty
        }
        return hardBias
    }

    /// Detokenizes `tokenID`, appends any resulting text to the accumulated
    /// output, and forwards it through `emit`. Shared by the sampled-token
    /// yield in `run()`'s main loop and the fast-forward emission loop in
    /// `processFastForwardTokens`, so both honor the same stop contract.
    ///
    /// - Parameters:
    ///   - tokenID: The token id to detokenize and emit.
    ///   - state: Loop state; mutates `detokenizer`/`accumulatedText` and,
    ///     on the non-stopping path, `tokenCount`.
    ///   - emit: Callback for the detokenized text delta; returning `false`
    ///     requests a stop.
    /// - Returns: `true` if `emit` returned `false` (caller must stop
    ///   generation immediately); `false` otherwise. `state.tokenCount` is
    ///   incremented only on the non-stopping path, matching the token
    ///   accounting the two original call sites both relied on.
    private static func emitToken(
        _ tokenID: Int, state: inout LoopState, emit: (String) -> Bool
    ) -> Bool {
        state.detokenizer.append(token: tokenID)
        if let text = state.detokenizer.next() {
            state.accumulatedText += text
            if !emit(text) {
                return true
            }
        }
        state.tokenCount += 1
        return false
    }

    /// Handles a non-empty fast-forward batch after committing the sampled
    /// token: yields each FF token's text through `emit` (honoring
    /// `maxTokens` and the caller's stop signal), feeds the sampled token
    /// AND the FF tokens through the model one at a time to advance the KV
    /// cache, and finally quantizes the cache and computes the next mask in
    /// the CPU/GPU overlap window. Extracting this out of `run()`'s main
    /// loop collapses two blocks that were each nested 4-5 levels deep (the
    /// FF-token emission loop and the FF-token batch-finalization loop)
    /// into a single top-level call.
    ///
    /// - Parameters:
    ///   - tokenID: The token just sampled and committed to the grammar
    ///     this iteration (the one whose commit produced `ffTokens`).
    ///     `CommitResult.tokens` never echoes this id back (see
    ///     `GrammarConstraint.commitToken`'s doc comment), so unlike the
    ///     non-FF path (`advanceSingleSampledToken`), nothing upstream of
    ///     this call has fed it through the model yet. It must get its own
    ///     forward pass here -- feeding `ffTokens` directly without it
    ///     would silently skip `tokenID`'s KV-cache entry, leaving every
    ///     later position's attention unable to see a token that was
    ///     nonetheless emitted and accepted by the grammar.
    ///   - ffTokens: The jump-forward token ids from `CommitResult.tokens`
    ///     (never includes the sampled token that triggered them).
    ///   - state: Loop state; mutates `detokenizer`/`accumulatedText`/
    ///     `tokenCount` (via `emitToken`) and `cache`/`modelState`/`logits`/
    ///     `mask`/`maskArray`.
    ///   - maxTokens: Generation budget; FF emission stops (without
    ///     feeding `tokenID` or the FF tokens to the model) once reached.
    ///   - model: The language model to feed `tokenID` and `ffTokens`
    ///     through, one token at a time, to advance the KV cache.
    ///   - onTokenCommitted: Invoked, in order, with each token id actually
    ///     fed through `model` this call (`tokenID` then every `ffTokens`
    ///     entry); `nil` is a no-op.
    ///   - kvBits: Bit width for KV-cache quantization, or `nil` to disable
    ///     it for this call.
    ///   - kvGroupSize: Group size used when quantizing the KV cache.
    ///   - quantizedKvStart: Token offset at which KV-cache quantization
    ///     begins.
    ///   - constraint: The grammar constraint whose next mask is computed
    ///     after the FF batch is fed through the model.
    ///   - vocabSize: The grammar's vocabulary size, used to interpret the
    ///     mask bitmask.
    ///   - logitDim: The model's logit dimension, used to size the mask
    ///     array.
    ///   - emit: Callback for each text delta; returning `false` requests a
    ///     stop.
    /// - Returns: `true` if the caller's `emit` requested a stop (or the
    ///   budget was exhausted mid-batch) and `run`'s main loop must break;
    ///   `false` to continue.
    /// - Throws: Rethrows whatever `updateMaskAfterForwardPass` throws
    ///   (ultimately `GrammarError.maskComputationFailed` from
    ///   `constraint.computeMask()`) while computing the next mask after
    ///   the FF batch.
    private static func processFastForwardTokens(
        _ tokenID: Int,
        _ ffTokens: [Int32],
        state: inout LoopState,
        maxTokens: Int,
        model: any LanguageModel,
        onTokenCommitted: ((Int) -> Void)?,
        kvBits: Int?,
        kvGroupSize: Int,
        quantizedKvStart: Int,
        constraint: GrammarConstraint,
        vocabSize: Int,
        logitDim: Int,
        emit: (String) -> Bool
    ) throws -> Bool {
        // Yield FF tokens to output. The caller's `emit` stop signal
        // (`emit(text) == false`) must halt generation immediately, just
        // like on the sampled-token path in `run()`. A bare early-return
        // here would only stop this batch, leaving `run`'s main loop to
        // run another full iteration -- wasting GPU work and violating
        // the caller's stop contract. Propagate through the `Bool`
        // return and let the caller break its own `while`. (`tokenID`
        // itself was already emitted by `run()`'s main loop before this
        // function was called, so only `ffTokens` need emitting here.)
        for ffToken in ffTokens {
            if state.tokenCount >= maxTokens {
                return true
            }
            if emitToken(Int(ffToken), state: &state, emit: emit) {
                return true
            }
        }

        // Feed the sampled token through the model FIRST, before any FF
        // token. It occupies the KV-cache position immediately after the
        // prior forward pass and immediately before `ffTokens[0]`'s -- the
        // same single-token forward pass `advanceSingleSampledToken` does
        // on the non-FF path. Its logits are irrelevant here (the grammar,
        // not a sample from these logits, forces every `ffToken` that
        // follows), so only the last FF token's logits are kept below,
        // exactly as before this token was added to the batch.
        _ = feedTokenThroughModel(
            tokenID, state: &state, model: model, onTokenCommitted: onTokenCommitted)

        // Process FF tokens one at a time to update KV cache.
        // Batching (T_q > 1 with populated cache) triggers an MLX
        // bug: scaledDotProductAttention in .causal mode creates a
        // mask of shape (T_q, T_q) instead of (T_q, T_kv), causing
        // a broadcast failure on models with global attention layers
        // (e.g., Gemma 3). Single-token passes (T_q=1) use the
        // optimized Metal kernel and skip the mask entirely.
        for (i, ffToken) in ffTokens.enumerated() {
            let logits = feedTokenThroughModel(
                Int(ffToken), state: &state, model: model, onTokenCommitted: onTokenCommitted)
            // Only need logits from the last FF token
            if i == ffTokens.count - 1 {
                state.logits = logits
            }
        }

        try updateMaskAfterForwardPass(
            state: &state, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKvStart: quantizedKvStart, constraint: constraint,
            vocabSize: vocabSize, logitDim: logitDim)

        return false
    }

    /// Returns `cache` wrapped as an optional for the model's `cache:`
    /// parameter, or `nil` when `cache` is empty (signaling "no prior state"
    /// to the model on the very first forward pass). Shared by the
    /// fast-forward and single-sampled-token forward-pass call sites, which
    /// both need this identical empty-to-nil translation.
    ///
    /// - Parameter cache: The KV cache to translate.
    /// - Returns: `cache` unchanged, or `nil` if `cache` is empty.
    private static func cacheOrNil(_ cache: [KVCache]) -> [KVCache]? {
        cache.isEmpty ? nil : cache
    }

    /// Feeds a single token through the model to advance the KV cache at
    /// `state`'s current position, updating `state.modelState` and firing
    /// `onTokenCommitted`.
    ///
    /// Every call already computes fresh logits as part of the forward
    /// pass -- this always returns them and lets each call site decide
    /// whether to keep them (e.g. only the last token of an FF batch needs
    /// its logits kept for the next sampling step; a token fed only to
    /// populate the cache can discard them).
    ///
    /// - Parameters:
    ///   - tokenID: The token id to feed through `model`.
    ///   - state: Loop state; reads `cache`/`modelState` and writes the
    ///     updated `modelState` back.
    ///   - model: The language model to run the single-token forward pass
    ///     on.
    ///   - onTokenCommitted: Invoked with `tokenID` after the forward pass;
    ///     `nil` is a no-op.
    /// - Returns: The forward pass's output logits.
    private static func feedTokenThroughModel(
        _ tokenID: Int,
        state: inout LoopState,
        model: any LanguageModel,
        onTokenCommitted: ((Int) -> Void)?
    ) -> MLXArray {
        let input = LMInput.Text(tokens: MLXArray([Int32(tokenID)]))
        let result = model(
            input[text: .newAxis],
            cache: cacheOrNil(state.cache),
            state: state.modelState
        )
        state.modelState = result.state
        onTokenCommitted?(tokenID)
        return result.logits
    }

    /// Feeds the sampled token through the model (no fast-forward was
    /// emitted for it) to advance the KV cache and produce the next
    /// logits, then quantizes the cache and computes the next mask in the
    /// CPU/GPU overlap window. The fast-forward counterpart is
    /// `processFastForwardTokens`; kept separate since this path has none
    /// of the batch/emission concerns that FF handling does.
    ///
    /// - Parameters:
    ///   - tokenID: The token sampled and committed to the grammar this
    ///     iteration (no fast-forward batch was produced for it).
    ///   - state: Loop state; mutated the same way `processFastForwardTokens`
    ///     mutates it (`cache`/`modelState`/`logits`/`mask`/`maskArray`).
    ///   - model: The language model to feed `tokenID` through to advance
    ///     the KV cache.
    ///   - onTokenCommitted: Invoked with `tokenID` after the forward pass;
    ///     `nil` is a no-op.
    ///   - kvBits: Bit width for KV-cache quantization, or `nil` to disable
    ///     it for this call.
    ///   - kvGroupSize: Group size used when quantizing the KV cache.
    ///   - quantizedKvStart: Token offset at which KV-cache quantization
    ///     begins.
    ///   - constraint: The grammar constraint whose next mask is computed
    ///     after the forward pass.
    ///   - vocabSize: The grammar's vocabulary size, used to interpret the
    ///     mask bitmask.
    ///   - logitDim: The model's logit dimension, used to size the mask
    ///     array.
    /// - Throws: Rethrows whatever `updateMaskAfterForwardPass` throws
    ///   (ultimately `GrammarError.maskComputationFailed` from
    ///   `constraint.computeMask()`) while computing the next mask after
    ///   the forward pass.
    private static func advanceSingleSampledToken(
        _ tokenID: Int,
        state: inout LoopState,
        model: any LanguageModel,
        onTokenCommitted: ((Int) -> Void)?,
        kvBits: Int?,
        kvGroupSize: Int,
        quantizedKvStart: Int,
        constraint: GrammarConstraint,
        vocabSize: Int,
        logitDim: Int
    ) throws {
        // Normal single-token forward pass (lazy)
        state.logits = feedTokenThroughModel(
            tokenID, state: &state, model: model, onTokenCommitted: onTokenCommitted)

        try updateMaskAfterForwardPass(
            state: &state, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKvStart: quantizedKvStart, constraint: constraint,
            vocabSize: vocabSize, logitDim: logitDim)
    }

    /// Logs a diagnostic mask-state snapshot for the token about to be
    /// sampled. Callers gate this on `diagnosticLog` themselves.
    ///
    /// - Parameters:
    ///   - state: Loop state; reads `mask` and `tokenCount`.
    ///   - vocabSize: The grammar's vocabulary size, used to interpret the
    ///     mask bitmask.
    private static func logMaskSnapshot(state: LoopState, vocabSize: Int) {
        let snapshot = state.mask.mask.withUnsafeBufferPointer { buffer -> MaskSnapshot in
            let ptr: UnsafePointer<UInt32>? =
                if state.mask.needsApply, let base = buffer.baseAddress {
                    UnsafeRawPointer(base).assumingMemoryBound(to: UInt32.self)
                } else {
                    // Either the grammar needs no mask applied, or the
                    // mask buffer is empty (zero bitmask words) -- either
                    // way there is no valid bitmask to snapshot.
                    nil
                }
            return MaskSnapshot.capture(
                sampleMask: ptr,
                vocabSize: vocabSize,
                tokenIndex: state.tokenCount,
                isStop: state.mask.isTerminated
            )
        }
        logger.info("\(snapshot.summary())")
    }

    /// Logs a "why did the loop stop" diagnostic line when `diagnosticLog`
    /// is enabled; no-op otherwise. Centralizes the `logPrefix` +
    /// `stopReasonFragment` template shared by every stop-reason log site.
    ///
    /// - Parameters:
    ///   - diagnosticLog: When `false`, this call is a no-op.
    ///   - reason: The stop-reason fragment to log (e.g.
    ///     `"mask.isTerminated"`).
    ///   - tokenCount: The token count at which the loop stopped.
    private static func logStopReason(diagnosticLog: Bool, _ reason: String, tokenCount: Int) {
        guard diagnosticLog else { return }
        logger.info("\(logPrefix)\(stopReasonFragment)\(reason) at token \(tokenCount)")
    }

    /// Logs periodic generation progress (once per main loop iteration,
    /// not per FF token).
    ///
    /// - Parameters:
    ///   - state: Loop state; reads `tokenCount` and `accumulatedText`.
    ///   - clock: The clock `startInstant` was captured from, used to
    ///     compute elapsed time.
    ///   - startInstant: The instant generation began.
    private static func logProgress(
        state: LoopState, clock: ContinuousClock, startInstant: ContinuousClock.Instant
    ) {
        let elapsed = clock.now - startInstant
        let ms = durationToMilliseconds(elapsed)
        let prefix = String(state.accumulatedText.prefix(200))
        logger.info("\(logPrefix) token=\(state.tokenCount) elapsed=\(ms)ms text=\(prefix)")
    }

    /// Converts an elapsed `Duration` since some start instant into total
    /// milliseconds, combining whole seconds with the sub-second attosecond
    /// remainder. Shared by `run()`'s final-stats log and `logProgress()`'s
    /// periodic log, both of which need the same elapsed-time-to-milliseconds
    /// conversion.
    ///
    /// - Parameter duration: The elapsed duration to convert.
    /// - Returns: The equivalent duration in whole milliseconds.
    private static func durationToMilliseconds(_ duration: Duration) -> Int64 {
        duration.components.seconds * millisecondsPerSecond
            + duration.components.attoseconds / attosecondsPerMillisecond
    }

    /// Quantizes the KV cache and computes the next grammar mask in the
    /// CPU/GPU overlap window, updating `state.cache`/`state.mask`/
    /// `state.maskArray` in place. Thin wrapper around
    /// `advanceMaskOnCpuWhileGpuRuns` that threads `LoopState` through, so
    /// `processFastForwardTokens` and `advanceSingleSampledToken` -- which
    /// both need this identical post-forward-pass sequence -- can call it
    /// with a single line instead of duplicating the tuple-destructuring
    /// call.
    ///
    /// - Parameters:
    ///   - state: Loop state; mutates `cache`/`mask`/`maskArray`. Reads
    ///     `logits` (the just-produced forward-pass output).
    ///   - kvBits: Bit width for KV-cache quantization, or `nil` to disable
    ///     it for this call.
    ///   - kvGroupSize: Group size used when quantizing the KV cache.
    ///   - quantizedKvStart: Token offset at which KV-cache quantization
    ///     begins.
    ///   - constraint: The grammar constraint to compute the next mask
    ///     from.
    ///   - vocabSize: The grammar's vocabulary size, used to interpret the
    ///     mask bitmask.
    ///   - logitDim: The model's logit dimension, used to size the mask
    ///     array.
    /// - Throws: Rethrows whatever `advanceMaskOnCpuWhileGpuRuns` throws
    ///   (ultimately `GrammarError.maskComputationFailed` from
    ///   `constraint.computeMask()`).
    private static func updateMaskAfterForwardPass(
        state: inout LoopState,
        kvBits: Int?,
        kvGroupSize: Int,
        quantizedKvStart: Int,
        constraint: GrammarConstraint,
        vocabSize: Int,
        logitDim: Int
    ) throws {
        (state.mask, state.maskArray) = try Self.advanceMaskOnCpuWhileGpuRuns(
            logits: state.logits, cache: &state.cache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKvStart: quantizedKvStart, constraint: constraint,
            vocabSize: vocabSize, logitDim: logitDim)
    }

    /// Quantizes the KV cache (a no-op unless `kvBits` is set), then computes
    /// the next grammar mask and its sample array while the GPU finishes the
    /// forward pass that just ran -- the exact sequence `run`'s main loop
    /// needs after every per-token model call, whether that call fed a
    /// single sampled token or the last token of a fast-forward batch.
    /// Extracted because both branches ran this identical 8-line sequence.
    ///
    /// - Parameters:
    ///   - logits: The forward pass's output logits. Used only to drive the
    ///     GPU overlap (`asyncEval` kicks it off, `eval` waits for it).
    ///   - cache: The KV cache to quantize in place, matching `run`'s own
    ///     `cache:` parameter semantics.
    ///   - kvBits: Bit width for KV-cache quantization, or nil to disable.
    ///   - kvGroupSize: Group size for KV-cache quantization.
    ///   - quantizedKvStart: Token offset at which quantization begins.
    ///   - constraint: The grammar constraint to compute the next mask from.
    ///   - vocabSize: The grammar's vocabulary size.
    ///   - logitDim: The model's logit dimension.
    /// - Returns: The freshly computed mask and its prebuilt additive sample
    ///   array (see `buildMaskArray`).
    /// - Throws: Rethrows whatever `computeMaskAndArray` throws (ultimately
    ///   `GrammarError.maskComputationFailed` from
    ///   `constraint.computeMask()`).
    private static func advanceMaskOnCpuWhileGpuRuns(
        logits: MLXArray,
        cache: inout [KVCache],
        kvBits: Int?,
        kvGroupSize: Int,
        quantizedKvStart: Int,
        constraint: GrammarConstraint,
        vocabSize: Int,
        logitDim: Int
    ) throws -> (MaskResult, MLXArray?) {
        maybeQuantizeKVCache(
            cache: &cache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKvStart)

        // Kick off GPU computation asynchronously
        asyncEval(logits)

        // Overlap: compute the next mask AND build its sample array on the
        // CPU while the GPU runs the forward pass.
        let (mask, maskArray) = try computeMaskAndArray(
            constraint: constraint, vocabSize: vocabSize, logitDim: logitDim)

        // Wait for GPU to finish (may already be done)
        eval(logits)

        return (mask, maskArray)
    }

    /// Computes the grammar constraint's next mask and its prebuilt additive
    /// sample array in one call. Shared by the pre-loop initial-mask setup in
    /// `run()` and by `advanceMaskOnCpuWhileGpuRuns`'s CPU/GPU overlap window,
    /// both of which need this identical compute-then-build pair.
    ///
    /// - Parameters:
    ///   - constraint: The grammar constraint to compute the next mask from.
    ///   - vocabSize: The grammar's vocabulary size.
    ///   - logitDim: The model's logit dimension.
    /// - Returns: The freshly computed mask and its prebuilt additive sample
    ///   array (see `buildMaskArray`).
    /// - Throws: Rethrows `GrammarError.maskComputationFailed` if
    ///   `constraint.computeMask()` fails to compute the next mask.
    private static func computeMaskAndArray(
        constraint: GrammarConstraint, vocabSize: Int, logitDim: Int
    ) throws -> (MaskResult, MLXArray?) {
        let mask = try constraint.computeMask()
        let maskArray = buildMaskArray(for: mask, vocabSize: vocabSize, logitDim: logitDim)
        return (mask, maskArray)
    }

    /// The outcome of a `run` call that reached a stop state: everything a
    /// caller needs to report usage and participate in KV-cache reuse
    /// (e.g. `PromptCache`) on a subsequent call.
    ///
    /// Not `Sendable`: `KVCache` (a plain class hierarchy) isn't
    /// `Sendable`, matching `run`'s own `cache:` parameter. Callers that
    /// need to carry this across an actor boundary should box it the same
    /// way `PromptCache` boxes `[KVCache]` elsewhere (see `SendableBox`).
    public struct RunResult {
        /// Total tokens generated (including FF tokens) -- the same
        /// quantity `run` used to return bare.
        public let tokenCount: Int
        /// The cache array as it stands after this call. Always use this,
        /// never the caller's own pre-call `cache:` argument, as the
        /// authoritative post-call state -- see `run`'s `cache:` parameter
        /// doc for why the two can diverge once `kvBits` is set.
        public let cache: [KVCache]
    }

    // MARK: - Internal (visible for testing)

    /// Build the set of token ids that terminate generation.
    ///
    /// Pulls from three sources, all carried by the `ModelConfiguration` and
    /// tokenizer (all required for chat-tuned models to stop correctly):
    ///
    /// 1. `configuration.eosTokenIds` — loaded from `config.json` /
    ///    `generation_config.json` at model-load time. Chat models like
    ///    Gemma 3 ship `eos_token_id` as an array (e.g. `[1, 106]` for
    ///    `<eos>` + `<end_of_turn>`); this source is the only way to pick
    ///    up the turn-ender when the tokenizer's primary EOS is the
    ///    completion EOS.
    /// 2. `tokenizer.eosTokenId` — the tokenizer's single primary EOS.
    /// 3. `configuration.extraEOSTokens` — hardcoded-by-token-string
    ///    additions from registry entries (e.g. `["<end_of_turn>"]` on
    ///    some Gemma variants in `LLMModelFactory`). Callers needing extra
    ///    stop tokens add them here (via the model configuration), not as a
    ///    per-call argument.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer whose primary EOS id and extra-token
    ///     lookups (`convertTokenToId`) contribute to the result.
    ///   - configuration: The model configuration whose `eosTokenIds` and
    ///     `extraEOSTokens` contribute to the result.
    /// - Returns: The set of token ids that terminate generation.
    static func buildStopTokenIDs(
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration
    ) -> Set<Int> {
        var stopTokenIDs = Set(configuration.eosTokenIds)
        if let eos = tokenizer.eosTokenId {
            stopTokenIDs.insert(eos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIDs.insert(id)
            }
        }
        return stopTokenIDs
    }

    /// Apply a *prebuilt* grammar mask and optional bias to logits, then argmax.
    ///
    /// The mask is built once per token in the eval loop's overlap window (see
    /// `buildMaskArray`) and passed in here, so this runs no `bitmaskToMLXArray`
    /// work on the sampling critical path.
    ///
    /// - Parameters:
    ///   - logits: Raw model output logits (shape: [batch, seq, vocab]).
    ///   - maskArray: Prebuilt additive grammar mask (length == logit dim;
    ///     0.0 allowed, -inf disallowed), or nil when the grammar forces all
    ///     tokens (no mask to apply).
    ///   - closingBias: Optional logit bias favoring closing tokens. Applied
    ///     after the grammar mask so masked-out tokens remain at -inf.
    /// - Returns: The sampled token ID.
    static func applyMaskAndSample(
        logits rawLogits: MLXArray,
        maskArray: MLXArray?,
        closingBias: MLXArray? = nil
    ) -> UInt32 {
        // Extract last-position logits: [batch, seq, vocab] -> [vocab]
        var logits = rawLogits[0..., -1, 0...]

        if let maskArray {
            logits = logits + maskArray
        }

        if let bias = closingBias {
            let logitDim = logits.shape[logits.ndim - 1]
            let biasDim = bias.shape[0]
            if biasDim < logitDim {
                // Model logit dimension can exceed tokenizer vocab (padding/special tokens).
                // Pad with zeros so the bias has no effect on extra positions.
                let padding = MLXArray.zeros([logitDim - biasDim])
                logits = logits + concatenated([bias, padding])
            } else if biasDim > logitDim {
                // Tokenizer vocab can exceed model logit dimension (added special tokens
                // beyond the embedding size). Truncate to match.
                logits = logits + bias[0 ..< logitDim]
            } else {
                logits = logits + bias
            }
        }

        // Grammar-constrained generation samples greedily by construction.
        let sampled = argMax(logits, axis: -1)
        return sampled.item(UInt32.self)
    }

    /// Build the additive grammar-mask array for a freshly computed `MaskResult`,
    /// or nil when the grammar needs no mask applied (unconditional FF splice).
    ///
    /// Hoisted out of `applyMaskAndSample` so the loop can build it in the
    /// CPU/GPU overlap window (alongside `computeMask`) instead of on the
    /// critical path at sample time. `logitDim` is the model's logit dimension
    /// (constant across the generation); `vocabSize` is the grammar bitmask's
    /// valid-bit count.
    ///
    /// - Parameters:
    ///   - mask: The freshly computed grammar mask to convert.
    ///   - vocabSize: The grammar bitmask's valid-bit count.
    ///   - logitDim: The model's logit dimension; the returned array's
    ///     length.
    /// - Returns: The additive mask array, or `nil` when `mask.needsApply`
    ///   is `false` (the grammar forces all tokens, so no mask is needed).
    static func buildMaskArray(for mask: MaskResult, vocabSize: Int, logitDim: Int) -> MLXArray? {
        guard mask.needsApply else { return nil }
        return mask.mask.withUnsafeBufferPointer { buffer -> MLXArray? in
            guard let base = buffer.baseAddress else {
                // Empty mask buffer (zero bitmask words, e.g. a zero-vocab
                // tokenizer): no bits to apply, so behave as if no mask
                // needed applying rather than crash on a nil baseAddress.
                return nil
            }
            let ptr = UnsafeRawPointer(base).assumingMemoryBound(to: UInt32.self)
            return bitmaskToMLXArray(ptr, maskBitCount: vocabSize, totalCount: logitDim)
        }
    }

    /// Convert a packed bitmask (1 bit per token) to an MLXArray of floats.
    /// Allowed tokens get 0.0, disallowed tokens get -inf.
    ///
    /// `maskBitCount` is the number of valid bits in the mask (= tokenizer vocab
    /// size). `totalCount` is the model's logit dimension. When the tokenizer
    /// has more tokens than the model has logits (e.g. added special tokens
    /// beyond the embedding dimension), we only read `min(maskBitCount, totalCount)`
    /// bits. Positions beyond the mask are left at -inf.
    ///
    /// Internal (not private) so the mask-build microbenchmark can time it in
    /// isolation.
    ///
    /// - Parameters:
    ///   - maskPtr: Pointer to the packed bitmask's backing storage (1 bit
    ///     per token).
    ///   - maskBitCount: The number of valid bits in the mask (the
    ///     tokenizer vocab size).
    ///   - totalCount: The model's logit dimension; the returned array's
    ///     length.
    /// - Returns: An `MLXArray` of length `totalCount` with `0.0` at
    ///   allowed positions and `-.infinity` elsewhere.
    static func bitmaskToMLXArray(
        _ maskPtr: UnsafePointer<UInt32>,
        maskBitCount: Int,
        totalCount: Int
    ) -> MLXArray {
        var floats = [Float](repeating: -Float.infinity, count: totalCount)
        let readCount = min(maskBitCount, totalCount)
        for i in 0 ..< readCount {
            let word = maskPtr[i / 32]
            let bit = (word >> (UInt32(i) % 32)) & 1
            if bit == 1 {
                floats[i] = 0.0
            }
        }
        return MLXArray(floats)
    }
}
