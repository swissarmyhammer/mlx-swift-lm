// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

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
///
/// With an MTP drafter the loop speculates (see ``GuidedSpeculation``) and
/// writes the output plain decoding writes.
public enum GuidedGenerationLoop {

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
    ///   - kvCache: Typed KV-cache capacity and compression configuration. Prefer
    ///     this over the legacy scalar quantization parameters.
    ///   - kvBits: Bit width for KV-cache quantization, or nil to disable. When
    ///     set, the KV cache is quantized after each forward pass to reduce
    ///     memory use, mirroring the unconstrained `TokenIterator`. nil is a
    ///     no-op, so models that don't quantize are unaffected.
    ///   - kvGroupSize: Group size for KV-cache quantization (default 64, matching
    ///     `GenerateParameters`). Only used when `kvBits` is non-nil.
    ///   - quantizedKVStart: Token offset at which quantization begins (default 0).
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
    ///   - prefill: Prompt prefill parameters (step size, chunking strategy,
    ///     progress callback). Defaults to a 512-token step with balanced
    ///     chunking.
    ///   - emit: Callback for each text delta. Return `false` to stop.
    /// - Returns: Total number of tokens generated (including FF tokens).
    /// - Throws: `GuidedGenerationError.incompleteOutput` if maxTokens is
    ///   exhausted before the grammar reaches a stop state.
    ///   `GuidedGenerationError.prematureEOS` if the model emits EOS
    ///   before the grammar accepts.
    @discardableResult
    public static func run(
        input: LMInput,
        context: ModelContext,
        constraint: GrammarConstraint,
        maxTokens: Int,
        vocabSize: Int,
        kvCache: KVCacheConfiguration? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        completionReserve: Int = 64,
        hardReserve: Int = 0,
        closingBias: MLXArray? = nil,
        whitespaceBias: MLXArray? = nil,
        whitespaceTokenIDs: Set<Int> = [],
        diagnosticLog: Bool = false,
        prefill: PrefillParameters = .init(stepSize: PrefillParameters.defaultStepSize),
        emit: (String) -> Bool
    ) throws -> Int {
        try run(
            input: input, context: context, constraint: constraint, maxTokens: maxTokens,
            vocabSize: vocabSize, kvCache: kvCache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart, completionReserve: completionReserve,
            hardReserve: hardReserve, closingBias: closingBias, whitespaceBias: whitespaceBias,
            whitespaceTokenIDs: whitespaceTokenIDs, diagnosticLog: diagnosticLog,
            prefill: prefill, speculation: nil, emit: emit)
    }

    /// Runs the guided generation loop, speculating with the MTP drafter of `speculation`
    /// when the target can verify its rounds in place.
    ///
    /// The prefill feeds the drafter the hidden rows of each prompt chunk, so a new drafter
    /// state covers the whole prompt; this loop keeps no prompt cache. The output is the output
    /// of plain guided decoding: see ``GuidedSpeculation``. A drafter that cannot draft for the
    /// target, or a target that cannot verify in place, leaves the generation to plain decoding.
    ///
    /// - Parameters:
    ///   - speculation: The MTP drafter of the generation, or `nil` for plain decoding.
    ///   - Other parameters: As ``run(input:context:constraint:maxTokens:vocabSize:kvCache:kvBits:kvGroupSize:quantizedKVStart:completionReserve:hardReserve:closingBias:whitespaceBias:whitespaceTokenIDs:diagnosticLog:prefill:emit:)``
    ///     takes them.
    /// - Returns: Total number of tokens generated (including FF tokens).
    /// - Throws: As the plain overload throws.
    @discardableResult
    package static func run(
        input: LMInput,
        context: ModelContext,
        constraint: GrammarConstraint,
        maxTokens: Int,
        vocabSize: Int,
        kvCache: KVCacheConfiguration? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        completionReserve: Int = 64,
        hardReserve: Int = 0,
        closingBias: MLXArray? = nil,
        whitespaceBias: MLXArray? = nil,
        whitespaceTokenIDs: Set<Int> = [],
        diagnosticLog: Bool = false,
        prefill: PrefillParameters = .init(stepSize: PrefillParameters.defaultStepSize),
        speculation: MTPSpeculation?,
        emit: (String) -> Bool
    ) throws -> Int {
        let parameters = GenerateParameters(
            kvCache: kvCache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart)
        var target = try GuidedTarget(model: context.model, parameters: parameters)
        let drafter = GuidedSpeculation.drafter(of: speculation, for: target)
        let feed = drafter.flatMap { _ in
            speculation?.prefillFeed(
                target: context.model, parameters: parameters, cachePosition: 0,
                continuation: nil)
        }
        let prefilled = try target.prefill(
            input, parameters: prefill, drafterFeed: feed?.chunkHandler)
        let logitDim = prefilled.logits.shape[prefilled.logits.ndim - 1]
        let rules = GuidedDecisionRules(
            constraint: constraint, maxTokens: maxTokens, vocabSize: vocabSize,
            logitDim: logitDim,
            stopTokenIDs: buildStopTokenIDs(
                tokenizer: context.tokenizer, configuration: context.configuration),
            unknownTokenID: context.tokenizer.unknownTokenId,
            completionReserve: completionReserve, hardReserve: hardReserve,
            closingBias: closingBias, whitespaceBias: whitespaceBias,
            whitespaceTokenIDs: whitespaceTokenIDs)

        return try withoutActuallyEscaping(emit) { emit in
            var decoder = try GuidedDecoder(
                rules: rules, tokenizer: context.tokenizer, diagnosticLog: diagnosticLog,
                emit: emit)
            guard decoder.hasBudget, let first = try decoder.decide(from: prefilled.logits)
            else { return try decoder.finish() }
            if let speculation, let drafter, let feed,
                var speculating = GuidedSpeculation(
                    speculation: speculation, drafter: drafter, feed: feed, prefill: prefilled,
                    firstFeed: first, target: target)
            {
                try speculating.decode(from: first, decoder: &decoder, target: &target)
            } else {
                try decodePlainly(from: first, decoder: &decoder, target: &target)
            }
            return try decoder.finish()
        }
    }

    /// Decodes the rest of a generation one token per decision, from `pending`, the tokens the
    /// last decision feeds the target.
    ///
    /// Each forward runs on the GPU while the CPU computes the grammar mask of the next
    /// decision.
    ///
    /// - Parameters:
    ///   - pending: The tokens of the last decision, which no forward fed yet.
    ///   - decoder: The decisions and the output of the generation.
    ///   - target: The target.
    /// - Throws: `CancellationError`, or the error of the matcher.
    static func decodePlainly(
        from pending: [Int32], decoder: inout GuidedDecoder, target: inout GuidedTarget
    ) throws {
        var feed = pending
        while true {
            let logits = target.feedOneAtATime(feed)
            // Kick off GPU computation, and compute the next mask and its array on the CPU
            // while the GPU runs the forward pass.
            asyncEval(logits)
            try decoder.advance()
            eval(logits)
            guard decoder.hasBudget, let next = try decoder.decide(from: logits) else { return }
            feed = next
        }
    }

    // MARK: - Internal (visible for testing)

    /// The tokens a guided decision feeds the target: the decided token, or the tokens the
    /// grammar fast-forwarded after it when there are any.
    ///
    /// When a commit fast-forwards, the loop feeds the forced tokens alone and not the decided
    /// token before them, as it always has. An MTP round feeds the same tokens for each of its
    /// drafts, so its output stays the output of plain decoding.
    ///
    /// - Parameters:
    ///   - sampled: The decided token.
    ///   - fastForward: The tokens the grammar forced after it.
    /// - Returns: The tokens of the next forward, at least one.
    static func modelFeed(sampled: Int32, fastForward: [Int32]) -> [Int32] {
        fastForward.isEmpty ? [sampled] : fastForward
    }

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
        let logits = constrainedLogits(
            rawLogits[0..., -1, 0...], maskArray: maskArray, bias: closingBias)

        // Grammar-constrained generation samples greedily by construction.
        let sampled = argMax(logits, axis: -1)
        return sampled.item(UInt32.self)
    }

    /// `logits` with a prebuilt grammar mask and an optional bias added.
    ///
    /// - Parameters:
    ///   - logits: Logits of one or more rows, shape `[..., logit dim]`.
    ///   - maskArray: Prebuilt additive grammar mask, or nil when the grammar
    ///     forces all tokens.
    ///   - bias: Optional logit bias, added after the mask so masked-out tokens
    ///     remain at -inf.
    /// - Returns: The constrained logits.
    static func constrainedLogits(
        _ logits: MLXArray, maskArray: MLXArray?, bias: MLXArray?
    ) -> MLXArray {
        var logits = logits

        if let maskArray {
            logits = logits + maskArray
        }

        if let bias {
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
        return logits
    }

    /// Build the additive grammar-mask array for a freshly computed `MaskResult`,
    /// or nil when the grammar needs no mask applied (unconditional FF splice).
    ///
    /// Hoisted out of `applyMaskAndSample` so the loop can build it in the
    /// CPU/GPU overlap window (alongside `computeMask`) instead of on the
    /// critical path at sample time. `logitDim` is the model's logit dimension
    /// (constant across the generation); `vocabSize` is the grammar bitmask's
    /// valid-bit count.
    static func buildMaskArray(for mask: MaskResult, vocabSize: Int, logitDim: Int) -> MLXArray? {
        guard mask.needsApply else { return nil }
        return mask.mask.withUnsafeBufferPointer { buffer in
            let ptr = UnsafeRawPointer(buffer.baseAddress!).assumingMemoryBound(to: UInt32.self)
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
