// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// The decisions and the output of one guided generation: the position of the next decision,
/// the tokens counted so far, and the text emitted from them.
///
/// Plain decoding decides each token here from the target's logits. An MTP round emits each
/// draft the target accepted through ``accept(_:)``, which the matcher already took, and decides
/// the token after the accepted drafts here as plain decoding does. Both paths thus emit the same
/// text from the same decisions.
struct GuidedDecoder {

    /// The logger of the guided generation loop.
    private static let logger = MLXLogger(label: "GuidedGenerationLoop")

    /// The tokens between two progress lines of the diagnostic log.
    private static let progressLogInterval = 50

    /// The characters of output a progress line of the diagnostic log shows.
    private static let progressLogPrefixLength = 200

    /// The index, from the end of the sorted constrained logits, of the second-best logit.
    private static let runnerUpLogitIndex = -2

    /// The milliseconds in one second.
    private static let millisecondsPerSecond: Int64 = 1000

    /// The attoseconds in one millisecond.
    private static let attosecondsPerMillisecond: Int64 = 1_000_000_000_000_000

    /// The rules of the generation.
    let rules: GuidedDecisionRules

    /// The position of the next decision.
    private(set) var position: GuidedPosition

    /// The tokens counted so far, sampled and fast-forwarded.
    private(set) var tokenCount = 0

    /// Whether the grammar stopped the generation: it terminated, or allowed a stop token.
    private(set) var grammarStopped = false

    /// Whether the consumer of the output stopped the generation.
    private(set) var consumerStopped = false

    /// The whitespace run after the last decision.
    private var whitespace: WhitespaceRunTracker

    /// Decodes the emitted tokens into text.
    private var detokenizer: any StreamingDetokenizer

    /// The text emitted so far, for the diagnostic log.
    private var emittedText = ""

    /// Takes each piece of text; `false` stops the generation.
    private let emit: (String) -> Bool

    /// Whether the generation writes the diagnostic log.
    private let diagnosticLog: Bool

    /// The test sink that records the decisions of the generation, or `nil`: a bound sink
    /// that does not record decoding (``GuidedGenerationDiagnosticSink/recordsDecoding``)
    /// reaches no decision.
    private let diagnosticSink: GuidedGenerationDiagnosticSink?

    /// The clock of the diagnostic log.
    private let clock = ContinuousClock()

    /// When the decoding started, for the diagnostic log.
    private let startInstant: ContinuousClock.Instant

    /// Starts the decisions of a generation at the matcher's first position.
    ///
    /// - Parameters:
    ///   - rules: The rules of the generation.
    ///   - tokenizer: The tokenizer that decodes the output.
    ///   - diagnosticLog: Whether the generation writes the diagnostic log.
    ///   - emit: Takes each piece of text; `false` stops the generation.
    /// - Throws: The error of the matcher.
    init(
        rules: GuidedDecisionRules, tokenizer: any Tokenizer, diagnosticLog: Bool,
        emit: @escaping (String) -> Bool
    ) throws {
        self.rules = rules
        self.position = try rules.startPosition()
        self.whitespace = position.whitespace
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
        self.emit = emit
        self.diagnosticLog = diagnosticLog
        self.diagnosticSink = GuidedGenerationDiagnosticSink.current.flatMap {
            $0.recordsDecoding ? $0 : nil
        }
        self.startInstant = clock.now
    }

    /// Whether the token budget allows another decision.
    var hasBudget: Bool { tokenCount < rules.maxTokens }

    /// Decides the token at ``position`` from the last row of `logits`, commits it to the
    /// matcher, and emits it with the tokens the grammar fast-forwarded after it.
    ///
    /// - Parameter logits: Target logits, shape `[B, positions, vocabulary]`.
    /// - Returns: The tokens to feed the target next, or `nil` when the generation stops.
    /// - Throws: `CancellationError` when the task is cancelled, or the error of the matcher.
    mutating func decide(from logits: MLXArray) throws -> [Int32]? {
        try Task.checkCancellation()
        guard !stopsBeforeDeciding() else { return nil }
        note(position)
        let token = Int(
            GuidedGenerationLoop.applyMaskAndSample(
                logits: logits, maskArray: position.maskArray,
                closingBias: rules.bias(at: position)))
        let whitespaceAfter = rules.whitespace(after: token, from: position.whitespace)
        guard !rules.endsGeneration(token, at: position) else {
            logStop("EOS/unk tokenId=\(token) at token \(tokenCount)")
            grammarStopped = true
            return nil
        }
        diagnosticSink?.recordDecisionMargin(topTwoMargin(of: logits))
        let commit = try rules.constraint.commitToken(Int32(token))
        return emit(token, commit: commit, whitespaceAfter: whitespaceAfter)
    }

    /// The gap between the two best logits of the last row of `logits` under the mask and the
    /// bias of ``position``.
    private func topTwoMargin(of logits: MLXArray) -> Float {
        let best = sorted(rules.constrainedLogits(logits[0..., -1, 0...], at: position), axis: -1)
        return (best[0..., -1] - best[0..., Self.runnerUpLogitIndex]).item(Float.self)
    }

    /// Emits a draft the target accepted. The matcher took the draft when it was drafted.
    ///
    /// - Parameter draft: The accepted draft, decided at ``position``.
    /// - Returns: The tokens to feed the target for the draft, or `nil` when the generation stops.
    mutating func accept(_ draft: GuidedDraft) -> [Int32]? {
        note(draft.position)
        return emit(draft.token, commit: draft.commit, whitespaceAfter: draft.whitespaceAfter)
    }

    /// Stops the generation when the grammar already terminated at ``position``, which is then
    /// the last decision.
    ///
    /// - Returns: `true` when the generation stopped.
    mutating func stopsBeforeDeciding() -> Bool {
        guard position.mask.isTerminated else { return false }
        note(position)
        logStop("mask.isTerminated at token \(tokenCount)")
        grammarStopped = true
        return true
    }

    /// Moves to the position after the last decision, which the matcher stands at.
    ///
    /// - Parameter computed: The position after the last decision when a draft chain computed
    ///   it already, or `nil` to compute it from the matcher.
    /// - Throws: The error of the matcher.
    mutating func advance(reusing computed: GuidedPosition? = nil) throws {
        position = try computed ?? rules.position(tokenCount: tokenCount, whitespace: whitespace)
    }

    /// Moves back to `recorded`, the position of a rejected draft, after the matcher rolled back
    /// to it. The matcher's own mask decides the position, and the recorded mask array serves
    /// when the masks agree.
    ///
    /// - Parameter recorded: The position the draft chain recorded for the rejected draft.
    /// - Throws: The error of the matcher.
    mutating func restore(to recorded: GuidedPosition) throws {
        whitespace = recorded.whitespace
        let mask = try rules.constraint.computeMask()
        position =
            recorded.hasMask(mask)
            ? recorded
            : try rules.position(tokenCount: recorded.tokenCount, whitespace: recorded.whitespace)
    }

    /// Emits the rest of the text, and reports how the generation ended.
    ///
    /// - Returns: The tokens the generation counted.
    /// - Throws: ``GuidedGenerationError/incompleteOutput`` when the budget ran out before the
    ///   grammar stopped.
    mutating func finish() throws -> Int {
        if !consumerStopped, let text = detokenizer.finish(), !text.isEmpty {
            emittedText += text
            _ = emit(text)
        }
        if diagnosticLog {
            Self.logger.info(
                "[GuidedGen] done tokens=\(tokenCount) elapsed=\(milliseconds(since: startInstant))ms"
            )
            if let logs = rules.constraint.flushLogs() {
                Self.logger.warning("[GuidedGen] xgrammar logs:\n\(logs)")
            }
        }
        diagnosticSink?.recordTermination(
            grammarTerminated: grammarStopped, generatedTokenCount: tokenCount)
        if !grammarStopped && tokenCount >= rules.maxTokens {
            throw GuidedGenerationError.incompleteOutput
        }
        return tokenCount
    }

    /// Emits a decided token and the tokens the grammar fast-forwarded after it, as far as the
    /// budget and the consumer allow.
    ///
    /// A commit that terminates the grammar stops before its fast-forwarded tokens.
    private mutating func emit(
        _ token: Int, commit: CommitResult, whitespaceAfter: WhitespaceRunTracker
    ) -> [Int32]? {
        whitespace = whitespaceAfter
        diagnosticSink?.recordSampledToken(token)
        guard append(token) else { return nil }
        logProgress()
        guard !commit.isTerminated else {
            logStop("commitResult.isTerminated at token \(tokenCount)")
            grammarStopped = true
            return nil
        }
        for forced in commit.tokens {
            guard hasBudget else { return nil }
            diagnosticSink?.recordFastForwardToken(Int(forced))
            guard append(Int(forced)) else { return nil }
        }
        return GuidedGenerationLoop.modelFeed(sampled: Int32(token), fastForward: commit.tokens)
    }

    /// Emits one token: its text, when the detokenizer has some, then counts it.
    ///
    /// - Returns: `false` when the consumer stopped the generation, before the token counted.
    private mutating func append(_ token: Int) -> Bool {
        detokenizer.append(token: token)
        if let text = detokenizer.next() {
            emittedText += text
            guard emit(text) else {
                consumerStopped = true
                return false
            }
        }
        tokenCount += 1
        return true
    }

    /// Records the decision at `position` for the diagnostic log and the test sink.
    private func note(_ position: GuidedPosition) {
        if diagnosticLog {
            Self.logger.info("\(snapshot(of: position).summary())")
        }
        diagnosticSink?.recordDecision(
            GuidedDecisionRecord(
                tokenCount: position.tokenCount, mask: position.mask.mask,
                needsApply: position.mask.needsApply, isTerminated: position.mask.isTerminated,
                bias: rules.bias(at: position)?.asArray(Float.self)))
    }

    /// The mask snapshot of `position` for the diagnostic log.
    private func snapshot(of position: GuidedPosition) -> MaskSnapshot {
        position.mask.mask.withUnsafeBufferPointer { buffer in
            MaskSnapshot.capture(
                sampleMask: position.mask.needsApply
                    ? buffer.baseAddress.map {
                        UnsafeRawPointer($0).assumingMemoryBound(to: UInt32.self)
                    } : nil,
                vocabSize: rules.vocabSize, tokenIndex: position.tokenCount,
                isStop: position.mask.isTerminated)
        }
    }

    /// Writes why the generation stopped to the diagnostic log.
    private func logStop(_ reason: String) {
        guard diagnosticLog else { return }
        Self.logger.info("[GuidedGen] Stop reason: \(reason)")
    }

    /// Writes a progress line to the diagnostic log every ``progressLogInterval`` tokens.
    private func logProgress() {
        guard diagnosticLog, tokenCount % Self.progressLogInterval == 0 else { return }
        let prefix = String(emittedText.prefix(Self.progressLogPrefixLength))
        Self.logger.info(
            "[GuidedGen] token=\(tokenCount) elapsed=\(milliseconds(since: startInstant))ms text=\(prefix)"
        )
    }

    /// The whole milliseconds from `instant` to now.
    private func milliseconds(since instant: ContinuousClock.Instant) -> Int64 {
        let elapsed = (clock.now - instant).components
        return elapsed.seconds * Self.millisecondsPerSecond
            + elapsed.attoseconds / Self.attosecondsPerMillisecond
    }
}
