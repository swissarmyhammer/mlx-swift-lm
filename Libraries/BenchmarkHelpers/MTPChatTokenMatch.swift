// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// One token a chat turn generated, with the candidates the sampler chose among
/// when the session reported them.
public struct MTPChatToken: Sendable, Equatable {
    /// The vocabulary token ID.
    public let token: Int
    /// The most likely tokens at this step in descending order, or empty when
    /// the session did not report them: a speculative session does not ask for
    /// them, because asking makes it decode without speculation, and no session
    /// reports them for its stop token.
    public let candidates: [GenerateTokenLogProbability]

    /// Creates a token with the candidates of its step.
    public init(token: Int, candidates: [GenerateTokenLogProbability] = []) {
        self.token = token
        self.candidates = candidates
    }
}

/// How the tokens of an MTP turn compare with the tokens of the plain turn.
///
/// A greedy MTP turn cannot always reproduce the plain turn bit for bit. Its
/// verify pass runs two tokens through the target in one forward, and the
/// kernels of a two-token forward round differently from a one-token step: the
/// 4-bit quantized matmul gives different bits for two rows than for one, and
/// the one-token decode attention kernel differs from the attention of MLX. The
/// logits of the verify pass then differ from plain decoding by up to
/// ``MTPChatTokenMatch/nearTieLogitTolerance``, and where plain's two best
/// tokens lie closer than that the argmax can flip. Plain decoding flips the
/// same way between a warm and a cold prompt cache.
public enum MTPChatTokenMatch: Sendable, Equatable {
    /// The MTP turn generated exactly the tokens of the plain turn.
    case exact
    /// The turns part where plain's two best tokens lie within
    /// ``nearTieLogitTolerance`` of each other, and MTP took the second.
    case nearTie(step: Int, gap: Float)
    /// The turns part anywhere else. `gap` is the distance between plain's two
    /// best logits at `step`, or `nil` when plain did not report them.
    case diverged(step: Int, gap: Float?)

    /// The widest gap between plain's two best logits at which MTP may still
    /// take the second: the decoding-noise bound of a verify pass.
    ///
    /// Measured on Qwen3.8-27B-4bit over the same prefilled cache and the same
    /// 48 continuation tokens: a two-token verify forward was bit-equal to
    /// one-token decoding on 0 of 96 rows, and its logits differed by at most
    /// 0.48. The analysis is on kanban task ^chgkh4c.
    public static let nearTieLogitTolerance: Float = 0.5

    /// Compares the tokens of an MTP turn with the tokens of the plain turn of
    /// the same prompt.
    public init(mtp: [MTPChatToken], plain: [MTPChatToken]) {
        guard
            let step = zip(mtp, plain).enumerated().first(where: { $1.0.token != $1.1.token })?
                .offset
                ?? (mtp.count == plain.count ? nil : Swift.min(mtp.count, plain.count))
        else {
            self = .exact
            return
        }
        let candidates = step < plain.count ? plain[step].candidates : []
        guard candidates.count >= 2 else {
            self = .diverged(step: step, gap: nil)
            return
        }
        let gap = candidates[0].logProbability - candidates[1].logProbability
        let takesRunnerUp = step < mtp.count && mtp[step].token == candidates[1].token
        self =
            takesRunnerUp && gap <= Self.nearTieLogitTolerance
            ? .nearTie(step: step, gap: gap) : .diverged(step: step, gap: gap)
    }

    /// Whether the MTP turn wrote what plain decoding writes, up to a near-tie.
    public var holds: Bool {
        if case .diverged = self { return false }
        return true
    }

    /// The match in a result line.
    public var summary: String {
        switch self {
        case .exact:
            return "exact"
        case .nearTie(let step, let gap):
            return "near-tie at token \(step), gap \(gap)"
        case .diverged(let step, let gap):
            return "diverged at token \(step), gap \(gap.map { "\($0)" } ?? "unknown")"
        }
    }
}

/// An event of a chat turn whose tokens are recorded: a token, or an event of
/// the session's own handler.
enum MTPChatTokenEvent: Sendable, GenerationProviding {
    case token(MTPChatToken)
    case generation(Generation)

    var generation: Generation? {
        if case .generation(let generation) = self { return generation }
        return nil
    }
}

/// Records each token of a chat turn, and the most likely candidates of its
/// step when the session reports them.
///
/// Reading the candidates does not change decoding: a greedy iterator samples
/// from the logits either way and only reads the log probabilities beside them.
struct MTPChatTokenRecorder: GenerationTransformer {
    /// The candidates to report for each step, or `nil` to report none. A
    /// speculative session must report none: asking for them turns its MTP off.
    let candidateCount: Int?

    func configure(_ parameters: inout GenerateParameters) {
        if let candidateCount {
            parameters.logProbabilities = candidateCount
        }
    }

    func makeHandler(wrapping base: TextToolTokenLoopHandler) -> Handler {
        Handler(base: base)
    }

    /// Emits each token before the events the session's handler makes of it.
    struct Handler: TokenLoopHandler {
        var base: TextToolTokenLoopHandler

        var additionalStopTokenIDs: Set<Int> { base.additionalStopTokenIDs }
        var receivesStopTokens: Bool { base.receivesStopTokens }

        mutating func onToken(
            _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
            emit: (sending MTPChatTokenEvent) -> Bool
        ) -> TokenLoopDisposition {
            guard emit(.token(Self.record(token, logProbabilities))) else { return .cancelled }
            return base.onToken(token, logProbabilities: logProbabilities) {
                emit(.generation($0))
            }
        }

        mutating func onStopToken(
            _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
            emit: (sending MTPChatTokenEvent) -> Bool
        ) -> TokenLoopDisposition {
            guard emit(.token(Self.record(token, logProbabilities))) else { return .cancelled }
            return base.onStopToken(token, logProbabilities: logProbabilities) {
                emit(.generation($0))
            }
        }

        mutating func onGenerationEnd(
            emit: (sending MTPChatTokenEvent) -> Bool
        ) -> TokenLoopDisposition {
            base.onGenerationEnd { emit(.generation($0)) }
        }

        func infoEvent(_ info: GenerateCompletionInfo) -> MTPChatTokenEvent {
            .generation(base.infoEvent(info))
        }

        private static func record(
            _ token: Int, _ logProbabilities: DeferredTokenLogProbabilities?
        ) -> MTPChatToken {
            MTPChatToken(
                token: token,
                candidates: logProbabilities?.materialize().topLogProbabilities ?? [])
        }
    }
}
