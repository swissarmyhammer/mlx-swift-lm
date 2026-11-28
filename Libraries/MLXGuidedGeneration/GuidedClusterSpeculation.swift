// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// MTP speculation of one guided generation on a ``ClusterLanguageModel``, with the drafter its
/// rank 0 holds.
///
/// Each round is one ``RemoteModelOp/speculate(_:)``, one round trip to the worker, as a plain
/// generation of the cluster model speculates (``ClusterMTPTokenIterator``): every rank rewinds
/// the drafts the round before did not keep, rank 0 drafts after the round's token, and one
/// forward verifies the token and the drafts. Rank 0 drafts without the grammar mask, which
/// stays in this process.
///
/// The generation then walks the tokens plain guided decoding feeds after the round's token:
/// the tokens the grammar already fixed (the rest of the last decision's feed), and each token
/// it decides from the verify row of its position, as plain decoding decides it from a forward
/// of one token (``GuidedGenerationLoop/decodePlainly(from:decoder:target:)``). Each draft
/// that equals the token there is kept, and the walk stops at the first draft that differs.
/// Every decision thus reads logits conditioned on the tokens plain decoding fed, and guided
/// output with MTP is the output of plain guided decoding. A draft the grammar would mask ends
/// its round at that row, and the target decides the token there.
///
/// The drafts a round did not keep stay in the caches of the worker until the next round
/// rewinds them, and the last round's are trimmed when the generation ends, so the caches of
/// every rank then hold the prompt and each token a forward fed.
struct GuidedClusterSpeculation {

    /// The cluster target, whose rank 0 holds the drafter.
    private let model: ClusterLanguageModel

    /// The most drafts of one round.
    private let maximumDrafts: Int

    /// The drafter top-1 probability below which rank 0 stops a round's chain.
    private let minimumDraftProbability: Float

    /// The drafts of the last round that the generation did not keep, which the caches of the
    /// worker still hold.
    private var rejectedCount = 0

    /// The test sink that records each round, or `nil`.
    private let diagnosticSink = GuidedGenerationDiagnosticSink.current

    /// Makes the speculation of `speculation`, whose drafter is the one rank 0 of `model`
    /// holds.
    ///
    /// - Parameters:
    ///   - speculation: The speculation of the generation.
    ///   - drafter: The drafter of `speculation`: ``ClusterLanguageModel/mtpDrafter`` of `model`.
    ///   - model: The cluster target.
    init(speculation: MTPSpeculation, drafter: RemoteMTPDrafter, model: ClusterLanguageModel) {
        self.model = model
        self.maximumDrafts =
            min(speculation.blockSize, drafter.maximumBlockSize ?? speculation.blockSize) - 1
        self.minimumDraftProbability = speculation.minimumDraftProbability
    }

    /// Decodes the rest of the generation from `pending`, the tokens the last decision feeds the
    /// target.
    ///
    /// - Parameters:
    ///   - pending: The tokens of the last decision, which no forward fed yet.
    ///   - decoder: The decisions and the output of the generation.
    ///   - target: The cluster target.
    /// - Throws: `CancellationError`, the failure a round or the final trim kept, for example
    ///   ``ClusterModelError/memberLost(_:)``, or the error of the matcher.
    mutating func decode(
        from pending: [Int32], decoder: inout GuidedDecoder, target: GuidedTarget
    ) throws {
        var queued = pending[...]
        while let token = queued.popFirst() {
            try Task.checkCancellation()
            let round = try target.speculate(
                on: model, from: token, rejectedCount: rejectedCount,
                draftLimit: draftLimit(queued: queued.count, decoder: decoder),
                minimumDraftProbability: minimumDraftProbability)
            let accepted = try Self.walk(
                round.drafts, verifyLogits: round.logits, queued: &queued, decoder: &decoder)
            rejectedCount = round.drafts.count - accepted
            diagnosticSink?.recordSpeculativeRound(
                draftedTokenCount: round.drafts.count, acceptedDraftTokenCount: accepted)
        }
        try target.trimSpeculated(rejectedCount)
    }

    /// The most drafts the next round may propose: as many as the tokens it may still feed
    /// after its own, the fixed tokens `queued` after it and the budget, allow.
    ///
    /// - Parameters:
    ///   - queued: The fixed tokens after the round's token.
    ///   - decoder: The decisions of the generation.
    /// - Returns: The draft limit.
    private func draftLimit(queued: Int, decoder: GuidedDecoder) -> Int {
        min(maximumDrafts, queued + max(0, decoder.rules.maxTokens - decoder.tokenCount))
    }

    /// Walks `drafts` along the tokens plain guided decoding feeds after the round's token,
    /// deciding each token the grammar does not fix from its verify row.
    ///
    /// - Parameters:
    ///   - drafts: The drafts of the round.
    ///   - verifyLogits: The target logits of each verified position, shape
    ///     `[1, drafts + 1, vocabulary]`: row `i` follows the round's token and the first `i`
    ///     drafts.
    ///   - queued: The fixed tokens after the round's token. On return, the tokens the next
    ///     round feeds, the first of them its token, or none when the generation stops.
    ///   - decoder: The decisions and the output of the generation.
    /// - Returns: The drafts kept: each equals the token plain decoding feeds at its position.
    /// - Throws: `CancellationError`, or the error of the matcher.
    private static func walk(
        _ drafts: [Int32], verifyLogits: MLXArray, queued: inout ArraySlice<Int32>,
        decoder: inout GuidedDecoder
    ) throws -> Int {
        var accepted = 0
        while true {
            if queued.isEmpty {
                try decoder.advance()
                guard decoder.hasBudget,
                    let feed = try decoder.decide(
                        from: verifyLogits[0..., accepted ..< (accepted + 1), 0...])
                else { return accepted }
                queued = feed[...]
            }
            guard accepted < drafts.count, drafts[accepted] == queued.first else {
                return accepted
            }
            queued.removeFirst()
            accepted += 1
        }
    }
}
