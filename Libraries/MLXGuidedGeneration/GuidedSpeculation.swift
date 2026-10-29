// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// A drafter guided generation speculates with: a stateful drafter that drafts through a logit
/// processor, so each draft step takes the grammar mask and biases, and that resumes from the
/// prompt chunks the prefill feeds it. The Qwen 3.5 MTP heads are such drafters.
typealias GuidedMTPDrafter = ResumableMTPDrafterModel & StatefulMTPDistributionDrafterModel

/// MTP speculation of one guided generation.
///
/// Each round drafts after a decided token. Every draft step's logits take the mask and the
/// biases of its position, and its argmax goes to the grammar matcher. One verify forward then
/// gives the target logits of every drafted position, and the target accepts each draft its own
/// constrained argmax equals. After a rejection the matcher rolls back over the rejected drafts
/// and their forced tokens, the target decides the token there as plain decoding does, and the
/// main cache rewinds the rejected tail in place. Guided output with MTP is thus the output of
/// plain guided decoding.
///
/// The tokens a draft feeds the target are the tokens plain decoding feeds for it, the draft and
/// then the tokens its commit forced: see
/// ``GuidedGenerationLoop/modelFeed(sampled:fastForward:)``. A draft whose commit forces tokens
/// ends its chain, and the draft with its forced tokens joins the verify of its round. A decision
/// of the target that forces tokens feeds itself and them in verify-shaped forwards of their own.
/// Both run on the targets that verify in place only (the Qwen 3.5 hybrids), where a verify
/// forward of several tokens on a populated cache gives each row the one-token logits; every
/// other target decodes plainly.
///
/// The adaptive guard of ``MTPSpeculation/adaptivePolicy`` lowers the block size while rounds
/// decode slower than plain decode, and pauses speculation when the smallest block still does.
/// A paused decision feeds its tokens one forward each, as plain decoding does, and the drafter
/// takes them in one commit when speculation resumes.
struct GuidedSpeculation {

    /// The drafter.
    private let drafter: any GuidedMTPDrafter

    /// The drafter's own state: one entry for each token the main cache holds, and one for the
    /// token the next forward feeds first.
    private var drafterState: MTPDrafterState

    /// The most drafts of one round.
    private let maximumDrafts: Int

    /// The target hidden row of the last position the main cache keeps.
    private var lastKeptHidden: MLXArray

    /// The target position deltas of the last forward, or `nil` for a target without them.
    private var positionDeltas: MLXArray?

    /// Whether the target stopped reporting its drafter state, which ends the speculation.
    private var stoodDown = false

    /// The guard that lowers the block size or pauses speculation while rounds decode slower
    /// than plain decode, or `nil` when its policy is disabled.
    private var adaptiveGuard: MTPAdaptiveGuard?

    /// The tokens the paused decisions fed since the last round, which the drafter takes when
    /// speculation resumes.
    private var pausedFeed: [Int32] = []

    /// The target hidden row of each token of ``pausedFeed``.
    private var pausedHidden: [MLXArray] = []

    /// The tokens the paused decisions emitted, which are no drafts.
    private var pausedTokenCount = 0

    /// The test sink that records each round, or `nil`.
    private let diagnosticSink = GuidedGenerationDiagnosticSink.current

    /// The tokens a verify feeds for a round and whether they hold the feed of its last draft.
    private struct VerifyBatch {

        /// The decided token the round drafts after, then the feed of each draft the verify
        /// holds.
        let tokens: [Int32]

        /// Whether ``tokens`` hold the feed of the last draft. They do not when the last draft
        /// terminated the grammar, or forced more tokens than one verify holds.
        let feedsLastDraft: Bool

        /// The verified positions the main cache keeps when the target accepts `accepted` of
        /// `draftCount` drafts: the decided token and the accepted drafts, or every position.
        func keptPositions(accepted: Int, of draftCount: Int) -> Int {
            accepted < draftCount ? 1 + accepted : tokens.count
        }
    }

    /// The drafter of `speculation`, when guided generation can speculate with it on `target`.
    ///
    /// A drafter that cannot draft for the target is logged once and recorded on the
    /// diagnostic sink with its stated reason.
    ///
    /// - Parameters:
    ///   - speculation: The speculation of the generation, or `nil`.
    ///   - target: The target and its new cache.
    /// - Returns: The drafter, or `nil` for plain decoding.
    static func drafter(
        of speculation: MTPSpeculation?, for target: GuidedTarget
    ) -> (any GuidedMTPDrafter)? {
        guard let speculation else { return nil }
        if let fallback = MTPDrafterFallback.incompatibility(
            of: speculation.drafter, with: target.model)
        {
            GuidedGenerationDiagnosticSink.current?.recordSpeculationFallback(
                fallback.statedReason, reason: fallback.reason)
            return nil
        }
        guard let drafter = speculation.drafter as? any GuidedMTPDrafter,
            !drafter.requiresSharedTargetKV,
            target.verifiesInPlace(tokens: MTPDraftBlockOutput.minimumChainedMTPBlockSize)
        else { return nil }
        return drafter
    }

    /// Primes the drafter over the prompt the target just prefilled, up to the first decided
    /// token.
    ///
    /// - Parameters:
    ///   - speculation: The speculation of the generation.
    ///   - drafter: The drafter of `speculation`.
    ///   - feed: The feed the prefill chunks went through.
    ///   - prefill: What the prefill gave.
    ///   - firstFeed: The tokens the first decision feeds the target.
    ///   - target: The target.
    /// - Returns: The speculation, or `nil` when the drafter cannot start, and the generation
    ///   decodes plainly.
    init?(
        speculation: MTPSpeculation, drafter: any GuidedMTPDrafter, feed: MTPDrafterPrefillFeed,
        prefill: GuidedPrefill, firstFeed: [Int32], target: GuidedTarget
    ) {
        let blockSize = min(
            speculation.blockSize, drafter.maximumBlockSize ?? speculation.blockSize,
            target.mostVerifiedTokens)
        guard let first = firstFeed.first, blockSize > 1,
            let segmentTokens = prefill.segmentTokens, let segmentHidden = prefill.segmentHidden,
            var state = feed.primedState(
                finalSegment: segmentTokens, segmentHidden: segmentHidden,
                firstBonus: MLXArray([first]), positionDeltas: prefill.positionDeltas)
        else { return nil }
        state.minimumDraftProbability = speculation.minimumDraftProbability
        self.drafter = drafter
        self.drafterState = state
        self.maximumDrafts = blockSize - 1
        self.lastKeptHidden = segmentHidden[0..., (-1)..., 0...]
        self.positionDeltas = prefill.positionDeltas
        self.adaptiveGuard = MTPAdaptiveGuard(
            policy: speculation.adaptivePolicy, blockSize: blockSize)
    }

    /// Decodes the rest of the generation from `pending`, the tokens the last decision feeds
    /// the target.
    ///
    /// A round that cannot verify in place hands the rest of the generation to plain decoding.
    ///
    /// - Parameters:
    ///   - pending: The tokens of the last decision, which no forward fed yet.
    ///   - decoder: The decisions and the output of the generation.
    ///   - target: The target.
    /// - Throws: `CancellationError`, or the error of the matcher.
    mutating func decode(
        from pending: [Int32], decoder: inout GuidedDecoder, target: inout GuidedTarget
    ) throws {
        defer { reportAdaptiveDecisions() }
        var feed = pending
        while decoder.hasBudget {
            try Task.checkCancellation()
            guard !stoodDown, let drafts = draftsThisRound(decoder: decoder, target: target)
            else {
                try GuidedGenerationLoop.decodePlainly(
                    from: feed, decoder: &decoder, target: &target)
                return
            }
            let emittedBefore = decoder.tokenCount
            let next = try decideNext(
                after: feed, drafts: drafts, decoder: &decoder, target: &target)
            adaptiveGuard?.recordEmitted(decoder.tokenCount - emittedBefore)
            guard let next else { return }
            feed = next
        }
    }

    /// Decides the tokens after `feed` as the adaptive guard asks: a paused decision, or a round
    /// of `drafts` drafts after the drafter takes the paused decisions since the last round.
    ///
    /// - Returns: The tokens the last decision feeds the target, or `nil` when the generation
    ///   stops.
    private mutating func decideNext(
        after feed: [Int32], drafts: Int, decoder: inout GuidedDecoder, target: inout GuidedTarget
    ) throws -> [Int32]? {
        if var adaptiveGuard {
            let step = adaptiveGuard.beginUnit(
                contextLength: target.cacheStorage.processedTokenCount)
            self.adaptiveGuard = adaptiveGuard
            if step == .decodePlainly {
                let emittedBefore = decoder.tokenCount
                let next = try decidePaused(after: feed, decoder: &decoder, target: &target)
                pausedTokenCount += decoder.tokenCount - emittedBefore
                return next
            }
        }
        catchUpDrafter(next: feed, target: target)
        guard !stoodDown else { return feed }
        return feed.count == 1
            ? try speculate(after: feed[0], drafts: drafts, decoder: &decoder, target: &target)
            : try feedForcedTokens(
                feed, chunkLength: Self.forcedChunkLength(drafts: drafts, target: target),
                decoder: &decoder, target: &target)
    }

    /// Reports the decisions of the adaptive guard to the diagnostic sink, when both exist.
    private func reportAdaptiveDecisions() {
        guard let adaptiveGuard else { return }
        diagnosticSink?.recordAdaptiveDecisions(
            of: adaptiveGuard.controller, plainTokens: pausedTokenCount)
    }

    /// The drafts of the next round, or `nil` when the target cannot verify a round in place.
    private func draftsThisRound(decoder: GuidedDecoder, target: GuidedTarget) -> Int? {
        let guardedDrafts = (adaptiveGuard?.controller.blockSize).map { $0 - 1 } ?? maximumDrafts
        let drafts = min(guardedDrafts, decoder.rules.maxTokens - decoder.tokenCount)
        guard drafts > 0, target.verifiesInPlace(tokens: drafts + 1) else { return nil }
        return drafts
    }

    /// One paused decision: feeds `feed` one token per forward, as plain decoding does, keeps
    /// each forward's hidden row for the drafter, and decides the token after them.
    ///
    /// - Returns: The tokens the decision feeds the target, or `nil` when the generation stops.
    private mutating func decidePaused(
        after feed: [Int32], decoder: inout GuidedDecoder, target: inout GuidedTarget
    ) throws -> [Int32]? {
        var lastLogits: MLXArray?
        for token in feed {
            discardSpeculativePromptCacheCheckpoints(target.cache)
            let output = target.verify([token])
            target.keepVerified(1, of: 1)
            lastLogits = output.logits
            recordPausedRow(token: token, output: output)
        }
        guard let lastLogits else {
            preconditionFailure("A guided feed holds at least one token")
        }
        asyncEval(lastLogits)
        try decoder.advance()
        return try decoder.decide(from: lastLogits)
    }

    /// Keeps the hidden row a paused forward of `token` gave, or ends the speculation when the
    /// forward gave none.
    private mutating func recordPausedRow(token: Int32, output: LMOutput) {
        guard let hidden = output.state?[mtpLastHiddenStatesKey], hidden.dim(1) > 0 else {
            stoodDown = true
            return
        }
        pausedFeed.append(token)
        pausedHidden.append(hidden[0..., (-1)..., 0...])
        positionDeltas = output.state?[mtpPositionDeltasKey] ?? positionDeltas
    }

    /// Commits the tokens of the paused decisions since the last round to the drafter in one
    /// commit, with `next`, the tokens the last decision feeds, after them.
    private mutating func catchUpDrafter(next: [Int32], target: GuidedTarget) {
        let fed = pausedFeed
        let hidden = pausedHidden
        pausedFeed = []
        pausedHidden = []
        guard !stoodDown, !fed.isEmpty else { return }
        commitDrafter(
            fed: fed, kept: fed.count, hidden: concatenated(hidden, axis: 1),
            positionDeltas: positionDeltas, next: next, target: target)
    }

    /// One round: drafts after `bonus`, verifies the drafts, emits the drafts the target
    /// accepted, and decides the token after them.
    ///
    /// - Returns: The tokens the round's last decision feeds the target, or `nil` when the
    ///   generation stops.
    private mutating func speculate(
        after bonus: Int32, drafts maximum: Int, decoder: inout GuidedDecoder,
        target: inout GuidedTarget
    ) throws -> [Int32]? {
        discardSpeculativePromptCacheCheckpoints(target.cache)
        try decoder.advance()
        guard !decoder.stopsBeforeDeciding() else { return nil }
        let chain = try draft(after: bonus, drafts: maximum, decoder: decoder, target: target)
        let drafts = chain.drafts
        let batch = Self.verifyBatch(after: bonus, drafts: drafts, target: target)
        let output = target.verify(batch.tokens)
        let accepted = Self.acceptedDraftCount(
            drafts, verifyLogits: output.logits, rules: decoder.rules)
        diagnosticSink?.recordSpeculativeRound(
            draftedTokenCount: drafts.count, acceptedDraftTokenCount: accepted)

        var lastFeed = [bonus]
        for draft in drafts.prefix(accepted) {
            guard let feed = decoder.accept(draft) else { return nil }
            lastFeed = feed
        }
        let kept = batch.keptPositions(accepted: accepted, of: drafts.count)
        let next: [Int32]?
        if accepted < drafts.count {
            try rollBack(chain, from: accepted)
            try decoder.restore(to: drafts[accepted].position)
            next = try Self.decide(at: kept - 1, of: output.logits, decoder: &decoder)
        } else if !batch.feedsLastDraft {
            next = lastFeed
        } else {
            try decoder.advance(reusing: chain.nextPosition)
            next = try Self.decide(at: kept - 1, of: output.logits, decoder: &decoder)
        }
        guard let next else { return nil }
        target.keepVerified(kept, of: batch.tokens.count)
        commitDrafter(
            fed: batch.tokens, kept: kept, hidden: output.state?[mtpLastHiddenStatesKey],
            positionDeltas: output.state?[mtpPositionDeltasKey], next: next, target: target)
        return next
    }

    /// Drafts up to `maximum` tokens after `bonus`, each under the grammar.
    private mutating func draft(
        after bonus: Int32, drafts maximum: Int, decoder: GuidedDecoder, target: GuidedTarget
    ) throws -> GuidedDraftChain {
        let chain = GuidedDraftChain(rules: decoder.rules, start: decoder.position)
        _ = drafter.draftBlockWithLogits(
            target: target.model, lastToken: MLXArray([bonus]), lastHidden: lastKeptHidden,
            sharedKV: [:], positionDeltas: positionDeltas,
            queryOffset: target.cacheStorage.processedTokenCount, blockSize: maximum + 1,
            state: &drafterState, processor: chain.processor, sampler: ArgMaxSampler())
        if let failure = chain.failure {
            throw failure
        }
        return chain
    }

    /// The tokens of each forward that feeds forced tokens: a whole verify where the target
    /// verifies one in place, else the verify of this round's drafts.
    private static func forcedChunkLength(drafts: Int, target: GuidedTarget) -> Int {
        target.verifiesInPlace(tokens: target.mostVerifiedTokens)
            ? target.mostVerifiedTokens : drafts + 1
    }

    /// Feeds `feed`, a decided token and the tokens its commit forced, in verify-shaped forwards
    /// of `chunkLength` tokens at most, and decides the token after them.
    ///
    /// - Returns: The tokens the decision feeds the target, or `nil` when the generation stops.
    private mutating func feedForcedTokens(
        _ feed: [Int32], chunkLength: Int, decoder: inout GuidedDecoder,
        target: inout GuidedTarget
    ) throws -> [Int32]? {
        let outputs = stride(from: 0, to: feed.count, by: chunkLength).map { start in
            discardSpeculativePromptCacheCheckpoints(target.cache)
            let chunk = Array(feed[start ..< min(start + chunkLength, feed.count)])
            let output = target.verify(chunk)
            target.keepVerified(chunk.count, of: chunk.count)
            return output
        }
        guard let last = outputs.last else {
            preconditionFailure("A guided feed holds at least one token")
        }
        asyncEval(last.logits)
        try decoder.advance()
        guard let next = try decoder.decide(from: last.logits) else { return nil }
        let hiddenRows = outputs.compactMap { $0.state?[mtpLastHiddenStatesKey] }
        commitDrafter(
            fed: feed, kept: feed.count,
            hidden: hiddenRows.count == outputs.count ? concatenated(hiddenRows, axis: 1) : nil,
            positionDeltas: last.state?[mtpPositionDeltasKey], next: next, target: target)
        return next
    }

    /// Commits the fed tokens the main cache kept, and the first token `next` feeds, to the
    /// drafter, each paired with the target hidden row of the position before it.
    ///
    /// A forward that reported no hidden rows ends the speculation.
    private mutating func commitDrafter(
        fed: [Int32], kept: Int, hidden: MLXArray?, positionDeltas: MLXArray?, next: [Int32],
        target: GuidedTarget
    ) {
        guard let hidden, let first = next.first else {
            stoodDown = true
            return
        }
        self.positionDeltas = positionDeltas ?? self.positionDeltas
        drafter.commitDrafterState(
            target: target.model, targetHidden: hidden,
            draftTokens: MLXArray(Array(fed.dropFirst()))[.newAxis], acceptedCount: kept - 1,
            finalToken: MLXArray([first]), positionDeltas: self.positionDeltas,
            state: &drafterState, sampler: mtpDrafterSeedSampler)
        lastKeptHidden = hidden[0..., (kept - 1) ..< kept, 0...]
    }

    /// Rolls the matcher back over the drafts from `index` on.
    private func rollBack(_ chain: GuidedDraftChain, from index: Int) throws {
        let rejected = chain.drafts[index...]
        try chain.rollBack(draftsFrom: index)
        diagnosticSink?.recordRollback(
            GuidedRollbackRecord(
                rejectedDraftCount: rejected.count,
                acceptanceCount: rejected.reduce(0) { $0 + $1.acceptanceCount }))
    }

    /// Decides the token after verify position `row`, when the budget allows a decision.
    private static func decide(
        at row: Int, of logits: MLXArray, decoder: inout GuidedDecoder
    ) throws -> [Int32]? {
        guard decoder.hasBudget else { return nil }
        return try decoder.decide(from: logits[0..., row ..< (row + 1), 0...])
    }

    /// The verify of a round: the decided token, every draft but the last, and the feed of the
    /// last draft when one verify holds it.
    private static func verifyBatch(
        after bonus: Int32, drafts: [GuidedDraft], target: GuidedTarget
    ) -> VerifyBatch {
        let tokens = [bonus] + drafts.dropLast().map { Int32($0.token) }
        guard let last = drafts.last else {
            return VerifyBatch(tokens: tokens, feedsLastDraft: true)
        }
        let withLast = tokens + last.feed
        guard !last.commit.isTerminated, target.verifiesInPlace(tokens: withLast.count) else {
            return VerifyBatch(tokens: tokens, feedsLastDraft: false)
        }
        return VerifyBatch(tokens: withLast, feedsLastDraft: true)
    }

    /// The drafts the target accepts: the longest run whose every draft is the target's own
    /// decision at its position, the argmax of the verify row before it under the position's
    /// mask and bias.
    private static func acceptedDraftCount(
        _ drafts: [GuidedDraft], verifyLogits: MLXArray, rules: GuidedDecisionRules
    ) -> Int {
        guard !drafts.isEmpty else { return 0 }
        let rows = drafts.enumerated().map { index, draft in
            rules.constrainedLogits(verifyLogits[0..., index, 0...], at: draft.position)
        }
        let decisions = argMax(concatenated(rows, axis: 0), axis: -1).asArray(Int.self)
        return zip(decisions, drafts).prefix { $0.0 == $0.1.token }.count
    }
}
