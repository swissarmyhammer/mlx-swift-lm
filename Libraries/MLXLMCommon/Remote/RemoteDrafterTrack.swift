// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private let drafterTrackLogger = MLXLogger(label: "remote-drafter-track")

/// The MTP drafter rank 0 of a worker drafts with, and the target it drafts for.
///
/// The drafter reads the token embeddings and the vocabulary projection of the target, which a
/// sharded target keeps whole on every rank, so rank 0 drafts for its own shard of the target.
struct RemoteDrafting {

    /// The drafter.
    let drafter: any ResumableMTPDrafterModel

    /// The target the drafter drafts for: the model of the executor.
    let target: any LanguageModel

    /// The sampler of the drafts: argmax, with the tokens the target never samples masked, as
    /// the local MTP iterator drafts for a stateful drafter.
    let draftSampler: any LogitSampler

    /// Makes the drafting of `drafter` for `target`.
    ///
    /// - Parameters:
    ///   - drafter: The drafter.
    ///   - target: The target the drafter drafts for.
    init(drafter: any ResumableMTPDrafterModel, target: any LanguageModel) {
        self.drafter = drafter
        self.target = target
        self.draftSampler =
            makeSuppressTokensProcessor(model: target).map {
                SuppressTokensSampler(base: ArgMaxSampler(), suppressor: $0)
            } ?? ArgMaxSampler()
    }

    /// The most drafts a round of at most `draftLimit` drafts may propose on `cache`: as many as
    /// a verify of the round can rewind in place, else none.
    ///
    /// - Parameters:
    ///   - draftLimit: The most drafts the app asked for.
    ///   - cache: The cache of each layer of the target.
    /// - Returns: The draft limit.
    func rewindableDraftLimit(_ draftLimit: Int, cache: [KVCache]) -> Int {
        let nativeLimit =
            (target as? any SpeculativeCacheRewindModel)?.maximumNativeTargetCacheRewind
            ?? draftLimit
        let blockLimit = drafter.maximumBlockSize.map { $0 - 1 } ?? draftLimit
        let limit = Swift.min(draftLimit, nativeLimit, blockLimit)
        return limit > 0 && canRewindSpeculativeVerify(cache, positions: limit + 1) ? limit : 0
    }
}

/// The drafter state rank 0 keeps beside the caches of one handle, fed by every forward of the
/// handle.
///
/// The state stands where the target cache stands, so it follows each op the prompt cache
/// runs on the handle: a copy copies it, a trim rewinds it while the hidden rows of the trimmed
/// positions are known, and a longer trim starts it cold at the next forward, as a local
/// generation cold-starts a drafter whose continuation the cache left behind.
///
/// Each entry of the drafter pairs a token with the target hidden row of the position before
/// it, as ``ResumableMTPDrafterModel/appendDrafterState(target:tokens:targetHidden:positionDeltas:state:sampler:)``
/// takes them.
struct RemoteDrafterTrack {

    /// Where the drafter state stands against the target cache.
    private enum Phase {

        /// The drafter holds nothing the cache can resume from: the next forward starts it at
        /// its first position.
        case cold

        /// The drafter holds an entry for each token of the cache from the first it holds, and
        /// `boundary` is the target hidden row of the last position of the cache.
        case aligned(boundary: MLXArray)

        /// The drafter holds one entry more than ``aligned(boundary:)``: the entry of the
        /// token of a round, which the cache does not hold yet, and the tentative entries of
        /// its drafts.
        case primed(boundary: MLXArray)

        /// The cache holds a verify the round has not settled: the drafter holds the entry of
        /// the round's token, and the tentative entries of its drafts.
        case verifying(RemoteVerify)
    }

    /// The drafting of the executor.
    private let drafting: RemoteDrafting

    /// The drafter state.
    private var state: MTPDrafterState

    /// Where the state stands.
    private var phase = Phase.cold

    /// Makes the track of an empty cache.
    ///
    /// - Parameter drafting: The drafting of the executor.
    init(drafting: RemoteDrafting) {
        self.drafting = drafting
        self.state = drafting.drafter.makeState(parameters: nil)
    }

    /// The cache position of the first entry of the drafter state: zero while the drafter has
    /// followed the cache since its first position, and the position of the last cold start
    /// otherwise.
    var startPosition: Int {
        state.cacheStartPosition
    }

    /// The bytes the drafter state holds.
    var residentByteCount: Int {
        state.cache.reduce(0) { $0 + $1.residentByteCount }
    }

    /// Every array the track holds, which the op that changed them evaluates: the drafter
    /// caches, the seed proposal, and the hidden rows of the phase.
    var heldArrays: [MLXArray] {
        state.cache.flatMap(\.evaluationState) + state.seedArrays + phaseArrays
    }

    /// The hidden rows and the drafts the phase holds.
    private var phaseArrays: [MLXArray] {
        switch phase {
        case .cold: []
        case .aligned(let boundary), .primed(let boundary): [boundary]
        case .verifying(let verify): [verify.previousBoundary, verify.hidden, verify.drafts]
        }
    }

    /// A copy whose drafter state shares no cache with this one.
    ///
    /// - Returns: The copy.
    func copy() -> RemoteDrafterTrack {
        var copied = self
        copied.state.cache = state.cache.map { $0.copy() }
        return copied
    }

    /// `state` asking the target for the hidden rows the drafter reads.
    ///
    /// - Parameter state: The model state of a forward.
    /// - Returns: The state of the forward.
    func emitting(_ state: LMOutput.State?) -> LMOutput.State {
        var emitting = state ?? LMOutput.State()
        emitting[mtpEmitFlagKey] = true
        emitting[mtpTargetLayerIdsKey] = drafting.drafter.targetLayerIds
        return emitting
    }

    // MARK: - The ops of the handle

    /// Takes a forward of `tokens` at `position` that is no verify, with its hidden rows.
    ///
    /// - Parameters:
    ///   - tokens: The tokens of the forward.
    ///   - position: The cache position of the first token.
    ///   - hidden: The target hidden row of each token, or nil when the forward emitted none.
    mutating func didForward(_ tokens: [Int32], at position: Int, hidden: MLXArray?) {
        settle()
        guard let hidden, hidden.dim(1) >= tokens.count, !tokens.isEmpty else {
            return becomeCold("the target emitted no hidden row for each token of a forward")
        }
        let last = hidden[0..., (tokens.count - 1) ..< tokens.count, 0...]
        guard case .aligned(let boundary) = phase else {
            // The first token's entry needs the hidden row before it, which no forward left.
            state = coldState(at: position)
            phase = .aligned(boundary: last)
            return append(Array(tokens.dropFirst()), hidden: hidden)
        }
        phase = .aligned(boundary: last)
        append(tokens, hidden: concatenated([boundary, hidden], axis: 1))
    }

    /// Takes a trim of `count` positions of the cache.
    ///
    /// - Parameter count: The positions the cache dropped.
    mutating func didTrim(_ count: Int) {
        guard count > 0 else { return }
        unprime()
        guard case .verifying(let verify) = phase else {
            phase = .cold
            return
        }
        let kept = verify.positions - count
        if kept > 0 {
            settle(verify, keeping: kept)
        } else if kept == 0, state.removeLastEntries(state.proposalAppended + 1) {
            phase = .aligned(boundary: verify.previousBoundary)
        } else {
            phase = .cold
        }
    }

    /// Takes the start of a round from `token`, after the cache rewound the drafts the round
    /// before rejected, and proposes its drafts.
    ///
    /// - Parameters:
    ///   - token: The token of the round.
    ///   - rejectedCount: The drafts the round before rejected, which the cache rewound.
    ///   - draftLimit: The most drafts.
    ///   - minimumDraftProbability: The confidence floor of a chaining drafter.
    ///   - position: The cache position of `token`.
    /// - Returns: The drafts, up to `draftLimit`.
    mutating func propose(
        after token: Int32, rejectedCount: Int, draftLimit: Int, minimumDraftProbability: Float,
        position: Int
    ) -> [Int32] {
        prime(with: token, rejectedCount: rejectedCount, drafting: draftLimit > 0)
        guard case .primed(let boundary) = phase else { return [] }
        state.minimumDraftProbability = minimumDraftProbability
        let drafts = drafting.drafter.draftBlock(
            target: drafting.target, lastToken: MLXArray([token]), lastHidden: boundary,
            sharedKV: [:], positionDeltas: nil, queryOffset: position,
            blockSize: draftLimit + 1, state: &state, sampler: drafting.draftSampler)
        return drafts.flattened().asType(.int32).asArray(Int32.self)
    }

    /// Takes the verify of a round of `token` and `drafts` at `position`, with its hidden rows.
    ///
    /// - Parameters:
    ///   - token: The token of the round.
    ///   - drafts: The drafts of the round.
    ///   - position: The cache position of `token`.
    ///   - hidden: The target hidden row of each verified token, or nil when none.
    mutating func didVerify(
        _ token: Int32, drafts: [Int32], at position: Int, hidden: MLXArray?
    ) {
        guard !drafts.isEmpty else {
            return didForward([token], at: position, hidden: hidden)
        }
        guard case .primed(let previousBoundary) = phase, let hidden,
            hidden.dim(1) > drafts.count
        else {
            return becomeCold("a verify left no hidden row for each verified token")
        }
        phase = .verifying(
            RemoteVerify(
                previousBoundary: previousBoundary, hidden: hidden,
                drafts: MLXArray(drafts)[.newAxis]))
    }

    // MARK: - Settling

    /// Settles the round before, and adds the entry of `token`, the round's token, when the
    /// round drafts: the state then stands ``Phase/primed(boundary:)``.
    ///
    /// - Parameters:
    ///   - token: The token of the round.
    ///   - rejectedCount: The drafts the round before rejected.
    ///   - drafting: Whether the round drafts.
    private mutating func prime(with token: Int32, rejectedCount: Int, drafting drafts: Bool) {
        unprime()
        let column = MLXArray([token])[.newAxis]
        if case .verifying(let verify) = phase, drafts, rejectedCount < verify.positions {
            return commit(verify, accepted: verify.drafts.dim(1) - rejectedCount, final: column)
        }
        didTrim(rejectedCount)
        guard drafts, case .aligned(let boundary) = phase else { return }
        guard
            drafting.drafter.appendDrafterState(
                target: drafting.target, tokens: column, targetHidden: boundary,
                positionDeltas: nil, state: &state, sampler: mtpDrafterSeedSampler)
        else {
            return becomeCold("the drafter could not take the token of a round")
        }
        phase = .primed(boundary: boundary)
    }

    /// Commits `verify`, of which the target accepted `accepted` drafts, and adds the entry of
    /// `final`, the token of the next round: the state then stands ``Phase/primed(boundary:)``.
    ///
    /// - Parameters:
    ///   - verify: The verify of the round before.
    ///   - accepted: The drafts of `verify` the target accepted.
    ///   - final: The token of the next round, shape `[1, 1]`.
    private mutating func commit(_ verify: RemoteVerify, accepted: Int, final: MLXArray) {
        drafting.drafter.commitDrafterState(
            target: drafting.target, targetHidden: verify.hidden, draftTokens: verify.drafts,
            acceptedCount: accepted, finalToken: final, positionDeltas: nil, state: &state,
            sampler: mtpDrafterSeedSampler)
        guard state.isAligned else {
            return becomeCold("the drafter could not commit a verified round")
        }
        phase = .primed(boundary: verify.hidden[0..., accepted ..< (accepted + 1), 0...])
    }

    /// Settles a verify the cache keeps whole, and drops the entry of a round's token the cache
    /// never took, before an op other than a round.
    private mutating func settle() {
        unprime()
        guard case .verifying(let verify) = phase else { return }
        settle(verify, keeping: verify.positions)
    }

    /// Drops the entry of the token of a round and the tentative entries of its drafts, which
    /// the cache never took.
    private mutating func unprime() {
        guard case .primed(let boundary) = phase else { return }
        guard state.removeLastEntries(state.proposalAppended + 1) else {
            return becomeCold("the drafter could not drop the entries of an unverified round")
        }
        phase = .aligned(boundary: boundary)
    }

    /// Settles `verify`, of which the cache kept the first `kept` positions: drops the
    /// tentative entries, and adds an entry for each kept draft.
    ///
    /// - Parameters:
    ///   - verify: The verify of the round.
    ///   - kept: The verified positions the cache kept, at least one.
    private mutating func settle(_ verify: RemoteVerify, keeping kept: Int) {
        guard state.removeLastEntries(state.proposalAppended) else {
            return becomeCold("the drafter could not drop the tentative entries of a round")
        }
        phase = .aligned(boundary: verify.hidden[0..., (kept - 1) ..< kept, 0...])
        let keptDrafts = verify.drafts[0..., ..<(kept - 1)].flattened().asArray(Int32.self)
        append(keptDrafts, hidden: verify.hidden)
    }

    /// Adds an entry for each of `tokens`, the entry of `tokens[i]` from the hidden row `i` of
    /// `hidden`. Adds nothing for no tokens.
    private mutating func append(_ tokens: [Int32], hidden: MLXArray) {
        guard !tokens.isEmpty else { return }
        guard
            drafting.drafter.appendDrafterState(
                target: drafting.target, tokens: MLXArray(tokens)[.newAxis],
                targetHidden: hidden[0..., ..<tokens.count, 0...], positionDeltas: nil,
                state: &state, sampler: mtpDrafterSeedSampler)
        else {
            return becomeCold("the drafter could not take the tokens of a forward")
        }
    }

    /// An empty drafter state whose first entry stands at `position`.
    private func coldState(at position: Int) -> MTPDrafterState {
        var fresh = drafting.drafter.makeState(parameters: nil)
        fresh.cacheStartPosition = position
        fresh.nextPosition = position
        return fresh
    }

    /// Forgets the drafter state for an unexpected `reason`: the next forward starts it cold.
    ///
    /// The target cache stays exact; only the acceptance of the next rounds falls.
    private mutating func becomeCold(_ reason: String) {
        drafterTrackLogger.error("The drafter of rank 0 starts cold: \(reason)")
        phase = .cold
    }
}

/// A verify the cache holds, which the next op settles.
private struct RemoteVerify {

    /// The target hidden row of the position before the round's token.
    let previousBoundary: MLXArray

    /// The target hidden row of each verified position, `[1, drafts + 1, hidden]`.
    let hidden: MLXArray

    /// The drafts, `[1, drafts]`.
    let drafts: MLXArray

    /// The positions the verify fed: the round's token and the drafts.
    var positions: Int { drafts.dim(1) + 1 }
}
