// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Feeds a shifted MTP drafter the target hidden rows of a prompt, one prefill chunk at a time.
///
/// A shifted drafter (see ``ResumableMTPDrafterModel``) pairs each token with the target hidden
/// row of the position before it. After the chunks have fed the target positions up to `p`, the
/// drafter state holds an entry for each fed position but the last, and ``boundaryHidden`` holds
/// the row of the last one: the form of a finalized ``MTPDrafterContinuation`` at `p`. The final
/// prompt segment and the first bonus token then resume the drafter from that boundary.
///
/// A class, because the target's prefill reaches it through the ``MTPPrefillChunkHandler`` it
/// captures, and the iterator reads what the chunks left once the prefill returns.
final class MTPDrafterPrefillFeed {

    private let drafter: any ResumableMTPDrafterModel
    private let target: any LanguageModel
    private let sampler: any LogitSampler

    /// The drafter state the chunks extend.
    private(set) var state: MTPDrafterState

    /// The target hidden row of the last fed position, whose entry waits for the token after it.
    /// `nil` before the first chunk of a drafter that starts without a boundary.
    private(set) var boundaryHidden: MLXArray?

    /// The prompt tokens the chunks fed.
    private(set) var fedTokenCount = 0

    /// Whether each chunk reached the drafter state. Once an append is refused, the chunks after it
    /// are only counted.
    private(set) var isIntact = true

    /// The position the drafter state restarts at when the first chunk arrives, or `nil` when the
    /// chunks extend the state as it is.
    ///
    /// The restart waits for a chunk because the state's cache layers are the stream's own: a
    /// prefill that calls no chunk handler must leave them as the stream had them.
    private var pendingRestartPosition: Int?

    /// Starts a feed at `cachePosition`, the position of the first prompt token the target
    /// prefills. Nothing in `state` changes before the first chunk arrives.
    ///
    /// - Parameters:
    ///   - drafter: The drafter the chunks feed.
    ///   - target: The target model of the drafter.
    ///   - sampler: The sampler of the drafter's seed proposals.
    ///   - state: The drafter state of the stream: the state of `continuation` when there is one.
    ///   - cachePosition: The target cache position before the prefill.
    ///   - continuation: The drafter continuation carried with the target cache, if any. A
    ///     continuation aligned with `cachePosition` keeps its entries and its boundary; else the
    ///     first chunk starts the drafter empty at `cachePosition`.
    init(
        drafter: any ResumableMTPDrafterModel, target: any LanguageModel,
        sampler: any LogitSampler, state: MTPDrafterState, cachePosition: Int,
        continuation: MTPDrafterContinuation?
    ) {
        boundaryHidden = continuation?.alignedBoundary(
            withMainTokenCount: cachePosition, drafterID: drafter.drafterID)
        if boundaryHidden == nil {
            pendingRestartPosition = cachePosition
        }
        self.drafter = drafter
        self.target = target
        self.sampler = sampler
        self.state = state
    }

    /// The handler a target's chunked prefill calls with each chunk.
    var chunkHandler: MTPPrefillChunkHandler {
        { [self] tokens, hiddenStates, positionDeltas in
            append(tokens: tokens, hiddenStates: hiddenStates, positionDeltas: positionDeltas)
        }
    }

    /// Appends the entries one prefill chunk completes, and keeps the chunk's last hidden row as
    /// the new boundary.
    ///
    /// The drafter cache is evaluated asynchronously right away, so the chunk's hidden rows are
    /// released once the next chunk replaces the boundary, and no row is copied to the CPU.
    ///
    /// - Parameters:
    ///   - tokens: The tokens of the chunk, shape `[B, n]` or `[n]`.
    ///   - hiddenStates: One target hidden row per token of the chunk, shape `[B, n, hidden]`.
    ///   - positionDeltas: The position deltas the drafter applies to the chunk.
    func append(tokens: MLXArray, hiddenStates: MLXArray, positionDeltas: MLXArray?) {
        let tokens = normalizedMTPTokenBatch(tokens)
        let count = tokens.dim(-1)
        guard count > 0 else { return }
        restartIfPending()
        let leadingRows = hiddenStates[0..., ..<(count - 1), 0...]
        let entryTokens = boundaryHidden == nil ? tokens[0..., 1...] : tokens
        let entryHidden =
            boundaryHidden.map { concatenated([$0, leadingRows], axis: 1) } ?? leadingRows
        if isIntact, entryTokens.dim(-1) > 0 {
            isIntact = drafter.appendDrafterState(
                target: target, tokens: entryTokens, targetHidden: entryHidden,
                positionDeltas: positionDeltas, state: &state, sampler: sampler)
        }
        // Only the final segment's seed starts the first round, and an unused seed would keep
        // this chunk's graph alive.
        state.seedToken = nil
        state.seedHidden = nil
        let boundary = hiddenStates[0..., (count - 1)..., 0...]
        boundaryHidden = boundary
        fedTokenCount += count
        asyncEval(state.cache.flatMap(\.state) + [boundary])
    }

    /// Starts the drafter state empty at ``pendingRestartPosition`` before the first chunk, so the
    /// chunks append to an empty cache. When the entries cannot be removed, no chunk reaches the
    /// drafter state.
    private func restartIfPending() {
        guard let position = pendingRestartPosition else { return }
        pendingRestartPosition = nil
        if !state.restart(at: position) {
            isIntact = false
        }
    }
}
