// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

// MARK: - What a session carries

/// The live key/value cache of one session, and the tokens it holds.
///
/// This is a reference type because the caches it carries are reference types
/// that generation writes into, and the model state beside them is not
/// `Sendable`. ``ExecutorPromptCacheStore`` hands one entry to at most one turn
/// at a time -- a check-out REMOVES the entry from the store -- thus no two
/// tasks write the same caches or read the same state, and the
/// `@unchecked Sendable` conformance rests on that.
// swiftlint:disable:next no_unchecked_sendable The store hands an entry to at most one turn at a time (a check-out removes it from the store), thus no two tasks write its caches or read its state at once.
final class ExecutorPromptCacheEntry: @unchecked Sendable {

    /// One cache for each layer of the model.
    let caches: [KVCache]

    /// The exact prompt tokens `caches` represents, in order.
    let tokens: [Int]

    /// The whole prompt the last pass rendered, which is not the same as
    /// `tokens`: the ledger also holds the tokens the model generated after
    /// that render.
    ///
    /// A protocol rule compares this with the next render to prove that the
    /// template rewrote no cached region before it splices past the turn the
    /// model committed. Empty means no render is on record, thus no rule may
    /// splice.
    let renderTokens: [Int]

    /// The tokens the last pass generated and `caches` does not hold, which is
    /// what ``ChatSession`` keeps as its uncommitted tokens.
    ///
    /// An MTP round returns its final verified token before any forward pass
    /// feeds it, thus a pass that stops on its token budget there leaves that
    /// token here. The next turn hands them to the reuse rules, and a splice
    /// feeds them ahead of the commit that closes the turn, thus the model
    /// reads its whole answer. Empty after a plain decode.
    let uncommittedTokens: [Int]

    /// The model state the last pass left with `caches`, or nil for a model
    /// that carries none.
    ///
    /// A Qwen VL model keys its M-RoPE anchor here and refuses a warm cache
    /// without it, thus the next turn seeds its iterator with this state. The
    /// state is tied to the prefill that made it, thus the caches never rewind
    /// under it.
    let state: LMOutput.State?

    /// The recurrent state of a hybrid model at an earlier position of
    /// `tokens`, or nil when the entry carries none.
    ///
    /// A next render that parts from `tokens` before their end cannot rewind
    /// the recurrent layers, but it can go back to this checkpoint when it
    /// agrees with `tokens` up to the checkpoint. The checkpoint lives in
    /// memory only: ``ExecutorPromptCacheFile`` does not write it, thus an
    /// entry that comes back from disk carries none.
    let checkpoint: PromptCacheCheckpoint?

    /// The position of `checkpoint`: the number of tokens the caches held
    /// there, or nil when the entry carries no checkpoint.
    var checkpointPosition: Int? {
        checkpoint?.tokens.count
    }

    /// The recurrent state of a hybrid model at the end of the instructions
    /// of `tokens`, or nil when the entry carries none.
    ///
    /// `checkpoint` stands at the end of the last message of the render, and
    /// it serves only a render that keeps that whole message. This one serves
    /// every render that starts with the same instructions, such as the first
    /// render of a session seeded with a copy of a template whose prompt was
    /// another. It lives in memory only, as `checkpoint` does.
    let instructionsCheckpoint: PromptCacheCheckpoint?

    /// The checkpoints the entry carries: `checkpoint`, then
    /// `instructionsCheckpoint`.
    var checkpoints: [PromptCacheCheckpoint] {
        Self.checkpoints(checkpoint, instructionsCheckpoint)
    }

    /// The ``checkpoints`` that satisfy `predicate`, longest first.
    ///
    /// A caller takes the first of them that restores, thus each choice of a
    /// checkpoint prefers the one that keeps the most tokens.
    ///
    /// - Parameter predicate: whether a checkpoint is a candidate.
    /// - Returns: the candidates, sorted by token count, longest first.
    func checkpoints(
        longestWhere predicate: (PromptCacheCheckpoint) -> Bool
    ) -> [PromptCacheCheckpoint] {
        checkpoints
            .filter(predicate)
            .sorted { $0.tokens.count > $1.tokens.count }
    }

    /// The checkpoints of the two that are present, in order.
    ///
    /// - Parameters:
    ///   - checkpoint: the checkpoint at the end of the last message, or nil.
    ///   - instructionsCheckpoint: the checkpoint at the end of the
    ///     instructions, or nil.
    /// - Returns: the checkpoints that are not nil.
    private static func checkpoints(
        _ checkpoint: PromptCacheCheckpoint?, _ instructionsCheckpoint: PromptCacheCheckpoint?
    ) -> [PromptCacheCheckpoint] {
        [checkpoint, instructionsCheckpoint].compactMap { $0 }
    }

    /// The MTP drafter continuation the last pass left where `caches` end,
    /// or nil when that pass did not speculate with a resumable drafter.
    ///
    /// The next turn resumes it when its render extends `tokens` and its
    /// drafter has the identity the continuation records; else that turn
    /// starts its drafter cold. It is not model state: the target never
    /// reads it.
    let drafterState: MTPDrafterContinuation?

    /// The bytes this entry holds in the memory of this process: the resident
    /// bytes of every local cache, plus the arrays of `state`, plus the bytes
    /// of each of ``checkpoints``, plus the bytes of `drafterState`.
    ///
    /// A ``RemoteKVCache`` counts no byte here: its buffers live on its worker,
    /// and ``workerByteCount`` counts them. The memory budget of the store thus
    /// does not apply to them.
    ///
    /// The count reads only shapes and element types, thus it evaluates
    /// nothing. It is computed one time, when the entry is made, because the
    /// store reads it at each check-in and each eviction. The checkpoint and
    /// the drafter state are in the count, thus the memory budget of the
    /// store controls them too.
    let byteCount: Int

    /// Whether a cache of this entry is a ``RemoteKVCache``.
    ///
    /// A prompt cache file cannot hold such a cache, thus a spill of the entry
    /// would drop it, and the next turn of its session would start cold. The
    /// store never spills such an entry for its budget.
    let holdsRemoteCache: Bool

    /// The bytes this entry holds on the workers of a clustered model: the
    /// resident bytes of every ``RemoteKVCache``, plus the bytes of the remote
    /// copies of each of ``checkpoints``. Zero for an entry of a local model.
    ///
    /// The store limits the sum over its entries to
    /// ``ExecutorPromptCacheStore/workerBudgetBytes``. It is computed one time,
    /// when the entry is made, from the status each worker reported last.
    let workerByteCount: Int

    /// Creates an entry for caches that hold `tokens`.
    ///
    /// - Parameters:
    ///   - caches: one cache for each layer of the model.
    ///   - tokens: the exact tokens `caches` represents.
    ///   - renderTokens: the whole prompt the last pass rendered, or empty
    ///     when no render is on record.
    ///   - uncommittedTokens: the tokens the last pass generated that `caches`
    ///     does not hold.
    ///   - state: the model state the last pass left, or nil.
    ///   - checkpoint: the recurrent state at an earlier position of
    ///     `tokens`, or nil.
    ///   - instructionsCheckpoint: the recurrent state at the end of the
    ///     instructions of `tokens`, or nil.
    ///   - drafterState: the MTP drafter continuation at the end of
    ///     `tokens`, or nil.
    ///   - minimumByteCount: the fewest bytes ``byteCount`` counts. A copy
    ///     gives the count of its source here, because its arrays keep the
    ///     whole buffers of the source alive until a write.
    init(
        caches: [KVCache], tokens: [Int], renderTokens: [Int] = [],
        uncommittedTokens: [Int] = [], state: LMOutput.State? = nil,
        checkpoint: PromptCacheCheckpoint? = nil,
        instructionsCheckpoint: PromptCacheCheckpoint? = nil,
        drafterState: MTPDrafterContinuation? = nil, minimumByteCount: Int = 0
    ) {
        self.caches = caches
        self.tokens = tokens
        self.renderTokens = renderTokens
        self.uncommittedTokens = uncommittedTokens
        self.state = state
        self.checkpoint = checkpoint
        self.instructionsCheckpoint = instructionsCheckpoint
        self.drafterState = drafterState
        self.holdsRemoteCache = caches.contains(where: \.isRemote)
        let checkpoints = Self.checkpoints(checkpoint, instructionsCheckpoint)
        self.workerByteCount =
            caches.reduce(0) { $0 + $1.residentByteCount(onWorker: true) }
            + checkpoints.reduce(0) { $0 + $1.workerByteCount }
        self.byteCount = max(
            minimumByteCount,
            caches.reduce(0) { $0 + $1.residentByteCount(onWorker: false) }
                + (state?.residentByteCount ?? 0)
                + checkpoints.reduce(0) { $0 + $1.byteCount } + (drafterState?.byteCount ?? 0))
    }

    /// The instructions checkpoint, when `promptTokens` starts with its
    /// tokens and has more to feed, thus when a pass of `promptTokens` keeps
    /// it for the next turn.
    ///
    /// - Parameter promptTokens: the whole rendered prompt of the pass.
    /// - Returns: `instructionsCheckpoint`, or nil when the entry carries none
    ///   or the prompt does not keep its instructions.
    func instructionsCheckpoint(serving promptTokens: [Int]) -> PromptCacheCheckpoint? {
        instructionsCheckpoint.flatMap { $0.serves(promptTokens) ? $0 : nil }
    }

    /// The drafter state a pass that extends `tokens` with the drafter
    /// `drafterID` resumes.
    ///
    /// - Parameter drafterID: the identity of the drafter of the pass, or nil
    ///   when the pass does not speculate with a resumable drafter.
    /// - Returns: `drafterState`, or nil when the entry keeps none, when
    ///   another drafter made it, or when it does not stand at the end of
    ///   `tokens`. The drafter of the pass then starts cold.
    func drafterState(resumableBy drafterID: MTPDrafterID?) -> MTPDrafterContinuation? {
        guard let drafterID, let drafterState,
            drafterState.isAligned(withMainTokenCount: tokens.count, drafterID: drafterID)
        else {
            return nil
        }
        return drafterState
    }

    /// An entry that holds the same tokens and the same state as this entry, and that a pass
    /// can write without a change to this entry.
    ///
    /// Each cache is a new object (``KVCache/checkedCopy()``), thus the two entries never share
    /// a mutable cache. The worker of a ``RemoteKVCache`` copies its caches under a new handle.
    /// The arrays of a copied local cache share their buffers with the arrays of this
    /// entry: an MLX array does not change in place, and a write into a cache makes a new
    /// array. A buffer that two arrays hold is not donated to a write, thus the first write of
    /// one session copies the buffer, and the other session keeps the old buffer
    /// (copy-on-write). The drafter state is copied with ``MTPDrafterContinuation/copy()``.
    /// The model state and the checkpoints are values that no pass writes: a checkpoint gives a
    /// new copy of its caches at each restore. Thus the two entries share them.
    ///
    /// The copy counts at least the ``byteCount`` of this entry. A copied cache holds slices
    /// of the buffers of this entry, and a slice keeps its whole buffer alive, the step
    /// padding included, until a write or until this entry is released. While both entries
    /// live, the memory budget of the store thus counts the shared bytes two times, which is
    /// conservative.
    ///
    /// - Returns: the new entry.
    /// - Throws: the error of the copy of a ``RemoteKVCache`` on its worker. The copies made
    ///   before it are dropped, thus their workers release them, and this entry does not
    ///   change.
    func copy() throws -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(
            caches: try caches.map { try $0.checkedCopy() }, tokens: tokens,
            renderTokens: renderTokens,
            uncommittedTokens: uncommittedTokens, state: state, checkpoint: checkpoint,
            instructionsCheckpoint: instructionsCheckpoint, drafterState: drafterState?.copy(),
            minimumByteCount: byteCount)
    }

    /// An entry that holds the first `tokenCount` tokens of `tokens`, or the
    /// longest checkpoint of this entry at or below them, and that a pass can
    /// write without a change to this entry.
    ///
    /// - A ledger of `tokenCount` tokens or fewer gives ``copy()``.
    /// - Caches that can rewind to `tokenCount` give copies of the caches
    ///   (``KVCache/checkedCopy()``) trimmed to `tokenCount`. An entry that
    ///   carries model state does not rewind, because the state is tied to
    ///   the prefill that made it.
    /// - Any other entry goes back to the longest of ``checkpoints`` at or
    ///   below `tokenCount`, from copies (``PromptCacheCheckpoint/restoredCopy(of:)``).
    ///
    /// The new entry keeps no render and no uncommitted tokens: both stand
    /// past the kept prefix. It keeps each checkpoint of this entry at or
    /// below the kept prefix.
    ///
    /// - Parameter tokenCount: the most leading tokens of `tokens` to keep.
    /// - Returns: the new entry, or nil when no prefix of at most `tokenCount`
    ///   tokens and at least one token can be kept.
    /// - Throws: the error of the copy of a ``RemoteKVCache`` on its worker.
    ///   The copies made before it are dropped, thus their workers release
    ///   them, and this entry does not change.
    func copy(keepingTokens tokenCount: Int) throws -> ExecutorPromptCacheEntry? {
        guard tokenCount < tokens.count else { return try copy() }
        guard tokenCount > 0 else { return nil }
        if let trimmed = try trimmedCopy(keepingTokens: tokenCount) {
            return trimmed
        }
        return try restoredCopy(keepingTokens: tokenCount)
    }

    /// Copies of the caches trimmed to `tokenCount`, or nil when the caches
    /// cannot rewind there.
    ///
    /// - Parameter tokenCount: the tokens to keep, fewer than `tokens`.
    /// - Returns: the new entry, or nil.
    /// - Throws: the error of the copy of a ``RemoteKVCache`` on its worker.
    private func trimmedCopy(keepingTokens tokenCount: Int) throws -> ExecutorPromptCacheEntry? {
        guard state == nil, canTrimPromptCache(caches, numTokens: tokens.count - tokenCount)
        else {
            return nil
        }
        let trimmed = try caches.map { try $0.checkedCopy() }
        guard rewindPromptCache(trimmed, to: tokenCount) else { return nil }
        // The trimmed arrays keep the whole buffers of this entry alive until
        // a write, as a whole copy does.
        return prefixEntry(
            caches: trimmed, tokens: Array(tokens.prefix(tokenCount)), state: nil,
            drafterState: nil, minimumByteCount: byteCount)
    }

    /// The caches at the longest of ``checkpoints`` at or below `tokenCount`,
    /// or nil when no checkpoint there restores.
    ///
    /// - Parameter tokenCount: the most tokens to keep.
    /// - Returns: the new entry, or nil.
    /// - Throws: the error of the copy of a ``RemoteKVCache`` on its worker.
    private func restoredCopy(keepingTokens tokenCount: Int) throws -> ExecutorPromptCacheEntry? {
        let kept = checkpoints(longestWhere: { $0.tokens.count <= tokenCount })
        for checkpoint in kept {
            if let restored = try checkpoint.restoredCopy(of: caches) {
                return prefixEntry(
                    caches: restored, tokens: checkpoint.tokens, state: checkpoint.state,
                    drafterState: checkpoint.drafterContinuation?.copy())
            }
        }
        return nil
    }

    /// An entry of `caches` that hold `tokens`, a prefix of the ledger of this
    /// entry, with each checkpoint of this entry at or below that prefix.
    ///
    /// - Parameters:
    ///   - caches: the caches of the new entry.
    ///   - tokens: the tokens `caches` represents.
    ///   - state: the model state at the end of `tokens`, or nil.
    ///   - drafterState: the MTP drafter continuation at the end of `tokens`,
    ///     or nil.
    ///   - minimumByteCount: the fewest bytes the new entry counts.
    /// - Returns: the new entry.
    private func prefixEntry(
        caches: [KVCache], tokens: [Int], state: LMOutput.State?,
        drafterState: MTPDrafterContinuation?, minimumByteCount: Int = 0
    ) -> ExecutorPromptCacheEntry {
        let kept = { (checkpoint: PromptCacheCheckpoint?) in
            checkpoint.flatMap { $0.tokens.count <= tokens.count ? $0 : nil }
        }
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: tokens, state: state, checkpoint: kept(checkpoint),
            instructionsCheckpoint: kept(instructionsCheckpoint), drafterState: drafterState,
            minimumByteCount: minimumByteCount)
    }
}

// MARK: - What a check-out gives

/// The file of one entry that the store wrote to disk, and the session it
/// belongs to.
///
/// The turn that gets a handle owns the file: the store keeps no record of it.
struct ExecutorPromptCacheSpilledHandle: Equatable, Sendable {

    /// The URL of the file.
    let url: URL

    /// The session the file belongs to.
    let key: ExecutorPromptCacheKey

    /// The key that the metadata of the file names, which the read of the file checks.
    ///
    /// It is `key`, unless the file is a copy of the file of another session
    /// (``ExecutorPromptCacheStore/copy(modelID:from:to:)``): a copy keeps the bytes, thus
    /// also the metadata, of the file of the source.
    let fileKey: ExecutorPromptCacheKey

    /// Creates the handle of the file at `url` of `key`.
    ///
    /// - Parameters:
    ///   - url: the URL of the file.
    ///   - key: the session the file belongs to.
    ///   - fileKey: the key that the metadata of the file names, or nil when it is `key`.
    init(url: URL, key: ExecutorPromptCacheKey, fileKey: ExecutorPromptCacheKey? = nil) {
        self.url = url
        self.key = key
        self.fileKey = fileKey ?? key
    }
}

/// What ``ExecutorPromptCacheStore/checkOut(_:)`` finds for one session.
enum ExecutorPromptCacheCheckout: Sendable {

    /// The entry was in memory, or its write to disk had not ended.
    case memory(ExecutorPromptCacheEntry)

    /// The entry is on disk, in the file of the handle.
    case spilled(ExecutorPromptCacheSpilledHandle)

    /// The store holds nothing for the session.
    case none

    /// The entry in memory, or nil when the check-out found no entry in
    /// memory.
    var entry: ExecutorPromptCacheEntry? {
        switch self {
        case .memory(let entry): entry
        case .spilled, .none: nil
        }
    }

    /// The file on disk, or nil when the check-out found no file.
    var spilledHandle: ExecutorPromptCacheSpilledHandle? {
        switch self {
        case .spilled(let handle): handle
        case .memory, .none: nil
        }
    }
}

extension ExecutorPromptCacheCheckout: Equatable {

    /// Two check-outs are equal when they give the same entry object, equal
    /// handles, or both give nothing.
    static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case (.memory(let lhsEntry), .memory(let rhsEntry)): lhsEntry === rhsEntry
        case (.spilled(let lhsHandle), .spilled(let rhsHandle)): lhsHandle == rhsHandle
        case (.none, .none): true
        case (.memory, _), (.spilled, _), (.none, _): false
        }
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
