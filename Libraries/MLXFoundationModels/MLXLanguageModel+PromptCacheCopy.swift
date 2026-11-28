// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// What ``MLXLanguageModel/copyPromptCache(fromSession:toSession:)`` did.
public enum PromptCacheCopyResult: Sendable, Hashable {

    /// Why the cache did not copy, although the source has a cache.
    public enum Refusal: Sendable, Hashable {

        /// The source and the destination are the same session.
        case sameSession

        /// The destination has a cache already. Release it first with
        /// ``MLXLanguageModel/releasePromptCache(sessionID:)``.
        case destinationHasCache

        /// A pass of the destination runs now and holds its cache.
        case destinationInUse

        /// No entry of the source could be copied: each file on disk failed to copy, and the
        /// worker of a clustered model failed the copy of each remote cache. The destination
        /// has no cache, and the source keeps its cache. The log has the error.
        case copyFailed

        /// The copy was to keep more tokens than the token ledger of the source holds, under
        /// each of its KV-cache configurations
        /// (``MLXLanguageModel/copyPromptCache(fromSession:toSession:keepingTokens:)``).
        case tokenCountBeyondLedger

        /// No entry of the source can keep a prefix at or below the token count of the copy
        /// (``MLXLanguageModel/copyPromptCache(fromSession:toSession:keepingTokens:)``): its
        /// caches cannot rewind to that count and it keeps no checkpoint at or below it, or it is
        /// a file on disk, which the copy can clone only whole.
        case noPrefixToKeep
    }

    /// The destination has a copy of the cache of the source. `tokenCount` is the length of
    /// the token ledger of the copy: the most tokens over the KV-cache configurations of the
    /// copy.
    case copied(tokenCount: Int)

    /// The store has no cache of the source for this model.
    case sourceNotFound

    /// A pass of the source runs now and holds its cache. Nothing is copied. Copy again after
    /// the pass, or let the first pass of the destination read its prompt in full.
    case sourceInUse

    /// Nothing is copied, for the reason.
    case refused(Refusal)
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel {

    /// Copies the prompt cache of one session of this model to a new session.
    ///
    /// Use it when the host makes a fork: call it before the first pass of the fork, with the
    /// identifier that the fork binds in ``promptCacheScope``. The first pass of the fork then
    /// compares its render with the token ledger of the copy and reads only the tokens after
    /// the common prefix. Do not bind the scope of the source for the fork: the pass of the
    /// fork would then take the cache of the source and leave its own tokens in it.
    ///
    /// The copy is whole: it has every token of the source. A render of the fork that parts
    /// from the source before its end rewinds the copy with the usual reuse rules. A hybrid
    /// model cannot rewind its recurrent layers: its pass goes back to the longest checkpoint of
    /// the copy at or before the point where the two renders part (the end of the last message
    /// of the prompt of the last pass, or the end of the instructions), or it reads the prompt
    /// in full. Thus the fork never resumes a recurrent state after that point, and a fork that
    /// keeps only the instructions of the source still reads them from the copy.
    ///
    /// The two sessions are independent after the copy. A pass of one session does not change
    /// the cache of the other, and a release of one session leaves the other.
    ///
    /// - Every KV-cache configuration: the source has one cache for each configuration that
    ///   its passes used, and each one is copied to the same configuration.
    /// - Copy-on-write: each cache of the copy is a new object, but its arrays share their
    ///   buffers with the source until one session writes into them. A file on disk is
    ///   cloned, which on APFS shares the blocks of the file.
    /// - Memory budget: the copy goes into the store like the cache at the end of a pass. It
    ///   is the most recently used entry, and it counts the bytes of the source in
    ///   ``promptCacheUsage``, also the bytes that it shares with the source: while both
    ///   sessions live, the shared bytes count two times. The budget
    ///   applies at once, thus the least recently used entries can go to disk.
    /// - A clustered model: the worker copies the caches of the source under a new handle.
    ///   The copy counts its bytes on the workers against the worker budget
    ///   (``configurePromptCache(workerBudgetBytes:)``), and the least recently used remote
    ///   caches can be released for it, but not the source. When the worker fails the copy,
    ///   the call returns ``PromptCacheCopyResult/Refusal/copyFailed``, and the source keeps
    ///   its cache. A lost member of the cluster fails the next pass of the model.
    /// - A source that a pass holds: the call does not wait. It returns
    ///   ``PromptCacheCopyResult/sourceInUse`` and copies nothing.
    ///
    /// The call does not throw. When it copies nothing, the first pass of the destination
    /// reads its prompt in full, as it does without a copy.
    ///
    /// - Parameters:
    ///   - source: the session to copy, as the host binds it with
    ///     `MLXLanguageModel.$promptCacheScope.withValue(.session(id))`.
    ///   - destination: the new session. It must have no cache.
    /// - Returns: what the copy did.
    public func copyPromptCache(
        fromSession source: String, toSession destination: String
    ) async -> PromptCacheCopyResult {
        await ExecutorPromptCacheStore.current.copy(
            modelID: .init(modelID), from: .init(source), to: .init(destination))
    }

    /// Copies the first tokens of the prompt cache of one session of this model to a new
    /// session.
    ///
    /// Use it to make a template that a host seeds sessions from: a session whose ledger holds
    /// the instructions, a first prompt and its answer gives a copy of the instructions alone,
    /// and each session seeded from that copy reads the instructions from it, whatever its first
    /// prompt is. The rules of ``copyPromptCache(fromSession:toSession:)`` hold, but each copy
    /// keeps a prefix of the token ledger of the source:
    ///
    /// - Caches that can rewind `tokenCount` tokens from the end of the ledger keep exactly the
    ///   first `tokenCount` tokens. The copy rewinds, and the source does not change.
    /// - Caches that cannot rewind that far (a hybrid model, or a sliding window that wrapped)
    ///   keep the longest checkpoint of the source at or below `tokenCount`: the end of the
    ///   instructions, or the end of the last message of the prompt of the last pass. The copy
    ///   goes back to that checkpoint.
    /// - A ledger of `tokenCount` tokens or fewer is copied whole.
    /// - A file on disk is cloned only whole: when its ledger holds more than `tokenCount`
    ///   tokens, it gives no copy.
    ///
    /// The result names the longest kept ledger over the KV-cache configurations of the source.
    /// A `tokenCount` beyond the longest ledger of the source gives
    /// ``PromptCacheCopyResult/Refusal/tokenCountBeyondLedger``, and a source that can keep no
    /// prefix at or below it gives ``PromptCacheCopyResult/Refusal/noPrefixToKeep``. Neither
    /// copies anything.
    ///
    /// - Parameters:
    ///   - source: the session to copy, as the host binds it with
    ///     `MLXLanguageModel.$promptCacheScope.withValue(.session(id))`.
    ///   - destination: the new session. It must have no cache.
    ///   - tokenCount: the most leading tokens of the ledger of the source the copy keeps.
    /// - Returns: what the copy did.
    public func copyPromptCache(
        fromSession source: String, toSession destination: String,
        keepingTokens tokenCount: Int
    ) async -> PromptCacheCopyResult {
        await ExecutorPromptCacheStore.current.copy(
            modelID: .init(modelID), from: .init(source), to: .init(destination),
            keepingTokens: tokenCount)
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
