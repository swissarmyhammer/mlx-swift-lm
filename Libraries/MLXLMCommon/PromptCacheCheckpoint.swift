// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The caches of a hybrid model at one position, kept so that a later turn can
/// go back to that position.
///
/// An attention cache rewinds to any earlier position, but a recurrent cache
/// (the linear layers of Qwen 3.5) holds one state that sums every token it
/// saw, and it cannot rewind. A rotating cache (a sliding-window layer) rewinds
/// only until its window wraps, which a later prefill can do. The checkpoint
/// thus keeps a copy of each cache that can lose its rewind, and nothing for
/// any other cache: a restore trims that cache back to the checkpoint. A
/// variance-normalized cache trims exactly only inside its raw tail, thus the
/// checkpoint keeps that tail alone (``VarianceNormalizedKVCache/Checkpoint``),
/// and a restore keeps the completed tiles of the live cache before it.
///
/// A copy shares the arrays of the live cache. MLX does not write into an array
/// that another array still holds, thus a copy keeps the state of the
/// checkpoint while the live caches move on. A rotating cache copy keeps only
/// its configured history (``RotatingKVCache/copyForCheckpoint()``), also inside
/// a ``CacheList``, so it does not pin the rows a multi-token prefill adds past
/// the window.
///
/// `ChatSession` and the executor of `MLXFoundationModels` both keep
/// checkpoints. Each one takes a checkpoint at the end of the last message of a
/// prompt (see ``boundaryProbeText``), because the generation prompt that
/// follows is the region a next render writes again.
package struct PromptCacheCheckpoint {

    /// The text of the user message that a caller adds to the messages of a
    /// prompt to find the end of the last message. The render of the messages
    /// plus this message parts from the prompt where the generation prompt
    /// starts. The text does not matter: the render parts before it.
    package static let boundaryProbeText = "."

    /// The tokens the caches represented at the checkpoint, in order.
    package let tokens: [Int]

    /// What a restore needs for one layer.
    private enum Saved {
        /// A trim takes the cache back to the checkpoint.
        case trim
        /// A copy of a cache that cannot rewind.
        case copy(KVCache)
        /// The raw tail of a variance-normalized cache.
        case tile(VarianceNormalizedKVCache.Checkpoint)

        /// The bytes this layer holds in the memory of this process. A copy of a remote
        /// cache lives on its worker, thus it counts none.
        var byteCount: Int {
            switch self {
            case .trim: 0
            case .copy(let copy): copy.localResidentByteCount
            case .tile(let checkpoint): checkpoint.retainedBytes
            }
        }
    }

    /// For each layer, what a restore needs.
    private let copies: [Saved]

    /// The model state the prefill left at the checkpoint, or nil.
    package let state: LMOutput.State?

    /// A compacted copy of the MTP drafter continuation at the checkpoint (see
    /// ``MTPDrafterContinuation/compacted()``), or nil when the caller gave
    /// none or gave one that does not end at the checkpoint.
    package let drafterContinuation: MTPDrafterContinuation?

    /// The bytes the copies, `state` and the drafter continuation hold in the
    /// memory of this process.
    package let byteCount: Int

    /// Takes a checkpoint of `caches`, which represent `tokens`.
    ///
    /// - Parameters:
    ///   - caches: the live caches, one for each layer of the model.
    ///   - tokens: the tokens `caches` represents.
    ///   - state: the model state the prefill left at this position, or nil.
    ///   - drafterContinuation: the live MTP drafter continuation at this
    ///     position, or nil. The checkpoint keeps a compacted copy of it, so
    ///     the live one can move on, and only when it is aligned with
    ///     `tokens`: a restore must never hand a drafter a state of another
    ///     position.
    /// - Returns: nil when no cache can lose its rewind, because then a rewind
    ///   serves every later render and a checkpoint adds nothing, or when a
    ///   cache does not stand at the end of `tokens`.
    package init?(
        caches: [KVCache], tokens: [Int], state: LMOutput.State?,
        drafterContinuation: MTPDrafterContinuation? = nil
    ) {
        guard Self.applies(to: caches), caches.allSatisfy({ $0.offset == tokens.count }) else {
            return nil
        }
        let copies = caches.map(Self.saved)
        let storedContinuation = drafterContinuation.flatMap { continuation in
            continuation.isAligned(
                withMainTokenCount: tokens.count, drafterID: continuation.drafterID)
                ? continuation.compacted() : nil
        }
        self.tokens = tokens
        self.copies = copies
        self.state = state
        self.drafterContinuation = storedContinuation
        self.byteCount =
            copies.reduce(0) { $0 + $1.byteCount }
            + (state?.residentByteCount ?? 0) + (storedContinuation?.byteCount ?? 0)
    }

    /// A copy of the drafter continuation this checkpoint keeps, which a
    /// restore hands to the generation that resumes the drafter `drafterID`
    /// here.
    ///
    /// Forks share a checkpoint, and a generation writes into the drafter
    /// caches it resumes. A restore thus hands out a copy, and the checkpoint
    /// keeps its own continuation for the next restore.
    ///
    /// - Parameter drafterID: the identity of the drafter of the generation.
    /// - Returns: the copy, or nil when the checkpoint keeps no continuation or
    ///   keeps one that the drafter `drafterID` cannot resume at this
    ///   checkpoint. The drafter then cold-starts here.
    package func drafterContinuation(resumableBy drafterID: MTPDrafterID)
        -> MTPDrafterContinuation?
    {
        guard let drafterContinuation,
            drafterContinuation.isAligned(withMainTokenCount: tokens.count, drafterID: drafterID)
        else {
            return nil
        }
        return drafterContinuation.copy()
    }

    /// Whether a model has a layer that cannot rewind, thus whether a
    /// checkpoint can serve it.
    ///
    /// - Parameter model: the model whose fresh caches are read.
    /// - Returns: `true` for a hybrid or sliding-window model.
    package static func applies(to model: any LanguageModel) -> Bool {
        guard let caches = try? model.newCache(parameters: nil) else { return false }
        return applies(to: caches)
    }

    /// Whether `caches` holds a cache that cannot rewind, or can stop being able
    /// to, thus whether a checkpoint can serve them.
    ///
    /// - Parameter caches: one cache for each layer of the model.
    /// - Returns: `true` for the caches of a hybrid or sliding-window model.
    package static func applies(to caches: [KVCache]) -> Bool {
        caches.contains(where: canLoseRewind)
    }

    /// Whether `caches` can lose a rewind while they take `positions` more
    /// tokens, thus whether a pass that feeds that many needs checkpoints.
    ///
    /// A rotating cache whose window the pass cannot wrap still rewinds after
    /// it, and a checkpoint would only cost a copy.
    ///
    /// - Parameters:
    ///   - caches: the live caches, one for each layer of the model.
    ///   - positions: the most tokens the pass can add, prompt and generation.
    /// - Returns: `true` when some cache cannot rewind after the pass.
    package static func applies(to caches: [KVCache], adding positions: Int) -> Bool {
        caches.contains { !$0.isTrimmable(after: positions) }
    }

    /// Whether `cache` cannot rewind now or can stop being able to: a cache with
    /// a window (``KVCache/maxSize``, as a rotating cache or a composite over
    /// one reports it) rewinds until its window wraps, a recurrent or
    /// convolution cache (an ``ArraysCache``) rewinds at most inside its
    /// rollback, and a variance-normalized cache rewinds exactly only inside its
    /// raw tail. A ``CacheList`` can lose its rewind when one of its children
    /// can. A remote cache answers for the layers it holds on its worker.
    ///
    /// It reads no ``KVCache/isTrimmable(after:)``, which a cache may answer
    /// with a side effect for the width of a pass it is asked about.
    private static func canLoseRewind(_ cache: KVCache) -> Bool {
        if let remote = cache as? RemoteKVCache {
            return remote.status.canLoseRewind
        }
        if let list = cache as? CacheList {
            return list.children.contains(where: canLoseRewind)
        }
        return !cache.isTrimmable || cache.maxSize != nil || cache is ArraysCache
            || cache is VarianceNormalizedKVCache
    }

    /// What a restore of `cache` needs.
    private static func saved(_ cache: KVCache) -> Saved {
        switch cache {
        case let tiles as VarianceNormalizedKVCache:
            return .tile(tiles.checkpoint())
        case _ where canLoseRewind(cache):
            return .copy(checkpointCopy(of: cache))
        default:
            return .trim
        }
    }

    /// A copy of `cache` for a checkpoint. A rotating cache keeps only its
    /// configured history, also inside a ``CacheList``.
    private static func checkpointCopy(of cache: KVCache) -> KVCache {
        switch cache {
        case let rotating as RotatingKVCache:
            rotating.copyForCheckpoint()
        case let list as CacheList:
            CacheList(caches: list.children.map(checkpointCopy))
        default:
            cache.copy()
        }
    }

    /// Whether this checkpoint can serve `promptTokens`: the prompt starts with
    /// the tokens of the checkpoint and has at least one token more to feed.
    ///
    /// - Parameter promptTokens: the whole rendered prompt.
    /// - Returns: `true` when a restore can serve the prompt.
    package func serves(_ promptTokens: [Int]) -> Bool {
        promptTokens.count > tokens.count && promptTokens.starts(with: tokens)
    }

    /// Takes `caches` back to this checkpoint.
    ///
    /// A cache that cannot rewind is replaced by a copy of its checkpoint copy,
    /// thus the checkpoint itself stays as it is. Any other cache is trimmed
    /// back to the checkpoint.
    ///
    /// - Parameter caches: the live caches, one for each layer of the model.
    /// - Returns: the caches at the checkpoint, or nil when a cache did not land
    ///   on the checkpoint and the caller must build new ones.
    package func restored(onto caches: [KVCache]) -> [KVCache]? {
        guard caches.count == copies.count else { return nil }
        var restored: [KVCache] = []
        for (cache, saved) in zip(caches, copies) {
            switch saved {
            case .copy(let copy):
                restored.append(copy.copy())
            case .tile(let checkpoint):
                guard let tiles = cache as? VarianceNormalizedKVCache,
                    tiles.canRestore(checkpoint)
                else { return nil }
                tiles.restore(checkpoint)
                restored.append(tiles)
            case .trim:
                guard rewindPromptCache([cache], to: tokens.count) else { return nil }
                restored.append(cache)
            }
        }
        return restored
    }

    /// Feeds `tokens` into `caches` in forward passes, the way a prefill does,
    /// and gives the model state the last pass left.
    ///
    /// A caller feeds the prompt up to a checkpoint with this function, takes
    /// the checkpoint, and leaves the rest of the prompt to the prefill of the
    /// generation.
    ///
    /// - Parameters:
    ///   - tokens: the tokens to feed, which follow what `caches` holds.
    ///   - caches: the live caches, one for each layer of the model.
    ///   - state: the model state to seed the first pass with, or nil.
    ///   - model: the model that feeds the caches.
    ///   - prefill: cuts the tokens into forward passes, in steps of the
    ///     ``LanguageModel/defaultPrefillStepSize`` of `model` when it sets no
    ///     step size, and never above the
    ///     ``LanguageModel/maximumPrefillStepSize`` of `model`.
    ///   - drafterFeed: takes the target hidden rows of each forward pass into
    ///     an MTP drafter, or nil. The returned state then holds none of the
    ///     values the model emits for the drafter.
    /// - Returns: the model state the last pass left.
    /// - Throws: `CancellationError` when the task is cancelled between two
    ///   forward passes.
    package static func prefill(
        _ tokens: [Int], into caches: [KVCache], state: LMOutput.State?,
        model: any LanguageModel, prefill: PrefillParameters,
        drafterFeed: MTPPrefillChunkHandler? = nil
    ) throws -> LMOutput.State? {
        var state = state
        let forward = { (range: Range<Int>) in
            let chunk = MLXArray(Array(tokens[range])).expandedDimensions(axis: 0)
            let chunkState = drafterFeed.map { (state ?? LMOutput.State()).feeding($0) } ?? state
            // The logits are never read, thus the model may skip all but the last row.
            let output = model.nextTokenLogits(
                LMInput.Text(tokens: chunk), cache: caches, state: chunkState)
            chunkState?.feedMTPPrefillChunk(tokens: chunk, output: output)
            state = drafterFeed == nil ? output.state : output.state?.withoutMTPExchange
            asyncEval(caches.flatMap(\.state))
        }
        let chunked = try prefill.forEachChunk(
            total: tokens.count, reserving: 0, defaultStepSize: model.defaultPrefillStepSize,
            maximumStepSize: model.maximumPrefillStepSize, forward)
        if chunked < tokens.count {
            forward(chunked ..< tokens.count)
        }
        eval(caches.flatMap(\.state))
        return state
    }

    /// Feeds `uncommittedTokens` and then the prompt from `start` up to
    /// `boundary` into `caches`, and takes a checkpoint there.
    ///
    /// - Parameters:
    ///   - boundary: the index of the prompt where the checkpoint goes.
    ///   - promptTokens: the whole rendered prompt of the pass.
    ///   - representedTokens: the tokens the caches represent once the pass has
    ///     fed the whole prompt. A protocol rule can keep generated tokens here
    ///     that the render does not hold.
    ///   - start: the leading prompt tokens the caches already hold.
    ///   - uncommittedTokens: the tokens the previous generation returned and
    ///     the caches miss, which a splice feeds ahead of the prompt (see
    ///     ``PromptCacheReuse/uncommittedTokens``). They go in the same forward
    ///     passes as the prompt tokens.
    ///   - caches: the live caches, one for each layer of the model.
    ///   - state: the model state to seed the first forward pass with, or nil.
    ///   - model: the model that feeds the caches.
    ///   - prefill: cuts the tokens into forward passes.
    ///   - drafterFeed: the feed of an MTP drafter that stands at `start`, or
    ///     nil. The fed tokens go into the drafter too, so the drafter
    ///     continuation stands at `boundary` and the checkpoint keeps it.
    ///   - realize: runs after the feed and before the checkpoint, and gives
    ///     the caches to checkpoint, or nil to checkpoint `caches`. A cache
    ///     plan whose conversion must run before a checkpoint (see
    ///     ``KVCachePlan/requiresRealizedCheckpoint``) applies it here.
    /// - Returns: the model state the fed tokens left, the checkpoint at
    ///   `boundary`, which is nil when a cache did not stand there, and the live
    ///   drafter continuation at `boundary`, which is nil without a feed or
    ///   when a chunk did not reach the drafter. Nil when `boundary` is not
    ///   after `start` or not before the end of the prompt: the caches hold
    ///   that position already, or only the prefill of the generation reaches
    ///   it.
    /// - Throws: `CancellationError` when the task is cancelled between two
    ///   forward passes.
    package static func prefillToCheckpoint(
        at boundary: Int, of promptTokens: [Int], representing representedTokens: [Int],
        from start: Int, after uncommittedTokens: [Int] = [], into caches: [KVCache],
        state: LMOutput.State?, model: any LanguageModel, prefill: PrefillParameters,
        drafterFeed: MTPDrafterPrefillFeed? = nil,
        realize: (() throws -> [KVCache])? = nil
    ) throws -> (
        state: LMOutput.State?, checkpoint: PromptCacheCheckpoint?,
        drafterContinuation: MTPDrafterContinuation?
    )? {
        guard boundary > start, boundary < promptTokens.count else { return nil }
        let fedState = try Self.prefill(
            uncommittedTokens + promptTokens[start ..< boundary], into: caches, state: state,
            model: model, prefill: prefill, drafterFeed: drafterFeed?.chunkHandler)
        let drafterContinuation = drafterFeed?.continuation
        let checkpoint = PromptCacheCheckpoint(
            caches: try realize?() ?? caches,
            tokens: Array(representedTokens.dropLast(promptTokens.count - boundary)),
            state: fedState, drafterContinuation: drafterContinuation)
        return (fedState, checkpoint, drafterContinuation)
    }

    /// The index of `prompt` where its last message ends: the start of the
    /// generation prompt that the chat template writes after it.
    ///
    /// A chat template writes a generation prompt after the last message (Qwen
    /// 3.8: `<think>\n`), and a next render can write the same assistant turn
    /// in a different form (`<think>\n\n</think>`). The next render thus parts
    /// from the cached tokens inside the generation prompt, where the recurrent
    /// caches of a hybrid model cannot rewind. A checkpoint at the end of the
    /// last message serves that render.
    ///
    /// The index is the prefix that `prompt` shares with a render of
    /// `messages` plus one user message with the text ``boundaryProbeText``
    /// (the full probe). With no generation prompt, the index is the end of
    /// `prompt`.
    ///
    /// The full probe costs as much as the render of `prompt`, thus this
    /// function renders the full probe only when it must. When the last
    /// message is a user message, it finds the tail that the generation prompt
    /// writes from a small sample conversation (see
    /// ``generationPromptTail(after:render:)``). When `prompt` ends with that
    /// tail, the index is the start of the tail. Else, and when the last
    /// message is not a user message, it renders the full probe.
    ///
    /// - Parameters:
    ///   - prompt: the prepared prompt of `messages`.
    ///   - messages: the chat messages `prompt` renders from.
    ///   - render: renders messages the way `prompt` was rendered, with the
    ///     same template variables and tools.
    /// - Returns: the index, or nil when `render` throws. A template can refuse
    ///   two user messages after each other, and then the pass takes no
    ///   checkpoint at the end of the last message.
    package static func lastMessageEnd(
        of prompt: LMInput, messages: [Chat.Message],
        render: ([Chat.Message]) async throws -> LMInput
    ) async -> Int? {
        let promptTokens = prompt.text.tokens.asArray(Int.self)
        if let tail = await generationPromptTail(after: messages, render: render),
            promptTokens.suffix(tail.count).elementsEqual(tail)
        {
            return promptTokens.count - tail.count
        }
        return await probedLastMessageEnd(of: promptTokens, messages: messages, render: render)
    }

    /// The index of `promptTokens` where its leading system messages end: the
    /// prefix it shares with a render of those messages plus one user message
    /// with the text ``boundaryProbeText``.
    ///
    /// A checkpoint there serves every conversation that starts with the same
    /// instructions and tools, such as a fork with a new history.
    ///
    /// - Parameters:
    ///   - promptTokens: the tokens of the prepared prompt of `messages`.
    ///   - messages: the chat messages the prompt renders from.
    ///   - render: renders messages the way the prompt was rendered.
    /// - Returns: the index, or nil when `messages` does not start with a
    ///   system message followed by another message, or when `render` throws.
    package static func leadingSystemEnd(
        of promptTokens: [Int], messages: [Chat.Message],
        render: ([Chat.Message]) async throws -> LMInput
    ) async -> Int? {
        let system = messages.prefix { $0.role == .system }
        guard !system.isEmpty, system.count < messages.count else { return nil }
        return await probedLastMessageEnd(
            of: promptTokens, messages: Array(system), render: render)
    }

    /// The index of `promptTokens` where its last message ends, from a render
    /// of `messages` plus one user message with the text ``boundaryProbeText``:
    /// the prefix that render and `promptTokens` share.
    ///
    /// - Parameters:
    ///   - promptTokens: the tokens of the prepared prompt of `messages`.
    ///   - messages: the chat messages the prompt renders from.
    ///   - render: renders messages the way the prompt was rendered.
    /// - Returns: the index, or nil when `render` throws.
    private static func probedLastMessageEnd(
        of promptTokens: [Int], messages: [Chat.Message],
        render: ([Chat.Message]) async throws -> LMInput
    ) async -> Int? {
        guard let probe = try? await render(messages + [.user(boundaryProbeText)]) else {
            return nil
        }
        return commonPrefixLength(of: promptTokens, and: probe.text.tokens.asArray(Int.self))
    }

    /// The tokens that a render writes after the end of the last message when
    /// that message is a user message: the generation prompt, from the point
    /// where one more user message parts from it.
    ///
    /// The function renders a small sample conversation: the leading system
    /// messages of `messages` and one user message with the text
    /// ``boundaryProbeText``. The full probe of that sample gives the end of
    /// its last message, and the tokens after that end are the tail. The
    /// sample is small, thus its two renders cost little.
    ///
    /// The tail serves only a conversation whose last message is a user
    /// message. One more user message does not change how a template such as
    /// Qwen 3.8 renders the messages before it then. After a tool message, the
    /// Qwen 3.8 template can render the reasoning of a past assistant turn in
    /// a different way, thus the caller must render the full probe.
    ///
    /// - Parameters:
    ///   - messages: the chat messages of the prompt.
    ///   - render: renders messages the way the prompt was rendered.
    /// - Returns: the tail, or nil when the last message is not a user message
    ///   or when `render` throws.
    private static func generationPromptTail(
        after messages: [Chat.Message], render: ([Chat.Message]) async throws -> LMInput
    ) async -> [Int]? {
        guard messages.last?.role == .user else { return nil }
        let sample = messages.prefix { $0.role == .system } + [.user(boundaryProbeText)]
        guard let sampleTokens = try? await render(Array(sample)).text.tokens.asArray(Int.self),
            let end = await probedLastMessageEnd(
                of: sampleTokens, messages: Array(sample), render: render)
        else {
            return nil
        }
        return Array(sampleTokens[end...])
    }
}

/// The checkpoints a session keeps of its caches, and the choice of the one a
/// new prompt goes back to.
package struct PromptCacheCheckpoints {

    /// The checkpoints of the last pass, in the order the session took them.
    private var checkpoints: [PromptCacheCheckpoint] = []

    /// Whether the last pass answered a tool message.
    private var lastPassAnsweredTool = false

    /// The checkpoints of the last pass that answered a user message, kept
    /// through the tool steps that follow it.
    ///
    /// A template such as Qwen3 drops the reasoning of every reply after the
    /// last user message when it renders the next user turn. That render thus
    /// parts from the ledger inside the first reply of a tool loop, and only a
    /// checkpoint of the pass before that reply can serve it.
    private var turn: [PromptCacheCheckpoint] = []

    /// The checkpoint at the end of the leading system messages, which outlives
    /// the other checkpoints while each prompt starts with it.
    private(set) var leading: PromptCacheCheckpoint?

    /// Creates an empty set.
    package init() {}

    /// Keeps `checkpoint` as the checkpoint at the end of the leading system
    /// messages.
    package mutating func setLeading(_ checkpoint: PromptCacheCheckpoint?) {
        leading = checkpoint
    }

    /// Forgets every checkpoint but the leading one, and that one too when
    /// `promptTokens` does not start with it.
    package mutating func removeAllButLeading(servingPrompt promptTokens: [Int]) {
        beginPass(servingPrompt: promptTokens, answeringTool: false)
    }

    /// Forgets the checkpoints a new pass replaces.
    ///
    /// A pass that answers a tool message keeps the checkpoints of the last
    /// pass that answered a user message, while `promptTokens` starts with
    /// them, and drops those of the tool step before it: the next step
    /// rewrites the call that step made. Any other pass keeps only the leading
    /// checkpoint. The leading checkpoint is dropped too when `promptTokens`
    /// does not start with it.
    ///
    /// - Parameters:
    ///   - promptTokens: the whole rendered prompt of the new pass.
    ///   - answeringTool: whether the new pass answers a tool message.
    package mutating func beginPass(servingPrompt promptTokens: [Int], answeringTool: Bool) {
        if answeringTool {
            if !lastPassAnsweredTool {
                turn = checkpoints
            }
            turn.removeAll { !$0.serves(promptTokens) }
        } else {
            turn.removeAll()
        }
        checkpoints.removeAll()
        lastPassAnsweredTool = answeringTool
        if let leading, !leading.serves(promptTokens) {
            self.leading = nil
        }
    }

    /// Adds `checkpoint` to the set.
    ///
    /// - Parameter checkpoint: the checkpoint to keep, or nil when the session
    ///   could not take one, which adds nothing.
    package mutating func append(_ checkpoint: PromptCacheCheckpoint?) {
        if let checkpoint {
            checkpoints.append(checkpoint)
        }
    }

    /// Every checkpoint: those of the last user turn that a tool loop keeps,
    /// those of the last pass in the order the session took them, then the
    /// leading one.
    package var all: [PromptCacheCheckpoint] {
        turn + checkpoints + [leading].compactMap { $0 }
    }

    /// Forgets every checkpoint.
    package mutating func removeAll() {
        checkpoints.removeAll()
        turn.removeAll()
        lastPassAnsweredTool = false
        leading = nil
    }

    /// The longest checkpoint that `promptTokens` starts with, thus the one
    /// that leaves the fewest tokens to feed.
    ///
    /// - Parameter promptTokens: the whole rendered prompt.
    /// - Returns: the checkpoint, or nil when none serves the prompt.
    package func longest(serving promptTokens: [Int]) -> PromptCacheCheckpoint? {
        all
            .filter { $0.serves(promptTokens) }
            .max { $0.tokens.count < $1.tokens.count }
    }
}
