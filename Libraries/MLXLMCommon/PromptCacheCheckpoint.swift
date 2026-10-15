// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The caches of a hybrid model at one position, kept so that a later turn can
/// go back to that position.
///
/// An attention cache rewinds to any earlier position, but a recurrent cache
/// (the linear layers of Qwen 3.5) holds one state that sums every token it
/// saw, and it cannot rewind. The checkpoint thus keeps a copy of each cache
/// that cannot rewind, and nothing for a cache that can: a restore trims that
/// cache back to the checkpoint.
///
/// A recurrent layer puts new state arrays in its slots and does not write into
/// the old ones, thus a copy holds the state of the checkpoint while the live
/// caches move on.
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

    /// For each layer, a copy of a cache that cannot rewind, or nil for a cache
    /// that a trim takes back to the checkpoint.
    private let copies: [KVCache?]

    /// The model state the prefill left at the checkpoint, or nil.
    package let state: LMOutput.State?

    /// The bytes the copies and `state` hold in memory.
    package let byteCount: Int

    /// Takes a checkpoint of `caches`, which represent `tokens`.
    ///
    /// - Parameters:
    ///   - caches: the live caches, one for each layer of the model.
    ///   - tokens: the tokens `caches` represents.
    ///   - state: the model state the prefill left at this position, or nil.
    /// - Returns: nil when every cache can rewind, because then a rewind serves
    ///   every later render and a checkpoint adds nothing, or when a cache does
    ///   not stand at the end of `tokens`.
    package init?(caches: [KVCache], tokens: [Int], state: LMOutput.State?) {
        guard Self.applies(to: caches), caches.allSatisfy({ $0.offset == tokens.count }) else {
            return nil
        }
        let copies = caches.map { $0.isTrimmable ? nil : $0.copy() }
        self.tokens = tokens
        self.copies = copies
        self.state = state
        self.byteCount =
            copies.reduce(0) { $0 + ($1?.residentByteCount ?? 0) }
            + (state?.residentByteCount ?? 0)
    }

    /// Whether a model has a layer that cannot rewind, thus whether a
    /// checkpoint can serve it.
    ///
    /// - Parameter model: the model whose fresh caches are read.
    /// - Returns: `true` for a hybrid model.
    package static func applies(to model: any LanguageModel) -> Bool {
        guard let caches = try? model.newCache(parameters: nil) else { return false }
        return applies(to: caches)
    }

    /// Whether `caches` holds a cache that cannot rewind, thus whether a
    /// checkpoint can serve them.
    ///
    /// - Parameter caches: one cache for each layer of the model.
    /// - Returns: `true` for the caches of a hybrid model.
    package static func applies(to caches: [KVCache]) -> Bool {
        caches.contains { !$0.isTrimmable }
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
        for (cache, copy) in zip(caches, copies) {
            if let copy {
                restored.append(copy.copy())
            } else if rewindPromptCache([cache], to: tokens.count) {
                restored.append(cache)
            } else {
                return nil
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
    ///   - prefill: cuts the tokens into forward passes.
    /// - Returns: the model state the last pass left.
    /// - Throws: `CancellationError` when the task is cancelled between two
    ///   forward passes.
    package static func prefill(
        _ tokens: [Int], into caches: [KVCache], state: LMOutput.State?,
        model: any LanguageModel, prefill: PrefillParameters
    ) throws -> LMOutput.State? {
        var state = state
        let forward = { (range: Range<Int>) in
            let chunk = MLXArray(Array(tokens[range])).expandedDimensions(axis: 0)
            state = model(LMInput.Text(tokens: chunk), cache: caches, state: state).state
            asyncEval(caches.flatMap(\.state))
        }
        let chunked = try prefill.forEachChunk(total: tokens.count, reserving: 0, forward)
        if chunked < tokens.count {
            forward(chunked ..< tokens.count)
        }
        eval(caches.flatMap(\.state))
        return state
    }

    /// Feeds the prompt from `start` up to `boundary` into `caches`, and takes
    /// a checkpoint there.
    ///
    /// - Parameters:
    ///   - boundary: the index of the prompt where the checkpoint goes.
    ///   - promptTokens: the whole rendered prompt of the pass.
    ///   - representedTokens: the tokens the caches represent once the pass has
    ///     fed the whole prompt. A protocol rule can keep generated tokens here
    ///     that the render does not hold.
    ///   - start: the leading prompt tokens the caches already hold.
    ///   - caches: the live caches, one for each layer of the model.
    ///   - state: the model state to seed the first forward pass with, or nil.
    ///   - model: the model that feeds the caches.
    ///   - prefill: cuts the tokens into forward passes.
    /// - Returns: the model state the fed tokens left and the checkpoint at
    ///   `boundary`, which is nil when a cache did not stand there. Nil when
    ///   `boundary` is not after `start` or not before the end of the prompt:
    ///   the caches hold that position already, or only the prefill of the
    ///   generation reaches it.
    /// - Throws: `CancellationError` when the task is cancelled between two
    ///   forward passes.
    package static func prefillToCheckpoint(
        at boundary: Int, of promptTokens: [Int], representing representedTokens: [Int],
        from start: Int, into caches: [KVCache], state: LMOutput.State?,
        model: any LanguageModel, prefill: PrefillParameters
    ) throws -> (state: LMOutput.State?, checkpoint: PromptCacheCheckpoint?)? {
        guard boundary > start, boundary < promptTokens.count else { return nil }
        let fedState = try Self.prefill(
            Array(promptTokens[start ..< boundary]), into: caches, state: state, model: model,
            prefill: prefill)
        let checkpoint = PromptCacheCheckpoint(
            caches: caches,
            tokens: Array(representedTokens.dropLast(promptTokens.count - boundary)),
            state: fedState)
        return (fedState, checkpoint)
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
    /// This function renders `messages` plus one user message with the text
    /// ``boundaryProbeText``, and takes the prefix that render and `prompt`
    /// share. With no generation prompt, the index is the end of `prompt`.
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
        guard let probe = try? await render(messages + [.user(boundaryProbeText)]) else {
            return nil
        }
        return commonPrefixLength(
            of: prompt.text.tokens.asArray(Int.self), and: probe.text.tokens.asArray(Int.self))
    }
}

/// The checkpoints a session keeps of its caches, and the choice of the one a
/// new prompt goes back to.
package struct PromptCacheCheckpoints {

    /// The checkpoints, in the order the session took them.
    private var checkpoints: [PromptCacheCheckpoint] = []

    /// Creates an empty set.
    package init() {}

    /// Adds `checkpoint` to the set.
    ///
    /// - Parameter checkpoint: the checkpoint to keep, or nil when the session
    ///   could not take one, which adds nothing.
    package mutating func append(_ checkpoint: PromptCacheCheckpoint?) {
        if let checkpoint {
            checkpoints.append(checkpoint)
        }
    }

    /// Forgets every checkpoint.
    package mutating func removeAll() {
        checkpoints.removeAll()
    }

    /// The longest checkpoint that `promptTokens` starts with, thus the one
    /// that leaves the fewest tokens to feed.
    ///
    /// - Parameter promptTokens: the whole rendered prompt.
    /// - Returns: the checkpoint, or nil when none serves the prompt.
    package func longest(serving promptTokens: [Int]) -> PromptCacheCheckpoint? {
        checkpoints.filter { $0.serves(promptTokens) }.max { $0.tokens.count < $1.tokens.count }
    }
}
