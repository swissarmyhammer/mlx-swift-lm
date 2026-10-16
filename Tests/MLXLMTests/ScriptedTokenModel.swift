// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon

/// A weight-free model that writes one scripted token for each forward pass and
/// records what each pass received.
///
/// A ``ChatSession`` test drives it to make the model write exact tokens: the
/// prefill of a turn writes the first token of the turn, and each decode step
/// writes the next one.
///
/// The model has one attention layer. It can also have one recurrent layer,
/// whose cache cannot rewind, as the caches of a hybrid model (Qwen 3.5). A
/// session then keeps checkpoints of the caches, and each feed to a checkpoint
/// is one more forward pass, which uses one token of the script.
///
/// The model is not `Sendable`, because ``passes`` and `step` are mutable and
/// carry no lock. One ``ChatSession`` owns this model, and a session runs its
/// forward passes ONE AT A TIME, thus every mutation happens inside
/// ``callAsFunction(_:cache:)`` and no two calls overlap. A test reads
/// ``passes`` only after its `respond(to:)` calls return.
final class ScriptedTokenModel: Module, LLMModel, KVCacheDimensionProvider {
    /// What one forward pass received.
    struct Pass {
        /// The position the attention cache held when the pass began.
        var offset: Int
        /// The tokens the pass fed.
        var tokens: [Int]
        /// The number of tokens the pass fed.
        var tokenCount: Int { tokens.count }
    }

    /// The logit of the scripted token. Every other token gets its negative.
    private static let scriptedLogit: Float = 30

    /// One attention layer with one head, thus one plain KV cache.
    let kvHeads: [Int] = [1]

    /// The size of the scripted vocabulary, which is the width of each logit row.
    let vocabularySize: Int

    /// The response protocol the model declares.
    let toolCallFormat: ToolCallFormat?

    /// Whether the model has a recurrent layer after its attention layer.
    let hasRecurrentLayer: Bool

    /// Every forward pass, in order.
    private(set) var passes: [Pass] = []

    /// One token for each forward pass, in order.
    private let script: [Int]

    /// The token of each pass after the script ends.
    private let finalToken: Int

    /// The index in ``script`` of the token of the next pass.
    private var step = 0

    /// - Parameters:
    ///   - script: one token for each forward pass, in order.
    ///   - finalToken: the token of each pass after the script ends.
    ///   - vocabularySize: the size of the scripted vocabulary.
    ///   - toolCallFormat: the response protocol the model declares.
    ///   - hasRecurrentLayer: whether the model has a recurrent layer, whose
    ///     cache cannot rewind.
    init(
        script: [Int], finalToken: Int, vocabularySize: Int, toolCallFormat: ToolCallFormat?,
        hasRecurrentLayer: Bool = false
    ) {
        self.script = script
        self.finalToken = finalToken
        self.vocabularySize = vocabularySize
        self.toolCallFormat = toolCallFormat
        self.hasRecurrentLayer = hasRecurrentLayer
        super.init()
    }

    /// One cache for the attention layer, and one recurrent cache when the
    /// model has a recurrent layer.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let attention = try makeAttentionKVCache(parameters: parameters)
        return hasRecurrentLayer ? [attention, MambaCache()] : [attention]
    }

    /// Writes the next scripted token as the logits of the last position.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenCount = inputs.dim(-1)
        passes.append(
            Pass(offset: cache?.first?.offset ?? 0, tokens: inputs.reshaped(-1).asArray(Int.self)))

        let entry = MLXArray.zeros([1, 1, tokenCount, 1])
        _ = cache?.first?.update(keys: entry, values: entry)
        for recurrent in cache?.compactMap({ $0 as? MambaCache }) ?? [] {
            recurrent.advance(tokenCount)
        }

        let token = step < script.count ? script[step] : finalToken
        step += 1
        var row = [Float](repeating: -Self.scriptedLogit, count: vocabularySize)
        row[token] = Self.scriptedLogit
        let logits = MLXArray(row, [1, 1, vocabularySize])
        if tokenCount == 1 {
            return logits
        }
        return concatenated(Array(repeating: logits, count: tokenCount), axis: 1)
    }

    /// The model has no adapter layers.
    var loraLayers: [Module] { [] }
}
