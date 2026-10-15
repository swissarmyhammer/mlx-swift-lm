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
/// The model is not `Sendable`, because ``passes`` and `step` are mutable and
/// carry no lock. One ``ChatSession`` owns this model, and a session runs its
/// forward passes ONE AT A TIME, thus every mutation happens inside
/// ``callAsFunction(_:cache:)`` and no two calls overlap. A test reads
/// ``passes`` only after its `respond(to:)` calls return.
final class ScriptedTokenModel: Module, LLMModel, KVCacheDimensionProvider {
    /// What one forward pass received.
    struct Pass {
        /// The position the cache held when the pass began.
        var offset: Int
        /// The number of tokens the pass fed.
        var tokenCount: Int
    }

    /// The logit of the scripted token. Every other token gets its negative.
    private static let scriptedLogit: Float = 30

    /// One attention layer with one head, thus one plain KV cache.
    let kvHeads: [Int] = [1]

    /// The size of the scripted vocabulary, which is the width of each logit row.
    let vocabularySize: Int

    /// The response protocol the model declares.
    let toolCallFormat: ToolCallFormat?

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
    init(script: [Int], finalToken: Int, vocabularySize: Int, toolCallFormat: ToolCallFormat?) {
        self.script = script
        self.finalToken = finalToken
        self.vocabularySize = vocabularySize
        self.toolCallFormat = toolCallFormat
        super.init()
    }

    /// Writes the next scripted token as the logits of the last position.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenCount = inputs.dim(-1)
        passes.append(Pass(offset: cache?.first?.offset ?? 0, tokenCount: tokenCount))

        let entry = MLXArray.zeros([1, 1, tokenCount, 1])
        _ = cache?.first?.update(keys: entry, values: entry)

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
