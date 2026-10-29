// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// The recurrent cache of a Qwen 3.5 linear layer reports its position.
///
/// A prompt cache compares the position of every layer's cache with its token
/// ledger. The attention layers count the tokens they take. The linear layers
/// must count them too, or the ledger and the caches never agree, and a hybrid
/// model starts every round cold. Card `^xx5g893` measured that on
/// `mlx-community/Qwen3.8-27B-mxfp4`.
@Suite(.serialized)
struct Qwen35RecurrentCachePositionTests {

    /// The number of tokens the prompt of each test holds.
    private static let promptTokenCount = 4

    /// The number of tokens fed after the prompt, one decode step.
    private static let decodedTokenCount = 1

    /// The tokens of the verify whose snapshot slots a rollback selects.
    private static let verifyTokenCount = 2

    /// The tokens of the verify the rollback rejects.
    private static let rejectedTokenCount = 1

    /// The number of layers of the small hybrid model.
    private static let hybridLayerCount = 2

    /// One full-attention layer for each two layers, thus layer 0 is linear
    /// and layer 1 is attention.
    private static let hybridFullAttentionInterval = 2

    /// The text configuration of the small models of this suite.
    private func textConfiguration() throws -> MLXLLM.Qwen35TextConfiguration {
        try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
    }

    /// The text configuration of a hybrid model: ``hybridLayerCount`` layers,
    /// with one full-attention layer for each ``hybridFullAttentionInterval``
    /// layers.
    ///
    /// The decoder resolves the layer schedule from these two values, thus the
    /// JSON gives them. A change to the values after the decode does not change
    /// the schedule.
    private func hybridTextConfiguration() throws -> MLXLLM.Qwen35TextConfiguration {
        try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 1, hiddenLayers: Self.hybridLayerCount,
                    fullAttentionInterval: Self.hybridFullAttentionInterval
                ).utf8))
    }

    /// The vision-language text configuration of the small models of this
    /// suite.
    private func visionTextConfiguration() throws
        -> MLXVLM.Qwen35Configuration.TextConfiguration
    {
        try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
    }

    /// A random hidden-state input of `tokenCount` tokens.
    private func hiddenStates(tokenCount: Int, width: Int) -> MLXArray {
        MLXRandom.normal([1, tokenCount, width])
    }

    @Test("the text linear layer advances its cache by the tokens it takes")
    func theTextLinearLayerAdvancesItsCacheByTheTokensItTakes() throws {
        let configuration = try textConfiguration()
        let layer = MLXLLM.Qwen35GatedDeltaNet(configuration)
        let cache = MambaCache()
        let prompt = hiddenStates(
            tokenCount: Self.promptTokenCount, width: configuration.hiddenSize)

        eval(layer(prompt, cache: cache))
        #expect(cache.offset == Self.promptTokenCount)

        let next = hiddenStates(
            tokenCount: Self.decodedTokenCount, width: configuration.hiddenSize)
        eval(layer(next, cache: cache))
        #expect(cache.offset == Self.promptTokenCount + Self.decodedTokenCount)
    }

    @Test("the vision-language linear layer advances its cache by the tokens it takes")
    func theVisionLanguageLinearLayerAdvancesItsCacheByTheTokensItTakes() throws {
        let configuration = try visionTextConfiguration()
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(configuration)
        let cache = MambaCache()
        let prompt = hiddenStates(
            tokenCount: Self.promptTokenCount, width: configuration.hiddenSize)

        eval(layer(prompt, cache: cache))
        #expect(cache.offset == Self.promptTokenCount)

        let next = hiddenStates(
            tokenCount: Self.decodedTokenCount, width: configuration.hiddenSize)
        eval(layer(next, cache: cache))
        #expect(cache.offset == Self.promptTokenCount + Self.decodedTokenCount)
    }

    @Test("a speculative snapshot slot restores the position of its token")
    func aSpeculativeSnapshotSlotRestoresThePositionOfItsToken() throws {
        let configuration = try textConfiguration()
        let layer = MLXLLM.Qwen35GatedDeltaNet(configuration)
        let cache = MambaCache()
        let input = hiddenStates(tokenCount: Self.verifyTokenCount, width: configuration.hiddenSize)

        eval(layer(input, cache: cache, keepsSnapshots: true))
        #expect(cache.offset == Self.verifyTokenCount)

        #expect(cache.selectSpeculativeSlot(rollingBack: Self.rejectedTokenCount))
        #expect(cache.offset == Self.verifyTokenCount - Self.rejectedTokenCount)
    }

    @Test("the vision-language speculative snapshot slot restores the position of its token")
    func theVisionLanguageSpeculativeSnapshotSlotRestoresThePositionOfItsToken() throws {
        let configuration = try visionTextConfiguration()
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(configuration)
        let cache = MambaCache()
        let input = hiddenStates(tokenCount: Self.verifyTokenCount, width: configuration.hiddenSize)

        eval(layer(input, cache: cache, keepsSnapshots: true))
        #expect(cache.offset == Self.verifyTokenCount)

        #expect(cache.selectSpeculativeSlot(rollingBack: Self.rejectedTokenCount))
        #expect(cache.offset == Self.verifyTokenCount - Self.rejectedTokenCount)
    }

    @Test("the compiled decode of a linear layer advances its cache by one")
    func theCompiledDecodeOfALinearLayerAdvancesItsCacheByOne() throws {
        let configuration = try hybridTextConfiguration()
        let layer = MLXLLM.Qwen35DecoderLayer(configuration, layerIdx: 0)
        #expect(layer.isLinear, "the premise: layer 0 of a two-layer interval is linear")
        let cache = MambaCache()
        let token = hiddenStates(
            tokenCount: Self.decodedTokenCount, width: configuration.hiddenSize)

        eval(layer(token, attentionMask: .none, ssmMask: nil, cache: cache))

        #expect(cache.offset == Self.decodedTokenCount)
    }

    @Test("every cache of a hybrid text model holds the position after prefill and decode")
    func everyCacheOfAHybridTextModelHoldsThePositionAfterPrefillAndDecode() throws {
        let configuration = try hybridTextConfiguration()
        let model = MLXLLM.Qwen35TextModel(configuration)
        let caches = try model.newCache(parameters: nil)
        #expect(
            caches.contains { $0 is MambaCache } && caches.contains { $0 is KVCacheSimple },
            "the premise: the model holds a recurrent cache and an attention cache")
        let prompt = MLXArray(Array(1 ... Int32(Self.promptTokenCount))).reshaped(
            [1, Self.promptTokenCount])

        eval(model(LMInput.Text(tokens: prompt), cache: caches, state: nil).logits)
        #expect(caches.allSatisfy { $0.offset == Self.promptTokenCount })

        let next = MLXArray([Int32(Self.promptTokenCount + 1)]).reshaped(
            [1, Self.decodedTokenCount])
        eval(model(LMInput.Text(tokens: next), cache: caches, state: nil).logits)
        #expect(
            caches.allSatisfy { $0.offset == Self.promptTokenCount + Self.decodedTokenCount })
    }
}
