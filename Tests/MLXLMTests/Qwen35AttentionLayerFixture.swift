// Copyright © 2026 Apple Inc.
//
// One small Qwen3.5 full-attention layer and a filled key cache, for the tests
// that check which attention kernel the layer reads its cache through.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// One small dense Qwen3.5 layer with full attention: six query heads for
/// one KV head, as in Qwen3.8, and a head width that each test picks.
struct Qwen35AttentionLayerFixture {

    /// The seed of each random operand of one test.
    struct Seeds: Sendable {
        /// The seed of the filled cache.
        let cache: UInt64
        /// The seed of the new tokens of the step.
        let tokens: UInt64
    }

    /// The number of query heads; six for each KV head, as in Qwen3.8.
    static let queryHeads = 6

    /// The number of KV heads.
    static let kvHeads = 1

    /// The width of the residual stream.
    static let hiddenSize = 32

    /// The index of the full-attention layer: with a full-attention interval
    /// of 2, layer 1 is the first full-attention layer.
    private static let attentionLayer = 1

    /// The number of random tensors of a filled cache: the keys and the
    /// values.
    private static let keysAndValues = 2

    /// The index of each random tensor of a traced step, along the leading
    /// axis of the one random draw that holds them all.
    private enum StepTensor: Int, CaseIterable {
        /// The queries of the new tokens.
        case queries
        /// The keys of the new tokens.
        case keys
        /// The values of the new tokens.
        case values
    }

    /// The head shape of the gated delta net (GDN) layers of a configuration.
    struct LinearHeads: Sendable {
        /// The number of key heads.
        let keyHeads: Int
        /// The number of value heads.
        let valueHeads: Int
        /// The width of one key head.
        let keyHeadDim: Int
        /// The width of one value head.
        let valueHeadDim: Int
    }

    /// The decoder layers: one GDN layer, then one full-attention layer.
    static let hiddenLayers = 2

    /// The width of the dense MLPs.
    static let intermediateSize = 64

    /// The GDN heads: one key head and two value heads, each of width 8.
    static let linearHeads = LinearHeads(
        keyHeads: 1, valueHeads: 2, keyHeadDim: 8, valueHeadDim: 8)

    /// The conv taps of the GDN layers.
    static let convKernel = 4

    /// The number of tokens of the vocabulary.
    static let vocabularySize = 32

    /// The layer interval of the full-attention layers.
    static let fullAttentionInterval = 2

    /// The width of one attention head.
    let headDim: Int

    /// A small dense Qwen3.5 text configuration with one full-attention
    /// layer.
    func configuration() throws -> Qwen35TextConfiguration {
        try Self.configuration(Qwen35TextConfiguration.self, headDim: headDim)
    }

    /// A small Qwen 3.5 configuration, decoded as `type`. Each parameter but
    /// the head width defaults to the shape of ``configuration()``.
    ///
    /// - Parameters:
    ///   - type: The configuration type that decodes the Qwen 3.5 keys.
    ///   - headDim: The width of one full-attention head.
    ///   - hiddenSize: The width of the residual stream.
    ///   - hiddenLayers: The number of decoder layers.
    ///   - intermediateSize: The width of the dense MLPs.
    ///   - queryHeads: The number of query heads of a full-attention layer.
    ///   - kvHeads: The number of KV heads of a full-attention layer.
    ///   - linearHeads: The head shape of the GDN layers.
    ///   - convKernel: The conv taps of the GDN layers.
    ///   - vocabularySize: The number of tokens of the vocabulary.
    ///   - fullAttentionInterval: The layer interval of the full-attention
    ///     layers.
    ///   - extraKeys: The keys that a test adds to the configuration, or whose
    ///     value it replaces.
    /// - Returns: The configuration.
    /// - Throws: The error of the serializer or of the decoder.
    static func configuration<Configuration: Decodable>(
        _ type: Configuration.Type,
        headDim: Int,
        hiddenSize: Int = Qwen35AttentionLayerFixture.hiddenSize,
        hiddenLayers: Int = Qwen35AttentionLayerFixture.hiddenLayers,
        intermediateSize: Int = Qwen35AttentionLayerFixture.intermediateSize,
        queryHeads: Int = Qwen35AttentionLayerFixture.queryHeads,
        kvHeads: Int = Qwen35AttentionLayerFixture.kvHeads,
        linearHeads: LinearHeads = Qwen35AttentionLayerFixture.linearHeads,
        convKernel: Int = Qwen35AttentionLayerFixture.convKernel,
        vocabularySize: Int = Qwen35AttentionLayerFixture.vocabularySize,
        fullAttentionInterval: Int = Qwen35AttentionLayerFixture.fullAttentionInterval,
        extraKeys: [String: Any] = [:]
    ) throws -> Configuration {
        let keys: [String: Any] = [
            "model_type": "qwen3_5",
            "hidden_size": hiddenSize,
            "num_hidden_layers": hiddenLayers,
            "intermediate_size": intermediateSize,
            "num_attention_heads": queryHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "linear_num_value_heads": linearHeads.valueHeads,
            "linear_num_key_heads": linearHeads.keyHeads,
            "linear_key_head_dim": linearHeads.keyHeadDim,
            "linear_value_head_dim": linearHeads.valueHeadDim,
            "linear_conv_kernel_dim": convKernel,
            "vocab_size": vocabularySize,
            "full_attention_interval": fullAttentionInterval,
        ]
        let json = try JSONSerialization.data(
            withJSONObject: keys.merging(extraKeys) { _, extra in extra })
        return try JSONDecoder().decode(type, from: json)
    }

    /// The full-attention layer of the configuration, with its weights in
    /// one dtype.
    ///
    /// - Parameter dtype: The dtype of the weights.
    /// - Returns: The layer.
    func layer(dtype: DType) throws -> Qwen35DecoderLayer {
        let layer = Qwen35DecoderLayer(try configuration(), layerIdx: Self.attentionLayer)
        layer.update(parameters: layer.parameters().mapValues { $0.asType(dtype) })
        return layer
    }

    /// A key cache that holds `keys` random keys and values.
    ///
    /// - Parameters:
    ///   - keys: The number of cached keys.
    ///   - dtype: The dtype of the cached keys and values.
    ///   - seed: The seed of the random keys and values.
    /// - Returns: The filled cache.
    func filledCache(keys: Int, dtype: DType, seed: UInt64) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let rows = MLXRandom.normal(
            [Self.keysAndValues, 1, Self.kvHeads, keys, headDim], key: MLXRandom.key(seed)
        ).asType(dtype)
        _ = cache.update(keys: rows[0], values: rows[1])
        eval(cache.state)
        return cache
    }

    /// The queries of a step at the rotary position of the cache. Call it
    /// before the step writes its keys, while the cache offset is the
    /// position of the first new token.
    ///
    /// - Parameters:
    ///   - queries: The queries before the rotary position.
    ///   - attention: The attention of the layer.
    ///   - cache: The cache before the step.
    /// - Returns: The rotated queries.
    static func rotated(
        _ queries: MLXArray, attention: Qwen35Attention, cache: KVCacheSimple
    ) -> MLXArray {
        applyRotaryPosition(attention.rope, to: queries, offset: cache.ropeOffset)
    }

    /// The mask of one kind for a step: the new tokens over the cached keys
    /// and the new keys, in the dtype of the rotated queries.
    ///
    /// - Parameters:
    ///   - kind: The mask kind.
    ///   - rotatedQueries: The queries of the step, from ``rotated(_:attention:cache:)``.
    ///   - keys: The number of keys the cache holds after the step.
    /// - Returns: The mask.
    static func mask(
        _ kind: AttentionMaskKind, rotatedQueries: MLXArray, keys: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kind.mode(
            queries: rotatedQueries.dim(AttentionKernelOperands.rowAxis), keys: keys,
            dtype: rotatedQueries.dtype)
    }

    /// The output that the attention route must give over the cache after
    /// the step wrote its keys: the output of the kernel of the mask kind,
    /// or the SDPA of MLX when `kernels` holds no kernel for the mask kind.
    ///
    /// - Parameters:
    ///   - kind: The mask kind of the step.
    ///   - kernels: The kernel that the route picks for each mask kind.
    ///   - rotatedQueries: The queries of the step, from ``rotated(_:attention:cache:)``.
    ///   - attention: The attention of the layer.
    ///   - cache: The cache after the step.
    /// - Returns: The expected output.
    static func routeOutput(
        with kind: AttentionMaskKind, picksFrom kernels: AttentionKernelTable,
        rotatedQueries: MLXArray, attention: Qwen35Attention, cache: KVCacheSimple
    ) -> MLXArray {
        let state = cache.state
        return AttentionKernelOperands(
            queries: rotatedQueries, keys: state[0], values: state[1], scale: attention.scale
        ).expectedRouteOutput(with: kind, picksFrom: kernels)
    }

    /// Checks that the general attention body of the layer gives the output
    /// of the route bit for bit, for `tokens` random tokens after
    /// `cachedKeys` cached keys: the kernel of the mask kind, or the SDPA of
    /// MLX when `kernels` holds no kernel for the mask kind.
    ///
    /// - Parameters:
    ///   - cachedKeys: The number of cached keys ahead of the step.
    ///   - tokens: The number of new tokens of the step.
    ///   - dtype: The dtype of the weights, the tokens and the cache.
    ///   - kind: The mask kind the step gives the body.
    ///   - kernels: The kernel that the body must pick for each mask kind.
    ///   - seeds: The seed of each random operand.
    ///   - sourceLocation: The location that a failure reports.
    func expectTheAttentionBodyGivesTheRouteOutput(
        cachedKeys: Int, tokens: Int, dtype: DType, mask kind: AttentionMaskKind,
        picksFrom kernels: AttentionKernelTable, seeds: Seeds,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let attention = try #require(
            try layer(dtype: dtype).selfAttn, sourceLocation: sourceLocation)
        let cache = filledCache(keys: cachedKeys, dtype: dtype, seed: seeds.cache)
        let x = MLXRandom.normal([1, tokens, Self.hiddenSize], key: MLXRandom.key(seeds.tokens))
            .asType(dtype)
        let (queries, gate, _, _) = attention.projectPreRope(x)
        let rotatedQueries = Self.rotated(queries, attention: attention, cache: cache)
        let keys = cachedKeys + tokens

        let output = attention(
            x, mask: Self.mask(kind, rotatedQueries: rotatedQueries, keys: keys), cache: cache)
        let expected = attention.mergeHeadsAndProject(
            attention: Self.routeOutput(
                with: kind, picksFrom: kernels, rotatedQueries: rotatedQueries,
                attention: attention, cache: cache),
            gate: gate)

        Self.expectTheStep(
            output, gives: expected, cache: cache, keys: keys, sourceLocation: sourceLocation)
    }

    /// Checks that the traced attention step of the layer
    /// (`attentionCacheStep`) gives the output of the route bit for bit, for
    /// `tokens` random query, key and value rows after `cachedKeys` cached
    /// keys: the kernel of the mask kind, or the SDPA of MLX when `kernels`
    /// holds no kernel for the mask kind.
    ///
    /// - Parameters:
    ///   - cachedKeys: The number of cached keys ahead of the step.
    ///   - tokens: The number of new tokens of the step.
    ///   - dtype: The dtype of the weights, the rows and the cache.
    ///   - kind: The mask kind the step gives the layer.
    ///   - kernels: The kernel that the step must pick for each mask kind.
    ///   - seeds: The seed of each random operand.
    ///   - sourceLocation: The location that a failure reports.
    func expectTheTracedStepGivesTheRouteOutput(
        cachedKeys: Int, tokens: Int, dtype: DType, mask kind: AttentionMaskKind,
        picksFrom kernels: AttentionKernelTable, seeds: Seeds,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let layer = try layer(dtype: dtype)
        let attention = try #require(layer.selfAttn, sourceLocation: sourceLocation)
        let cache = filledCache(keys: cachedKeys, dtype: dtype, seed: seeds.cache)
        let step = MLXRandom.normal(
            [StepTensor.allCases.count, 1, Self.queryHeads, tokens, headDim],
            key: MLXRandom.key(seeds.tokens)
        ).asType(dtype)
        let queries = step[StepTensor.queries.rawValue]
        let rotatedQueries = Self.rotated(queries, attention: attention, cache: cache)
        let keys = cachedKeys + tokens

        let output = layer.attentionCacheStep(
            queries: queries, keys: step[StepTensor.keys.rawValue][0..., ..<Self.kvHeads],
            values: step[StepTensor.values.rawValue][0..., ..<Self.kvHeads], cache: cache,
            mask: Self.mask(kind, rotatedQueries: rotatedQueries, keys: keys))
        let expected = Self.routeOutput(
            with: kind, picksFrom: kernels, rotatedQueries: rotatedQueries,
            attention: attention, cache: cache)

        Self.expectTheStep(
            output, gives: expected, cache: cache, keys: keys, sourceLocation: sourceLocation)
    }

    /// Checks that a step wrote its keys into the cache and gave the
    /// expected output bit for bit.
    ///
    /// - Parameters:
    ///   - output: The output of the step.
    ///   - expected: The output of the route, from ``routeOutput(with:picksFrom:rotatedQueries:attention:cache:)``.
    ///   - cache: The cache after the step.
    ///   - keys: The number of keys the cache must hold after the step.
    ///   - sourceLocation: The location that a failure reports.
    static func expectTheStep(
        _ output: MLXArray, gives expected: MLXArray, cache: KVCacheSimple, keys: Int,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(cache.offset == keys, sourceLocation: sourceLocation)
        AttentionKernelTolerance.expectBitForBit(output, expected, sourceLocation: sourceLocation)
    }
}
