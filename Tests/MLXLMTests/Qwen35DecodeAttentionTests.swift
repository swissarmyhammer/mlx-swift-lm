// Copyright © 2026 Apple Inc.
//
// A Qwen3.5 / Qwen3.8 decode step reads a long key cache through
// ``DecodeAttentionKernel``. Both decode paths of a full-attention layer use
// it: the traced decode step (`attentionCacheStep`) and the general attention
// body. The kernel is deterministic, thus each path must give the kernel
// output bit for bit. The SDPA of MLX adds in another order, so a path that
// kept it fails these tests.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35DecodeAttentionTests {

    /// The cached keys ahead of the decode step: the fewest keys the kernel
    /// reads, and some more.
    private static let cachedKeys = DecodeAttentionKernel.fewestKeys + 76

    /// The width of one attention head.
    private static let headDim = 64

    /// The number of query heads; six for each KV head, as in Qwen3.8.
    private static let queryHeads = 6

    /// The number of KV heads.
    private static let kvHeads = 1

    /// The width of the residual stream.
    private static let hiddenSize = 32

    /// The index of the full-attention layer: with a full-attention interval
    /// of 2, layer 1 is the first full-attention layer.
    private static let attentionLayer = 1

    /// The seed of each random operand.
    private enum Seed {
        static let cache: UInt64 = 20_261_020
        static let step: UInt64 = 20_261_021
    }

    /// A small dense Qwen3.5 text configuration with one full-attention layer.
    private static func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": \(queryHeads),
                "num_key_value_heads": \(kvHeads),
                "head_dim": \(headDim),
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// A key cache that holds ``cachedKeys`` random keys and values.
    private static func filledCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        let rows = MLXRandom.normal(
            [2, 1, kvHeads, cachedKeys, headDim], key: MLXRandom.key(Seed.cache))
        _ = cache.update(keys: rows[0], values: rows[1])
        eval(cache.state)
        return cache
    }

    /// The kernel output over the cache after the step wrote its key.
    private static func kernelOutput(
        rotatedQueries: MLXArray, cache: KVCacheSimple, scale: Float
    ) -> MLXArray {
        let state = cache.state
        return DecodeAttentionKernel.attention(
            queries: rotatedQueries, keys: state[0], values: state[1], scale: scale)
    }

    @Test func theTracedDecodeStepGivesTheKernelOutput() throws {
        let layer = Qwen35DecoderLayer(try Self.configuration(), layerIdx: Self.attentionLayer)
        let attention = try #require(layer.selfAttn)
        let cache = Self.filledCache()
        let step = MLXRandom.normal(
            [3, 1, Self.queryHeads, 1, Self.headDim], key: MLXRandom.key(Seed.step))
        let queries = step[0]
        let keys = step[1][0..., ..<Self.kvHeads]
        let values = step[2][0..., ..<Self.kvHeads]
        let rotatedQueries = applyRotaryPosition(
            attention.rope, to: queries, offset: cache.ropeOffset)

        let output = layer.attentionCacheStep(
            queries: queries, keys: keys, values: values, cache: cache, mask: .none)
        let expected = Self.kernelOutput(
            rotatedQueries: rotatedQueries, cache: cache, scale: attention.scale)

        #expect(cache.offset == Self.cachedKeys + 1)
        #expect(arrayEqual(output, expected).item(Bool.self))
    }

    @Test func theGeneralAttentionBodyGivesTheKernelOutput() throws {
        let layer = Qwen35DecoderLayer(try Self.configuration(), layerIdx: Self.attentionLayer)
        let attention = try #require(layer.selfAttn)
        let cache = Self.filledCache()
        let x = MLXRandom.normal([1, 1, Self.hiddenSize], key: MLXRandom.key(Seed.step))
        let (queries, gate, _, _) = attention.projectPreRope(x)
        let rotatedQueries = applyRotaryPosition(
            attention.rope, to: queries, offset: cache.ropeOffset)

        let output = attention(x, mask: .none, cache: cache)
        let expected = attention.mergeHeadsAndProject(
            attention: Self.kernelOutput(
                rotatedQueries: rotatedQueries, cache: cache, scale: attention.scale),
            gate: gate)

        #expect(cache.offset == Self.cachedKeys + 1)
        #expect(arrayEqual(output, expected).item(Bool.self))
    }
}
