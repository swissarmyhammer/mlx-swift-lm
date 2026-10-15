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

    /// The layer: a head of 64 numbers.
    private static let fixture = Qwen35AttentionLayerFixture(headDim: 64)

    /// The seed of each random operand.
    private enum Seed {
        static let cache: UInt64 = 20_261_020
        static let step: UInt64 = 20_261_021
    }

    /// A key cache that holds ``cachedKeys`` random keys and values.
    private static func filledCache() -> KVCacheSimple {
        fixture.filledCache(keys: cachedKeys, seed: Seed.cache)
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
        let layer = try Self.fixture.layer()
        let attention = try #require(layer.selfAttn)
        let cache = Self.filledCache()
        let step = MLXRandom.normal(
            [3, 1, Qwen35AttentionLayerFixture.queryHeads, 1, Self.fixture.headDim],
            key: MLXRandom.key(Seed.step))
        let queries = step[0]
        let keys = step[1][0..., ..<Qwen35AttentionLayerFixture.kvHeads]
        let values = step[2][0..., ..<Qwen35AttentionLayerFixture.kvHeads]
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
        let attention = try #require(try Self.fixture.layer().selfAttn)
        let cache = Self.filledCache()
        let x = MLXRandom.normal(
            [1, 1, Qwen35AttentionLayerFixture.hiddenSize], key: MLXRandom.key(Seed.step))
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
