// Copyright © 2026 Apple Inc.
//
// A Qwen3.8 prefill block reads its key cache through
// ``PrefillAttentionKernel``: the general attention body of a full-attention
// layer gives the queries of the block to it. The kernel is deterministic,
// thus the body must give the kernel output bit for bit. The SDPA of MLX adds
// in another order, so a body that kept it fails this test.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct Qwen35PrefillAttentionTests {

    /// The cached keys ahead of the prefill block.
    private static let cachedKeys = 76

    /// The query tokens of the prefill block: the fewest the kernel reads,
    /// and some more.
    private static let blockTokens = PrefillAttentionKernel.fewestQueries + 9

    /// The layer: a head of 256 numbers, the only width the kernel reads.
    private static let fixture = Qwen35AttentionLayerFixture(headDim: 256)

    /// The seed of each random operand.
    private enum Seed {
        static let cache: UInt64 = 20_261_120
        static let block: UInt64 = 20_261_121
    }

    @Test func thePrefillBlockGivesTheKernelOutput() throws {
        let attention = try #require(try Self.fixture.layer().selfAttn)
        let cache = Self.fixture.filledCache(keys: Self.cachedKeys, seed: Seed.cache)
        let x = MLXRandom.normal(
            [1, Self.blockTokens, Qwen35AttentionLayerFixture.hiddenSize],
            key: MLXRandom.key(Seed.block))
        let (queries, gate, _, _) = attention.projectPreRope(x)
        let rotatedQueries = applyRotaryPosition(
            attention.rope, to: queries, offset: cache.ropeOffset)

        let output = attention(x, mask: .causal, cache: cache)
        let state = cache.state
        let expected = attention.mergeHeadsAndProject(
            attention: PrefillAttentionKernel.attention(
                queries: rotatedQueries, keys: state[0], values: state[1],
                scale: attention.scale, causal: true),
            gate: gate)

        #expect(cache.offset == Self.cachedKeys + Self.blockTokens)
        #expect(arrayEqual(output, expected).item(Bool.self))
    }
}
