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
    private static let seeds = Qwen35AttentionLayerFixture.Seeds(
        cache: 20_261_120, tokens: 20_261_121)

    @Test func thePrefillBlockGivesTheKernelOutput() throws {
        try Self.fixture.expectTheAttentionBodyGivesTheKernelOutput(
            cachedKeys: Self.cachedKeys, tokens: Self.blockTokens, mask: .causal,
            seeds: Self.seeds, kernel: PrefillAttentionKernelTests.call(causal: true).attention)
    }
}
