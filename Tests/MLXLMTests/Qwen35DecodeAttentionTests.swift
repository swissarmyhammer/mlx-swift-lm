// Copyright © 2026 Apple Inc.
//
// A Qwen3.5 / Qwen3.8 decode step reads a long key cache through
// ``DecodeAttentionKernel``. Both decode paths of a full-attention layer use
// it: the traced decode step (`attentionCacheStep`) and the general attention
// body. The kernel is deterministic, thus each path must give the kernel
// output bit for bit. The SDPA of MLX adds in another order, so a path that
// kept it fails these tests. A step with a causal mask or a mask array keeps
// the SDPA of MLX, bit for bit.

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
    private static let seeds = Qwen35AttentionLayerFixture.Seeds(
        cache: 20_261_020, tokens: 20_261_021)

    /// The traced decode step gives the kernel output with no mask, and the
    /// SDPA of MLX with a causal mask or a mask array, in each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func theTracedDecodeStepGivesTheKernelOutputOrTheSDPA(
        dtype: DType, mask: AttentionMaskKind
    ) throws {
        try Self.fixture.expectTheTracedStepGivesTheRouteOutput(
            cachedKeys: Self.cachedKeys, tokens: 1, dtype: dtype, mask: mask,
            picksFrom: DecodeAttentionKernelTests.kernels, seeds: Self.seeds)
    }

    /// The general attention body of one token gives the kernel output with
    /// no mask, and the SDPA of MLX with a causal mask or a mask array, in
    /// each dtype.
    @Test(arguments: AttentionKernelTolerance.dtypes, AttentionMaskKind.allCases)
    func theGeneralAttentionBodyGivesTheKernelOutputOrTheSDPA(
        dtype: DType, mask: AttentionMaskKind
    ) throws {
        try Self.fixture.expectTheAttentionBodyGivesTheRouteOutput(
            cachedKeys: Self.cachedKeys, tokens: 1, dtype: dtype, mask: mask,
            picksFrom: DecodeAttentionKernelTests.kernels, seeds: Self.seeds)
    }
}
