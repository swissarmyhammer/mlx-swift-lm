// Copyright © 2026 Apple Inc.
//
// Pins `decodeConv` bit-for-bit against `generalConv` (MLX's `Convolution`
// kernel) through ``GDNDecodeConvolutionComparison``; a mismatch means compiled
// decode silently diverges from prefill.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35GDNDecodeBitwiseTests: XCTestCase {

    /// 256 conv channels — wide enough that an accumulation-order change in
    /// MLX's Convolution kernel cannot pass by coincidence (native-dtype
    /// accumulation diverges in ~47% of channels).
    private func tinyGDNConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    func testDecodeConvMatchesConvolutionKernelBitwise() throws {
        let config = try tinyGDNConfiguration()
        for dtype in [DType.float16, DType.bfloat16] {
            withRandomState(MLXRandom.RandomState(seed: 11)) {
                GDNDecodeConvolutionComparison.assertFusedMatchesGeneral(
                    Qwen35GatedDeltaNet(config), dtype: dtype)
            }
        }
    }
}
