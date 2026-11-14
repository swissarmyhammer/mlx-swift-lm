import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tiny-config tests of the Flint port: configuration, weight layout, and the
/// sliding-window / full-attention cache path.
final class FlintTests: XCTestCase {

    /// Four layers: two dense, then two MoE; layers 0 and 3 have no RoPE, and
    /// layers 1 and 2 slide over a window of 4.
    private let configJSON = """
        {
            "model_type": "flint", "hidden_size": 16, "num_hidden_layers": 4,
            "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 256, "rope_theta": 10000.0,
            "layer_types": ["full_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "no_rope_layer_indices": [0, 3], "sliding_window": 4,
            "use_sliding_window": true, "softmax_type": "learnable",
            "mlp_only_layers": [0, 1], "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 8, "shared_expert_intermediate_size": 16,
            "router_score_function": "sigmoid", "norm_topk_prob": true,
            "hidden_act": "silu", "tie_word_embeddings": false
        }
        """

    private func makeModel() throws -> FlintModel {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configJSON.utf8))
        let model = FlintModel(config)
        // Nonzero sinks and expert biases, so the paths that read them matter.
        var updates = [String: MLXArray]()
        for (key, value) in model.parameters().flattened()
        where key.hasSuffix("softmax_offset") || key.hasSuffix("expert_bias") {
            updates[key] = MLXRandom.normal(value.shape) * 0.5
        }
        try model.update(parameters: ModuleParameters.unflattened(updates), verify: [])
        eval(model)
        return model
    }

    func testConfigurationDecodesLayerKinds() throws {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(configJSON.utf8))
        XCTAssertEqual((0 ..< 4).map { config.window(layer: $0) }, [nil, 4, 4, nil])

        let model = FlintModel(config)
        let layers = model.languageModel.model.layers
        XCTAssertNil(layers[0].selfAttn.rope)
        XCTAssertNotNil(layers[1].selfAttn.rope)
        XCTAssertNil(layers[3].selfAttn.rope)
        XCTAssertTrue(layers[1].mlp is FlintMLP)
        XCTAssertTrue(layers[2].mlp is FlintSparseMoeBlock)

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.map { $0 is RotatingKVCache }, [false, true, true, false])
    }

    func testRejectsUnsupportedRouting() {
        let json = configJSON.replacingOccurrences(
            of: "\"router_score_function\": \"sigmoid\"",
            with: "\"router_score_function\": \"softmax\"")
        XCTAssertThrowsError(
            try JSONDecoder().decode(FlintConfiguration.self, from: Data(json.utf8)))
    }

    func testParametersUseTheCheckpointLayout() throws {
        let keys = Set(try makeModel().parameters().flattened().map(\.0))
        for key in [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.layers.0.self_attn.softmax_offset",
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.2.mlp.gate.weight",
            "language_model.model.layers.2.mlp.expert_bias",
            "language_model.model.layers.2.mlp.switch_mlp.up_proj.weight",
            "language_model.model.layers.2.mlp.shared_expert.down_proj.weight",
        ] {
            XCTAssertTrue(keys.contains(key), "missing \(key)")
        }
    }

    /// A Hugging Face checkpoint (no prefix, one tensor per expert, the router
    /// bias under the gate) sanitizes to the module layout.
    func testSanitizeStacksExpertsAndAddsPrefix() throws {
        let model = try makeModel()
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })

        var raw = [String: MLXArray]()
        for (key, value) in expected {
            let name = String(key.dropFirst("language_model.".count))
            if let range = name.range(of: ".switch_mlp.") {
                for expert in 0 ..< value.dim(0) {
                    raw[name.replacingCharacters(in: range, with: ".experts.\(expert).")] =
                        value[expert]
                }
            } else if name.hasSuffix(".mlp.expert_bias") {
                raw[
                    name.replacingOccurrences(
                        of: ".expert_bias", with: ".gate.e_score_correction_bias")] =
                    value
            } else {
                raw[name] = value
            }
        }

        let sanitized = model.sanitize(weights: raw)
        XCTAssertEqual(Set(sanitized.keys), Set(expected.keys))
        for (key, value) in expected {
            XCTAssertTrue(allClose(sanitized[key]!, value).item(Bool.self), key)
        }

        // An mlx-vlm checkpoint is already in the module layout.
        XCTAssertEqual(Set(model.sanitize(weights: expected).keys), Set(expected.keys))
    }

    /// Prefill and token-by-token decode agree past the sliding window, which
    /// checks the rotating cache, its masks, the sinks and the no-RoPE layers.
    func testIncrementalDecodeMatchesFullPrefill() throws {
        let model = try makeModel()
        let tokens = (0 ..< 11).map { Int32(($0 * 7 + 3) % 64) }
        let all = MLXArray(tokens).reshaped(1, -1)

        let fullCache = try model.newCache(parameters: nil)
        let full = model(all, cache: fullCache)

        let stepCache = try model.newCache(parameters: nil)
        let prefix = 3
        var steps = [model(all[0..., ..<prefix], cache: stepCache)]
        for i in prefix ..< tokens.count {
            steps.append(model(all[0..., i ..< (i + 1)], cache: stepCache))
        }
        let stepped = concatenated(steps, axis: 1)

        XCTAssertEqual(full.dtype, .float32)
        XCTAssertEqual(stepped.shape, full.shape)
        let difference = abs(stepped - full).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-4)
        for cache in fullCache + stepCache {
            XCTAssertEqual(cache.offset, tokens.count)
        }
    }
}
