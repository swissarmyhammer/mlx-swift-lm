import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

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

    /// The layout of ``configJSON`` with every projection width a multiple of
    /// ``quantizationGroupSize``, so that each layer quantizes.
    private var quantizableConfigJSON: String {
        configJSON
            .replacingOccurrences(of: "\"hidden_size\": 16", with: "\"hidden_size\": 64")
            .replacingOccurrences(
                of: "\"intermediate_size\": 32", with: "\"intermediate_size\": 64"
            )
            .replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 16")
            .replacingOccurrences(
                of: "\"moe_intermediate_size\": 8", with: "\"moe_intermediate_size\": 32"
            )
            .replacingOccurrences(
                of: "\"shared_expert_intermediate_size\": 16",
                with: "\"shared_expert_intermediate_size\": 32")
    }

    /// The group size of the quantized test checkpoints.
    private let quantizationGroupSize = 32

    private func makeModel(json: String? = nil) throws -> FlintModel {
        let config = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data((json ?? configJSON).utf8))
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

    // MARK: - Parameter dtypes at load

    /// The expert bias of the first MoE layer.
    private let expertBiasKey = "language_model.model.layers.2.mlp.expert_bias"
    /// The attention sinks of the first layer, and the query norm beside them.
    private let sinksKey = "language_model.model.layers.0.self_attn.softmax_offset"
    private let queryNormKey = "language_model.model.layers.0.self_attn.q_norm.weight"

    /// The parameters of `model` as a checkpoint, with the tensor under `key`
    /// stored in `dtype`.
    private func checkpointWeights(
        of model: FlintModel, storing key: String, in dtype: DType
    ) throws -> [String: MLXArray] {
        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        weights[key] = try XCTUnwrap(weights[key]).asType(dtype)
        return weights
    }

    /// The router adds the expert bias to float32 scores, so a bfloat16 bias
    /// loads in float32 and no forward pass casts it.
    func testSanitizeStoresTheExpertBiasInFloat32() throws {
        let model = try makeModel()
        let weights = try checkpointWeights(of: model, storing: expertBiasKey, in: .bfloat16)
        XCTAssertEqual(model.sanitize(weights: weights)[expertBiasKey]?.dtype, .float32)
    }

    /// The sinks load in the dtype of the queries that the query norm makes, so
    /// no forward pass casts them.
    func testSanitizeStoresTheSinksInTheDtypeOfTheQueryNorm() throws {
        let model = try makeModel()
        let weights = try checkpointWeights(of: model, storing: queryNormKey, in: .bfloat16)
        XCTAssertEqual(model.sanitize(weights: weights)[sinksKey]?.dtype, .bfloat16)
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

    // MARK: - Projection fusion at load

    /// 4-bit for every layer, with the overrides of `overrides` on top.
    private func quantizationPlan(
        overrides: [String: BaseConfiguration.QuantizationOption] = [:]
    ) -> BaseConfiguration.PerLayerQuantization {
        .init(
            quantization: .init(groupSize: quantizationGroupSize, bits: 4),
            perLayerQuantization: overrides)
    }

    /// A quantized model and its checkpoint, in the module layout of an mlx-vlm
    /// checkpoint (separate projections, `language_model.` prefix).
    private func makeQuantizedCheckpoint(
        plan: BaseConfiguration.PerLayerQuantization
    ) throws -> (reference: FlintModel, checkpoint: ModelCheckpoint) {
        let reference = try makeModel(json: quantizableConfigJSON)
        quantize(model: reference) { path, _ in
            plan.quantization(layer: path).map {
                (groupSize: $0.groupSize, bits: $0.bits, mode: $0.mode)
            }
        }
        eval(reference)
        let weights = Dictionary(
            uniqueKeysWithValues: reference.parameters().flattened().map { ($0.0, $0.1) })
        return (reference, ModelCheckpoint(weights: weights, perLayerQuantization: plan))
    }

    /// Loads `checkpoint` into a new model the way the model loader does.
    private func load(_ checkpoint: ModelCheckpoint) throws -> FlintModel {
        let model = try makeModel(json: quantizableConfigJSON)
        try applyCheckpoint(try model.prepareCheckpoint(checkpoint), to: model)
        eval(model)
        return model
    }

    private func parameterKeys(_ model: FlintModel) -> Set<String> {
        Set(model.parameters().flattened().map(\.0))
    }

    /// q/k/v, the dense and shared-expert gate/up and the routed gate/up each load
    /// as one projection when their parts share a quantization.
    func testLoadFusesProjectionsThatShareQuantization() throws {
        let (_, checkpoint) = try makeQuantizedCheckpoint(plan: quantizationPlan())
        let keys = parameterKeys(try load(checkpoint))
        let expected: Set = [
            "language_model.model.layers.0.self_attn.qkv_proj.scales",
            "language_model.model.layers.0.mlp.gate_up_proj.scales",
            "language_model.model.layers.2.self_attn.qkv_proj.scales",
            "language_model.model.layers.2.mlp.shared_expert.gate_up_proj.scales",
            "language_model.model.layers.2.mlp.switch_mlp.gate_up_proj.scales",
        ]
        XCTAssertEqual(expected.subtracting(keys), [])
    }

    /// The fused model computes the logits of the separate projections, for a
    /// prefill and for a decode step.
    func testFusedProjectionsMatchSeparateProjections() throws {
        let (reference, checkpoint) = try makeQuantizedCheckpoint(plan: quantizationPlan())
        let fused = try load(checkpoint)
        XCTAssertTrue(
            parameterKeys(fused).contains("language_model.model.layers.2.self_attn.qkv_proj.weight")
        )

        let prompt = MLXArray((0 ..< 10).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, -1)
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        let referenceCache = try reference.newCache(parameters: nil)
        let fusedCache = try fused.newCache(parameters: nil)
        let expected = concatenated(
            [reference(prompt, cache: referenceCache), reference(next, cache: referenceCache)],
            axis: 1)
        let actual = concatenated(
            [fused(prompt, cache: fusedCache), fused(next, cache: fusedCache)], axis: 1)

        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
    }

    /// Parts with different quantizations keep their own layers.
    func testLoadKeepsProjectionsWhoseQuantizationDiffers() throws {
        let eightBit = BaseConfiguration.QuantizationOption.quantize(
            .init(groupSize: quantizationGroupSize, bits: 8))
        let plan = quantizationPlan(overrides: [
            "language_model.model.layers.2.self_attn.k_proj": eightBit,
            "language_model.model.layers.2.mlp.shared_expert.up_proj": eightBit,
        ])
        let (_, checkpoint) = try makeQuantizedCheckpoint(plan: plan)
        let keys = parameterKeys(try load(checkpoint))
        let separate: Set = [
            "language_model.model.layers.2.self_attn.q_proj.scales",
            "language_model.model.layers.2.self_attn.k_proj.scales",
            "language_model.model.layers.2.self_attn.v_proj.scales",
            "language_model.model.layers.2.mlp.shared_expert.gate_proj.scales",
            "language_model.model.layers.2.mlp.shared_expert.up_proj.scales",
        ]
        XCTAssertEqual(separate.subtracting(keys), [])
    }

    /// A checkpoint without quantization keeps its layout, so a conversion
    /// saves the projections under their checkpoint names.
    func testLoadWithoutQuantizationKeepsSeparateProjections() throws {
        let reference = try makeModel(json: quantizableConfigJSON)
        let weights = Dictionary(
            uniqueKeysWithValues: reference.parameters().flattened().map { ($0.0, $0.1) })
        let keys = parameterKeys(try load(ModelCheckpoint(weights: weights)))
        XCTAssertEqual(keys, Set(weights.keys))
    }
}
