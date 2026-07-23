// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Tests for MiniMax-M3's swigluoai activation, sparse MoE block (kanban
/// ^mv9aq7w), and the full `MiniMaxM3TextConfiguration` + dense-attention
/// decoder/model/sanitize (kanban ^xgvth41). MSA sparse attention itself is
/// not built here -- that is a follow-up task (^8dbc476); every layer uses
/// dense `MiniMaxM3Attention`.
///
/// Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py`
/// (`MiniMaxSwiGLUOAI`, `MiniMaxPackedSwitchGLU`, `MiniMaxSparseMoeBlock`,
/// `_minimax_moe_select`), cross-checked against upstream mlx-lm PRs
/// #1398/#1401 and the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `config.json` / `model.safetensors.index.json` / safetensors shard
/// headers (downloaded directly while implementing ^xgvth41).
///
/// `.serialized`: several tests here seed the shared `MLXRandom` global state
/// and depend on it not being perturbed by other tests running concurrently
/// on the same thread pool (mirrors `RoPEApplicationTests`'s rationale).
@Suite(.serialized)
struct MiniMaxM3Tests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - MiniMaxM3SwiGLUOAI

    /// mlx-vlm's `_swiglu_oai` reference formula:
    ///   x_glu = clip(x_glu, max: limit)                  // upper bound only
    ///   x_linear = clip(x_linear, min: -limit, max: limit)
    ///   return x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)
    private func referenceSwiGLUOAI(
        xLinear: Float, xGlu: Float, alpha: Float, limit: Float, beta: Float
    ) -> Float {
        let glu = min(xGlu, limit)
        let linear = max(min(xLinear, limit), -limit)
        let sig = 1 / (1 + expf(-alpha * glu))
        return glu * sig * (linear + beta)
    }

    @Test("interior values match the reference formula")
    func interiorMatchesReferenceFormula() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.5
        let xGlu: Float = -0.25
        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("gate branch clips only the upper bound, leaving very negative gates untouched")
    func gateClipsUpperBoundOnly() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.0
        let xGlu: Float = -8.0  // below -limit

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        // Correct (asymmetric) formula: gate stays -8.
        let expectedUnclamped = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        // A wrong symmetric clamp would instead use gate == -7.
        let expectedIfWronglySymmetric = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: -7.0, alpha: 1.702, limit: 7.0, beta: 1.0)

        #expect(abs(out.item(Float.self) - expectedUnclamped) < 1e-8)
        #expect(abs(out.item(Float.self) - expectedIfWronglySymmetric) > 1e-6)
    }

    @Test("gate branch clips at the upper limit boundary")
    func gateClipsAtUpperBoundary() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.25
        let xGlu: Float = 12.0  // above limit

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: 7.0, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("linear branch clips symmetrically at both bounds", arguments: [12.0 as Float, -12.0 as Float])
    func linearClipsSymmetrically(xLinear: Float) {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xGlu: Float = 1.0

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let clampedLinear: Float = xLinear > 0 ? 7.0 : -7.0
        let expected = referenceSwiGLUOAI(
            xLinear: clampedLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("alpha, limit, and beta are configuration-driven, not hardcoded")
    func configDrivenParameters() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.0, limit: 3.0, beta: 0.0)
        let xLinear: Float = 1.0
        let xGlu: Float = 5.0  // above the custom limit of 3.0

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: 3.0, alpha: 1.0, limit: 3.0, beta: 0.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    // MARK: - MiniMaxM3SparseMoeBlock

    private func tinyConfig(
        numLocalExperts: Int = 8, numExpertsPerTok: Int = 2, hiddenSize: Int = 16,
        intermediateSize: Int = 32
    ) -> MiniMaxM3MoEConfiguration {
        MiniMaxM3MoEConfiguration(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numLocalExperts: numLocalExperts,
            numExpertsPerTok: numExpertsPerTok
        )
    }

    @Test("output shape is (B, L, hiddenSize)")
    func outputShapeMatchesInput() {
        MLXRandom.seed(0)
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        let x = MLXRandom.normal([2, 3, config.hiddenSize])
        let y = block(x)
        eval(y)

        #expect(y.shape == [2, 3, config.hiddenSize])
    }

    @Test("switch_mlp packs the shared expert as one extra expert row")
    func switchMLPHasOneMoreExpertThanRouted() {
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        #expect(block.switchMLP.numExperts == config.numLocalExperts + 1)
    }

    @Test("zeroing the packed shared-expert weights changes the output")
    func sharedExpertContributionIsObservable() throws {
        MLXRandom.seed(1)
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        let x = MLXRandom.normal([1, 4, config.hiddenSize])
        let baseline = block(x)
        eval(baseline)

        // The packed shared expert is the last (index numLocalExperts) row of
        // the fused switch_mlp weight tensors -- zero it out.
        let gateUp = block.switchMLP.gateUpProj.weight
        gateUp[config.numLocalExperts...] = MLXArray.zeros(like: gateUp[config.numLocalExperts...])
        let down = block.switchMLP.downProj.weight
        down[config.numLocalExperts...] = MLXArray.zeros(like: down[config.numLocalExperts...])

        try block.update(
            parameters: ModuleParameters.unflattened([
                "switch_mlp.gate_up_proj.weight": gateUp,
                "switch_mlp.down_proj.weight": down,
            ]), verify: [])

        let zeroedShared = block(x)
        eval(zeroedShared)

        let maxDiff = MLX.abs(baseline - zeroedShared).max().item(Float.self)
        #expect(maxDiff > 1e-6)
    }

    @Test("routing selects via biased scores but weights via unbiased, normalized, scaled scores")
    func routingUsesBiasedSelectionAndUnbiasedScaledWeights() throws {
        // 4 routed experts, top-2, hidden == numLocalExperts so the router
        // can be an identity matrix (logit_i == x_i), mirroring
        // LFM2MoeRoutingTests's style.
        let config = MiniMaxM3MoEConfiguration(
            hiddenSize: 4, intermediateSize: 8, numLocalExperts: 4, numExpertsPerTok: 2,
            routedScalingFactor: 2.0)
        let block = MiniMaxM3SparseMoeBlock(config)
        try block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": MLX.identity(4),
                "e_score_correction_bias": MLXArray([Float(0), 0, 1, 0]),
            ]), verify: [])
        eval(block)

        let logits: [Float] = [2, 1, 0, -1]
        let x = MLXArray(logits).reshaped(1, 1, 4)

        let (indices, weights) = block.route(x)
        eval(indices, weights)

        func sig(_ v: Float) -> Float { 1 / (1 + expf(-v)) }

        let idx = indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = weights.reshaped(-1).asArray(Float.self)
        let routed = Dictionary(uniqueKeysWithValues: zip(idx, w))

        // Selection uses the *biased* scores: expert 2's correction bias of
        // +1 promotes it ahead of expert 1 into the top-2, even though
        // sigmoid(0) < sigmoid(1) on the raw (unbiased) scores.
        #expect(Set(routed.keys) == [0, 2])

        // Weighting uses the *unbiased* scores, renormalized to sum to 1,
        // then scaled by routed_scaling_factor (2.0).
        let denom = sig(2) + sig(0)
        #expect(abs((routed[0] ?? .nan) - (sig(2) / denom) * 2.0) < 1e-4)
        #expect(abs((routed[2] ?? .nan) - (sig(0) / denom) * 2.0) < 1e-4)
    }

    @Test("the packed shared expert contributes with unscaled weight 1.0, not routed_scaling_factor")
    func sharedExpertWeightIsUnscaled() throws {
        // A distinctive routed_scaling_factor far from 1.0: if the shared
        // expert's weight were (incorrectly) also multiplied by it, the
        // assertion below would fail.
        let config = MiniMaxM3MoEConfiguration(
            hiddenSize: 4, intermediateSize: 8, numLocalExperts: 4, numExpertsPerTok: 2,
            routedScalingFactor: 5.0)
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        // Zero every *routed* expert's weights (rows 0..<numLocalExperts).
        // Since swigluoai(0, 0) == 0, their contribution is exactly zero
        // regardless of which are selected or how they're weighted, so
        // `block(x)` reduces to exactly `1.0 * sharedExpert(x)`.
        let gateUp = block.switchMLP.gateUpProj.weight
        gateUp[..<config.numLocalExperts] = MLXArray.zeros(
            like: gateUp[..<config.numLocalExperts])
        let down = block.switchMLP.downProj.weight
        down[..<config.numLocalExperts] = MLXArray.zeros(like: down[..<config.numLocalExperts])

        try block.update(
            parameters: ModuleParameters.unflattened([
                "switch_mlp.gate_up_proj.weight": gateUp,
                "switch_mlp.down_proj.weight": down,
            ]), verify: [])

        let x = MLXRandom.normal([1, 2, config.hiddenSize])
        let output = block(x)
        eval(output)

        // Manually compute the shared expert's output with an explicit
        // weight of 1.0, independent of `routed_scaling_factor`.
        let sharedIndices = MLXArray.full(
            [1, 2, 1], values: MLXArray(Int32(config.numLocalExperts)), dtype: .int32)
        let sharedOnly = block.switchMLP(x, sharedIndices).squeezed(axis: -2)
        eval(sharedOnly)

        let maxDiff = MLX.abs(output - sharedOnly).max().item(Float.self)
        #expect(maxDiff < 1e-4)
    }

    // MARK: - Configuration fixtures

    /// Trimmed (but otherwise faithful) reproduction of
    /// `https://huggingface.co/mlx-community/MiniMax-M3-4bit/blob/main/config.json`,
    /// downloaded directly while implementing ^xgvth41. Trimmed only for
    /// test-file size: the giant per-layer `quantization`/`quantization_config`
    /// override maps (one entry per MoE layer, 3..<60) and the two 60-length
    /// `sparse_attention_config` schedule arrays (`sparse_disable_index_value`,
    /// `sparse_attention_freq`) are shortened to a few representative
    /// elements -- every `text_config` field relevant to this task is
    /// reproduced verbatim from the real file.
    private static let realVLConfigJSON = """
        {
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "model_type": "minimax_m3_vl",
            "image_token_index": 200025,
            "video_token_index": 200026,
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "max_position_embeddings": 1048576,
                "rms_norm_eps": 1e-06,
                "use_gemma_norm": true,
                "attention_output_gate": false,
                "rope_theta": 5000000,
                "rotary_dim": 64,
                "partial_rotary_factor": 0.5,
                "hidden_act": "swigluoai",
                "use_qk_norm": true,
                "tie_word_embeddings": false,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "moe_layer_freq": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "qk_norm_type": "per_head",
                "num_mtp_modules": 7,
                "num_nextn_predict_layers": 1,
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "routed_scaling_factor": 2.0,
                "sparse_attention_config": {
                    "use_sparse_attention": true,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_disable_index_value": [0, 0, 0, 1, 1],
                    "sparse_score_type": "max",
                    "sparse_init_block": 0,
                    "sparse_local_block": 1,
                    "sparse_attention_freq": [0, 0, 0, 1, 1]
                },
                "architectures": ["MiniMaxM3SparseForCausalLM"]
            },
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.layers.3.block_sparse_moe.gate": {
                    "group_size": 64,
                    "bits": 8,
                    "mode": "affine"
                }
            },
            "vision_config": {
                "hidden_size": 1280,
                "num_attention_heads": 16,
                "num_hidden_layers": 32,
                "intermediate_size": 5120,
                "patch_size": 14,
                "image_size": 2016,
                "model_type": "clip_vision_model",
                "num_channels": 3
            },
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full"
        }
        """

    /// Synthetic flat `minimax_m3` config (upstream mlx-lm PR #1401-style
    /// text-only conversion): the same `text_config` field values as
    /// `realVLConfigJSON`, but at the JSON root with no VL nesting.
    private static let flatTextOnlyConfigJSON = """
        {
            "model_type": "minimax_m3",
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 200064,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-06,
            "use_gemma_norm": true,
            "rope_theta": 5000000,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "hidden_act": "swigluoai",
            "use_qk_norm": true,
            "tie_word_embeddings": false,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "scoring_func": "sigmoid",
            "use_routing_bias": true,
            "moe_layer_freq": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "qk_norm_type": "per_head",
            "num_mtp_modules": 7,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "routed_scaling_factor": 2.0,
            "sparse_attention_config": {
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128
            }
        }
        """

    // MARK: - MiniMaxM3TextConfiguration / MiniMaxM3Configuration decode

    @Test("decodes the real nested VL config.json fixture")
    func decodesRealNestedVLConfig() throws {
        let data = Self.realVLConfigJSON.data(using: .utf8)!
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: data)

        #expect(config.modelType == "minimax_m3_vl")

        let text = config.textConfiguration
        #expect(text.hiddenSize == 6144)
        #expect(text.hiddenLayers == 60)
        #expect(text.attentionHeads == 64)
        #expect(text.kvHeads == 4)
        #expect(text.headDim == 128)
        #expect(text.vocabularySize == 200_064)
        #expect(text.maxPositionEmbeddings == 1_048_576)
        #expect(abs(text.rmsNormEps - 1e-6) < 1e-9)
        #expect(text.useGemmaNorm == true)
        #expect(abs(text.ropeTheta - 5_000_000) < 1)
        #expect(text.rotaryDim == 64)
        #expect(abs(text.partialRotaryFactor - 0.5) < 1e-6)
        #expect(text.hiddenAct == "swigluoai")
        #expect(text.useQkNorm == true)
        #expect(text.qkNormType == "per_head")
        #expect(text.tieWordEmbeddings == false)
        #expect(text.denseIntermediateSize == 12288)
        #expect(text.intermediateSize == 3072)
        #expect(text.sharedIntermediateSize == 3072)
        #expect(text.numLocalExperts == 128)
        #expect(text.numExpertsPerTok == 4)
        #expect(text.nSharedExperts == 1)
        #expect(text.scoringFunc == "sigmoid")
        #expect(text.useRoutingBias == true)
        #expect(abs(text.routedScalingFactor - 2.0) < 1e-6)
        #expect(abs(text.swigluAlpha - 1.702) < 1e-6)
        #expect(abs(text.swigluLimit - 7.0) < 1e-6)
        #expect(text.numMTPModules == 7)

        // moe_layer_freq: layers 0-2 dense, 3-59 MoE.
        #expect(text.moeLayerFreq.count == 60)
        #expect(!text.isMoELayer(0))
        #expect(!text.isMoELayer(1))
        #expect(!text.isMoELayer(2))
        #expect(text.isMoELayer(3))
        #expect(text.isMoELayer(59))

        #expect(text.sparseAttention.indexDim == 128)
        #expect(text.sparseAttention.numIndexHeads == 4)
        #expect(text.sparseAttention.topkBlocks == 16)
        #expect(text.sparseAttention.blockSize == 128)
    }

    @Test("flat minimax_m3 config decodes to the same text-config values as the nested fixture")
    func flatConfigMatchesNestedConfig() throws {
        let nested = try JSONDecoder().decode(
            MiniMaxM3Configuration.self, from: Self.realVLConfigJSON.data(using: .utf8)!)
        let flat = try JSONDecoder().decode(
            MiniMaxM3TextConfiguration.self, from: Self.flatTextOnlyConfigJSON.data(using: .utf8)!)

        let nestedText = nested.textConfiguration
        #expect(flat.hiddenSize == nestedText.hiddenSize)
        #expect(flat.hiddenLayers == nestedText.hiddenLayers)
        #expect(flat.attentionHeads == nestedText.attentionHeads)
        #expect(flat.kvHeads == nestedText.kvHeads)
        #expect(flat.headDim == nestedText.headDim)
        #expect(flat.vocabularySize == nestedText.vocabularySize)
        #expect(flat.rotaryDim == nestedText.rotaryDim)
        #expect(abs(flat.ropeTheta - nestedText.ropeTheta) < 1)
        #expect(flat.hiddenAct == nestedText.hiddenAct)
        #expect(flat.denseIntermediateSize == nestedText.denseIntermediateSize)
        #expect(flat.intermediateSize == nestedText.intermediateSize)
        #expect(flat.numLocalExperts == nestedText.numLocalExperts)
        #expect(flat.numExpertsPerTok == nestedText.numExpertsPerTok)
        #expect(flat.moeLayerFreq == nestedText.moeLayerFreq)
        #expect(flat.tieWordEmbeddings == nestedText.tieWordEmbeddings)
        #expect(flat.modelType == "minimax_m3")
    }

    @Test("a tiny synthetic config constructs without any config file")
    func tinySyntheticConfigConstructsDirectly() {
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 32, hiddenLayers: 4, attentionHeads: 4, kvHeads: 2, headDim: 8,
            vocabularySize: 48, denseIntermediateSize: 64, intermediateSize: 16,
            numLocalExperts: 4, numExpertsPerTok: 2, moeLayerFreq: [0, 0, 1, 1])

        #expect(config.hiddenLayers == 4)
        #expect(config.rotaryDim == 4)  // partialRotaryFactor 0.5 * headDim 8
        #expect(!config.isMoELayer(0))
        #expect(!config.isMoELayer(1))
        #expect(config.isMoELayer(2))
        #expect(config.isMoELayer(3))
    }

    @Test("decoding throws a DecodingError instead of crashing when use_gemma_norm is unsupported")
    func decodeThrowsForUnsupportedUseGemmaNorm() {
        let json = """
            { "model_type": "minimax_m3", "use_gemma_norm": false }
            """
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(MiniMaxM3TextConfiguration.self, from: Data(json.utf8))
        }
    }

    @Test(
        "decoding throws a DecodingError instead of crashing when qk_norm_type is unsupported and use_qk_norm is true"
    )
    func decodeThrowsForUnsupportedQkNormType() {
        let json = """
            { "model_type": "minimax_m3", "use_qk_norm": true, "qk_norm_type": "flat" }
            """
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(MiniMaxM3TextConfiguration.self, from: Data(json.utf8))
        }
    }

    @Test("an unsupported qk_norm_type does not throw when use_qk_norm is false")
    func decodeDoesNotThrowForUnsupportedQkNormTypeWhenDisabled() throws {
        let json = """
            { "model_type": "minimax_m3", "use_qk_norm": false, "qk_norm_type": "flat" }
            """
        let config = try JSONDecoder().decode(MiniMaxM3TextConfiguration.self, from: Data(json.utf8))
        #expect(config.qkNormType == "flat")
        #expect(config.useQkNorm == false)
    }

    // MARK: - Partial rotary

    /// Hand-rolled reference for `MLXFast.RoPE`'s non-traditional convention
    /// (verified against the C++ kernel source,
    /// `mlx/backend/metal/kernels/rope.metal`'s `rope_single_impl`): pairs
    /// `(x[i], x[i + dims/2])` for `i` in `[0, dims/2)` rotate by
    /// `theta = position * base^(-2i/dims)`:
    ///   rx1 = x1*cos(theta) - x2*sin(theta)
    ///   rx2 = x1*sin(theta) + x2*cos(theta)
    /// Dims beyond `dimensions` (here, 64..<128) pass through unchanged --
    /// this is the partial-rotary behavior `MiniMaxM3Attention` relies on
    /// (`rotary_dim: 64` out of `head_dim: 128`).
    @Test("only the first 64 of 128 head dims rotate; the tail passes through unchanged")
    func partialRotaryOnlyRotatesFirst64Dims() {
        let headDim = 128
        let rotaryDim = 64
        let base: Float = 5_000_000

        let rope = RoPE(dimensions: rotaryDim, traditional: false, base: base, scale: 1.0)

        // (B, heads, L, headDim) = (1, 1, 2, 128): row 0 is position 0
        // (identity -- theta==0), row 1 is position 1 (a real rotation).
        MLXRandom.seed(7)
        let x = MLXRandom.normal([1, 1, 2, headDim])
        let out = rope(x, offset: 0)
        eval(out, x)

        // Tail (dims 64..<128) is untouched at both positions.
        let tailIn = x[0, 0, 0..., rotaryDim...]
        let tailOut = out[0, 0, 0..., rotaryDim...]
        eval(tailIn, tailOut)
        #expect(MLX.abs(tailIn - tailOut).max().item(Float.self) == 0)

        // Hand-rolled reference at position 1 (row index 1), pair i=0
        // (columns 0 and 32) and pair i=10 (columns 10 and 42).
        for i in [0, 10, 31] {
            let x1 = x[0, 0, 1, i].item(Float.self)
            let x2 = x[0, 0, 1, i + rotaryDim / 2].item(Float.self)

            let freq = Foundation.pow(base, -Float(2 * i) / Float(rotaryDim))
            let theta = 1 * freq  // position 1
            let expectedRx1 = x1 * cos(theta) - x2 * sin(theta)
            let expectedRx2 = x1 * sin(theta) + x2 * cos(theta)

            let actualRx1 = out[0, 0, 1, i].item(Float.self)
            let actualRx2 = out[0, 0, 1, i + rotaryDim / 2].item(Float.self)

            #expect(abs(actualRx1 - expectedRx1) < 1e-3)
            #expect(abs(actualRx2 - expectedRx2) < 1e-3)
        }

        // Position 0 is the identity rotation (theta == 0 for every pair).
        let headIn = x[0, 0, 0, 0 ..< rotaryDim]
        let headOut = out[0, 0, 0, 0 ..< rotaryDim]
        eval(headIn, headOut)
        #expect(MLX.abs(headIn - headOut).max().item(Float.self) < 1e-5)
    }

    // MARK: - MiniMaxM3Attention QK-norm layout

    private func tinyAttentionConfig(
        hiddenSize: Int = 16, attentionHeads: Int = 4, kvHeads: Int = 2, headDim: Int = 8
    ) -> MiniMaxM3TextConfiguration {
        MiniMaxM3TextConfiguration(
            hiddenSize: hiddenSize, hiddenLayers: 1, attentionHeads: attentionHeads,
            kvHeads: kvHeads, headDim: headDim, vocabularySize: 32,
            denseIntermediateSize: 16, intermediateSize: 8, numLocalExperts: 2,
            numExpertsPerTok: 1, moeLayerFreq: [0])
    }

    @Test("q_norm/k_norm are per-head RMSNorms over headDim, not the full projected width")
    func qkNormIsPerHeadNotFlat() {
        let config = tinyAttentionConfig()
        let attention = MiniMaxM3Attention(config)
        eval(attention)

        // Per-head layout: weight shape is [headDim] (8), never
        // [heads * headDim] (32) -- the M2 flat-norm shape this must NOT match.
        #expect(attention.qNorm?.weight.shape == [config.headDim])
        #expect(attention.kNorm?.weight.shape == [config.headDim])
        #expect(attention.qNorm?.weight.shape != [config.attentionHeads * config.headDim])
    }

    @Test("a non-uniform q_norm weight applies identically across heads")
    func nonUniformQkNormAppliesIdenticallyAcrossHeads() throws {
        let config = tinyAttentionConfig()
        let attention = MiniMaxM3Attention(config)

        // A distinctly non-uniform per-headDim weight.
        let nonUniformWeight = MLXArray((0 ..< config.headDim).map { Float($0) })
        try attention.update(
            parameters: ModuleParameters.unflattened([
                "q_norm.weight": nonUniformWeight
            ]), verify: [])
        eval(attention)

        // Two heads, identical per-head content: reshaping (B=1, L=1,
        // heads=4, headDim=8) from a flat vector that repeats the same
        // 8 values across all 4 heads.
        let perHead = (0 ..< config.headDim).map { Float($0 + 1) }
        let flatInput = MLXArray((0 ..< config.attentionHeads).flatMap { _ in perHead })
        let reshaped = flatInput.reshaped(1, 1, config.attentionHeads, config.headDim)

        let normed = attention.qNorm!(reshaped)
        eval(normed)

        let head0 = normed[0, 0, 0, 0...]
        let head1 = normed[0, 0, 1, 0...]
        eval(head0, head1)
        #expect(MLX.abs(head0 - head1).max().item(Float.self) < 1e-6)
    }

    // MARK: - Tiny-model forward pass / KV cache / determinism

    private func tinyModelConfig(hiddenLayers: Int = 4, moeLayerFreq: [Int] = [0, 0, 1, 1])
        -> MiniMaxM3TextConfiguration
    {
        MiniMaxM3TextConfiguration(
            hiddenSize: 32, hiddenLayers: hiddenLayers, attentionHeads: 4, kvHeads: 2, headDim: 16,
            vocabularySize: 48, maxPositionEmbeddings: 256, rotaryDim: 8,
            denseIntermediateSize: 64, intermediateSize: 16, numLocalExperts: 4,
            numExpertsPerTok: 2, moeLayerFreq: moeLayerFreq)
    }

    @Test("tiny 2-dense + 2-MoE model forward pass produces (B, L, vocab) logits")
    func tinyModelForwardProducesExpectedShape() {
        MLXRandom.seed(0)
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let tokens = MLXArray(Array(0 ..< 6).map { Int32($0) }).reshaped(1, 6)
        let logits = model(tokens, cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 6, config.vocabularySize])
    }

    @Test("forward pass is deterministic under a fixed MLXRandom.seed")
    func forwardPassIsDeterministicUnderFixedSeed() {
        let tokens = MLXArray(Array(0 ..< 5).map { Int32($0) }).reshaped(1, 5)

        MLXRandom.seed(42)
        let modelA = MiniMaxM3Model(tinyModelConfig())
        let outA = modelA(tokens, cache: nil)
        eval(outA)

        MLXRandom.seed(42)
        let modelB = MiniMaxM3Model(tinyModelConfig())
        let outB = modelB(tokens, cache: nil)
        eval(outB)

        #expect(allClose(outA, outB, atol: 1e-6).item(Bool.self))
    }

    @Test("incremental decoding with a KV cache matches full-context forward")
    func incrementalDecodingMatchesFullContextForward() {
        MLXRandom.seed(3)
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let tokens = MLXArray(Array(0 ..< 5).map { Int32($0) }).reshaped(1, 5)

        let fullCache = model.newCache(parameters: nil)
        let fullLogits = model(tokens, cache: fullCache)
        eval(fullLogits)

        let incrementalCache = model.newCache(parameters: nil)
        var lastLogits: MLXArray = MLXArray(0)
        for i in 0 ..< 5 {
            let token = tokens[0..., i ..< (i + 1)]
            lastLogits = model(token, cache: incrementalCache)
        }
        eval(lastLogits)

        let match = allClose(
            fullLogits[0..., 4 ..< 5, 0...], lastLogits, atol: 1e-4)
        eval(match)
        #expect(match.item(Bool.self))
    }

    @Test("newCache returns one KVCacheSimple per decoder layer")
    func newCacheReturnsOneSimpleCachePerLayer() {
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == config.hiddenLayers)
        #expect(cache.allSatisfy { $0 is KVCacheSimple })
    }

    @Test("newCache returns 60 KVCacheSimple entries for the real 60-layer schedule")
    func newCacheReturns60EntriesForRealSchedule() {
        let realSchedule = [0, 0, 0] + Array(repeating: 1, count: 57)
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 8, hiddenLayers: 60, attentionHeads: 2, kvHeads: 1, headDim: 4,
            vocabularySize: 16, denseIntermediateSize: 8, intermediateSize: 4,
            numLocalExperts: 2, numExpertsPerTok: 1, moeLayerFreq: realSchedule)
        let model = MiniMaxM3Model(config)

        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 60)
        #expect(cache.allSatisfy { $0 is KVCacheSimple })
    }

    // MARK: - sanitize(weights:)

    /// The full set of parameter paths a real `MiniMaxM3Model` expects,
    /// derived from the model's own (already `language_model.`-prefixed)
    /// parameter tree -- avoids hand-enumerating every shape.
    private func expectedParameterKeys(for model: MiniMaxM3Model) -> Set<String> {
        Set(model.parameters().flattened().map(\.0))
    }

    @Test("sanitize drops vision/multi-modal/MTP/index weights and re-keys flat checkpoints")
    func sanitizeDropsExtraneousWeightsAndProducesExactParameterSet() throws {
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let expectedKeys = expectedParameterKeys(for: model)

        // Simulate a flat (non-`language_model.`-prefixed) checkpoint by
        // stripping the prefix from the model's own real parameters, then add
        // extraneous vision/MTP/sparse-indexer keys that must be dropped.
        var rawWeights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            precondition(key.hasPrefix("language_model."))
            let flatKey = String(key.dropFirst("language_model.".count))
            rawWeights[flatKey] = value
        }

        rawWeights["vision_tower.vision_model.embeddings.patch_embedding.weight"] = MLXArray.zeros(
            [4, 4])
        rawWeights["multi_modal_projector.linear_1.weight"] = MLXArray.zeros([4, 4])
        rawWeights["patch_merge_mlp.linear_1.weight"] = MLXArray.zeros([4, 4])
        rawWeights["mtp.0.embed_tokens.weight"] = MLXArray.zeros([4, 4])
        rawWeights["model.layers.3.self_attn.index_q_proj.weight"] = MLXArray.zeros([4, 4])
        rawWeights["model.layers.3.self_attn.index_k_proj.weight"] = MLXArray.zeros([4, 4])
        rawWeights["model.layers.3.self_attn.index_q_norm.weight"] = MLXArray.zeros([4])
        rawWeights["model.layers.3.self_attn.index_k_norm.weight"] = MLXArray.zeros([4])

        let sanitized = model.sanitize(weights: rawWeights)

        #expect(Set(sanitized.keys) == expectedKeys)

        // The strongest proof: the sanitized weights actually load.
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    @Test("sanitize passes fused, pre-stacked expert weights through unchanged")
    func sanitizePassesFusedExpertWeightsThrough() throws {
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let expectedKeys = expectedParameterKeys(for: model)
        let alreadyPrefixed = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened())

        let sanitized = model.sanitize(weights: alreadyPrefixed)

        #expect(Set(sanitized.keys) == expectedKeys)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    @Test("sanitize remaps per-expert w1/w2/w3 fallback weights into the fused switch_mlp layout")
    func sanitizeRemapsPerExpertFallbackWeights() {
        // A single MoE layer, tiny dims, to keep the synthetic per-expert
        // weight dict small. `hiddenSize` and `intermediateSize` are
        // deliberately unequal (6 vs. 4) so the shape assertion below can
        // actually distinguish a correct [numLocalExperts, 2*intermediateSize,
        // hiddenSize] layout from an accidentally-transposed one -- with
        // equal dims the two shapes would be indistinguishable.
        let hiddenSize = 6
        let intermediateSize = 4
        let numLocalExperts = 2
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: hiddenSize, hiddenLayers: 1, attentionHeads: 1, kvHeads: 1, headDim: 4,
            vocabularySize: 8, denseIntermediateSize: 4, intermediateSize: intermediateSize,
            numLocalExperts: numLocalExperts, numExpertsPerTok: 1, moeLayerFreq: [1])
        let model = MiniMaxM3Model(config)

        var raw: [String: MLXArray] = [:]
        for expert in 0 ..< numLocalExperts {
            let prefix = "language_model.model.layers.0.block_sparse_moe.experts.\(expert)"
            raw["\(prefix).w1.weight"] = MLXArray.full(
                [intermediateSize, hiddenSize], values: MLXArray(Float(expert + 1)))
            raw["\(prefix).w3.weight"] = MLXArray.full(
                [intermediateSize, hiddenSize], values: MLXArray(Float(expert + 10)))
            raw["\(prefix).w2.weight"] = MLXArray.full(
                [hiddenSize, intermediateSize], values: MLXArray(Float(expert + 100)))
        }

        let sanitized = model.sanitize(weights: raw)

        let gateUpKey = "language_model.model.layers.0.block_sparse_moe.switch_mlp.gate_up_proj.weight"
        let downKey = "language_model.model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"

        #expect(sanitized[gateUpKey] != nil)
        #expect(sanitized[downKey] != nil)
        // No leftover per-expert keys.
        #expect(!sanitized.keys.contains(where: { $0.contains(".experts.") }))

        let gateUp = sanitized[gateUpKey]!
        eval(gateUp)
        #expect(gateUp.shape == [numLocalExperts, 2 * intermediateSize, hiddenSize])

        // gate half (rows 0..<intermediateSize) comes from w1, up half
        // (rows intermediateSize..<2*intermediateSize) comes from w3.
        for expert in 0 ..< numLocalExperts {
            let gateRow = gateUp[expert, 0, 0].item(Float.self)
            let upRow = gateUp[expert, intermediateSize, 0].item(Float.self)
            #expect(abs(gateRow - Float(expert + 1)) < 1e-6)
            #expect(abs(upRow - Float(expert + 10)) < 1e-6)
        }
    }

    @Test("the language_model. prefix is preserved so per-module quantization overrides resolve")
    func languageModelPrefixPreservesPerModuleQuantizationOverrides() {
        // Mirrors the real config.json's per-layer override: layer 3's MoE
        // gate ships at 8-bit while the checkpoint default is 4-bit.
        let overridePath = "language_model.model.layers.3.block_sparse_moe.gate"
        let perLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                overridePath: .quantize(.init(groupSize: 64, bits: 8))
            ]
        )

        #expect(perLayerQuantization.quantization(layer: overridePath)?.bits == 8)

        // Pin the invariant this depends on: the model's own module tree
        // really does expose that exact path.
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)
        let keys = Set(model.parameters().flattened().map(\.0))
        #expect(keys.contains("\(overridePath).weight"))
    }
}
