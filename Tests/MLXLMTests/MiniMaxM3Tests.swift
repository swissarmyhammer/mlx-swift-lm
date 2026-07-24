// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Tests for MiniMax-M3's swigluoai activation, sparse MoE block (kanban
/// ^mv9aq7w), the full `MiniMaxM3TextConfiguration` + decoder/model/sanitize
/// (kanban ^xgvth41), and MiniMax Sparse Attention (MSA) -- the
/// `MiniMaxM3Indexer` block selector and `MiniMaxM3KVCache` sparse-aware
/// cache (kanban ^8dbc476). Layers 0..<3 always use dense
/// `MiniMaxM3Attention`; layers 3..<60 (`config.isMoELayer`) build an
/// indexer and run block-sparse attention once the KV history exceeds
/// `topkBlocks * blockSize` tokens (2048 for the verified default config),
/// falling back to dense attention below that -- numerically identical,
/// which doubles as the equivalence regression anchor below.
///
/// Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py`
/// (`MiniMaxSwiGLUOAI`, `MiniMaxPackedSwitchGLU`, `MiniMaxSparseMoeBlock`,
/// `_minimax_moe_select`, `MiniMaxAttention`, `MiniMaxM3Indexer`,
/// `MiniMaxM3KVCache`), cross-checked against upstream mlx-lm PRs
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

    // MARK: - MSA sparse attention (indexer, dense fallback, sparse KV cache)

    @Test(
        "sparse-capable attention (dense fallback, KV length <= topkBlocks * blockSize) matches dense-only attention within 1e-5"
    )
    func sparseAttentionDenseFallbackMatchesDenseAttention() throws {
        let config = tinyAttentionConfig(hiddenSize: 16, attentionHeads: 4, kvHeads: 2, headDim: 8)

        MLXRandom.seed(11)
        // Dense-only (layer 0, config.moeLayerFreq == [0]) vs. sparse-capable
        // (layer index forced into the MoE range) attention built from the
        // same config -- the indexer's own weights are randomly initialized
        // and irrelevant here since the sequence below is far shorter than
        // the default dense-fallback threshold (2048 tokens).
        let denseAttention = MiniMaxM3Attention(config, layerIndex: 0)
        let sparseConfig = MiniMaxM3TextConfiguration(
            hiddenSize: config.hiddenSize, hiddenLayers: 2, attentionHeads: config.attentionHeads,
            kvHeads: config.kvHeads, headDim: config.headDim, vocabularySize: 32,
            denseIntermediateSize: 16, intermediateSize: 8, numLocalExperts: 2,
            numExpertsPerTok: 1, moeLayerFreq: [0, 1])
        let sparseAttention = MiniMaxM3Attention(sparseConfig, layerIndex: 1)
        eval(denseAttention, sparseAttention)
        #expect(sparseAttention.indexer != nil)

        // Copy the shared (non-indexer) weights from the dense instance into
        // the sparse one so both compute attention from identical Q/K/V/O/
        // norm weights -- only the code path differs.
        let denseParams = denseAttention.parameters().flattened()
        let sharedKeys: Set<String> = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "q_norm.weight", "k_norm.weight",
        ]
        let shared = denseParams.filter { sharedKeys.contains($0.0) }
        try sparseAttention.update(parameters: ModuleParameters.unflattened(shared), verify: [])
        eval(sparseAttention)

        let tokens = MLXRandom.normal([1, 8, config.hiddenSize])
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.causal

        let denseOutput = denseAttention(tokens, mask: mask, cache: nil)
        let sparseOutput = sparseAttention(tokens, mask: mask, cache: nil)
        eval(denseOutput, sparseOutput)

        let maxDiff = MLX.abs(denseOutput - sparseOutput).max().item(Float.self)
        #expect(maxDiff <= 1e-5)
    }

    @Test("the indexer selects the expected blocks for a hand-constructed input (shrunk block=4, topk=2)")
    func indexerSelectsExpectedBlocksForHandConstructedInput() throws {
        // indexDim/numIndexHeads/hiddenSize/headDim all shrunk to 4, with
        // rotaryDim: 2 -- only the first 2 of indexDim's 4 dims rotate, so
        // placing every hand-constructed vector's only nonzero component in
        // dim 2 or 3 (below) passes through RoPE completely unchanged
        // (exactly, not approximately), keeping the selection math below
        // hand-computable regardless of position. Every vector also has
        // equal norm, so Gemma-mode RMSNorm's per-token scalar rescale
        // doesn't disturb cross-key score ranking either.
        let sparseConfig = MiniMaxM3SparseAttentionConfiguration(
            indexDim: 4, numIndexHeads: 1, topkBlocks: 2, blockSize: 4, initBlocks: 0,
            localBlocks: 1, scoreType: "max")
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 4, hiddenLayers: 1, attentionHeads: 1, kvHeads: 1, headDim: 4,
            vocabularySize: 8, rotaryDim: 2, denseIntermediateSize: 2, intermediateSize: 2,
            numLocalExperts: 2, numExpertsPerTok: 1, moeLayerFreq: [1],
            sparseAttention: sparseConfig)
        let attention = MiniMaxM3Attention(config, layerIndex: 0)

        // Identity index_q_proj / index_k_proj so idx_q == x and idx_k ==
        // x exactly (pre-norm).
        let identity = MLXArray(
            (0 ..< 4).flatMap { row in (0 ..< 4).map { col in Float(row == col ? 1 : 0) } }, [4, 4])
        try attention.update(
            parameters: ModuleParameters.unflattened([
                "index_q_proj.weight": identity,
                "index_k_proj.weight": identity,
            ]), verify: [])
        eval(attention)

        guard let indexer = attention.indexer else {
            Issue.record("expected a sparse indexer for a MoE-scheduled layer")
            return
        }

        // 3 blocks of 4 tokens each (block size 4): block 0 is orthogonal
        // filler (dim 2) -- unrelated to the query; block 1 is aligned with
        // the query (dim 3) -- the expected non-forced winner; block 2 is
        // the query's own (always force-selected via localBlocks=1) block,
        // ending with the query token itself, dim 3, at position 11.
        let a: [Float] = [0, 0, 0, 1]
        let b: [Float] = [0, 0, 1, 0]
        var rows: [[Float]] = Array(repeating: b, count: 4)  // block 0
        rows += Array(repeating: a, count: 4)  // block 1
        rows += Array(repeating: b, count: 3) + [a]  // block 2 (query last)
        let x = MLXArray(rows.flatMap { $0 }, [1, 12, 4])

        guard let (blockIndices, totalLen) = indexer(x, offset: nil, cache: nil, qStart: 0) else {
            Issue.record("expected block selection to run, not the dense fallback")
            return
        }
        eval(blockIndices)

        #expect(totalLen == 12)
        let lastQuerySelection = blockIndices[0, 11, 0...].asArray(Int32.self).sorted()
        #expect(lastQuerySelection == [1, 2])
    }

    @Test("the indexer selects the expected blocks with scoreType \"lse\", including the all-masked-block NaN guard")
    func indexerSelectsExpectedBlocksWithLogSumExpScoreType() throws {
        // Same hand-constructed setup as the "max" score-type test above,
        // but with scoreType: "lse" -- exercises the untested logSumExp
        // intra-block aggregation path, including its NaN guard: querying
        // from position 0 (block 0's first token) leaves blocks 1 and 2
        // entirely non-causal (every token-level score is -inf before the
        // block reshape), so `logSumExp` reduces an all-`-inf` row, which
        // would produce NaN without the guard that follows it.
        let sparseConfig = MiniMaxM3SparseAttentionConfiguration(
            indexDim: 4, numIndexHeads: 1, topkBlocks: 2, blockSize: 4, initBlocks: 0,
            localBlocks: 1, scoreType: "lse")
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 4, hiddenLayers: 1, attentionHeads: 1, kvHeads: 1, headDim: 4,
            vocabularySize: 8, rotaryDim: 2, denseIntermediateSize: 2, intermediateSize: 2,
            numLocalExperts: 2, numExpertsPerTok: 1, moeLayerFreq: [1],
            sparseAttention: sparseConfig)
        let attention = MiniMaxM3Attention(config, layerIndex: 0)

        let identity = MLXArray(
            (0 ..< 4).flatMap { row in (0 ..< 4).map { col in Float(row == col ? 1 : 0) } }, [4, 4])
        try attention.update(
            parameters: ModuleParameters.unflattened([
                "index_q_proj.weight": identity,
                "index_k_proj.weight": identity,
            ]), verify: [])
        eval(attention)

        guard let indexer = attention.indexer else {
            Issue.record("expected a sparse indexer for a MoE-scheduled layer")
            return
        }

        let a: [Float] = [0, 0, 0, 1]
        let b: [Float] = [0, 0, 1, 0]
        var rows: [[Float]] = Array(repeating: b, count: 4)  // block 0
        rows += Array(repeating: a, count: 4)  // block 1
        rows += Array(repeating: b, count: 3) + [a]  // block 2 (query last)
        let x = MLXArray(rows.flatMap { $0 }, [1, 12, 4])

        guard let (blockIndices, totalLen) = indexer(x, offset: nil, cache: nil, qStart: 0) else {
            Issue.record("expected block selection to run, not the dense fallback")
            return
        }
        eval(blockIndices)

        #expect(totalLen == 12)

        // Last query (position 11): same expected winner as the "max"
        // variant -- block 1's logsumexp over 4 equal high-similarity
        // tokens is still comfortably larger than block 0's over 4 equal
        // near-zero-similarity tokens, and block 2 is forced via localBlocks.
        let lastQuerySelection = blockIndices[0, 11, 0...].asArray(Int32.self).sorted()
        #expect(lastQuerySelection == [1, 2])

        // First query (position 0): only block 0 is causally valid; blocks 1
        // and 2 are entirely masked out (all `-inf` before the intra-block
        // logsumexp), which is exactly the NaN-guard's target case. No crash
        // and no NaN in the result is itself the assertion here, plus block 0
        // must still be selected (the one real, non-sentinel entry).
        let firstQuerySelection = blockIndices[0, 0, 0...].asArray(Int32.self)
        #expect(firstQuerySelection.contains(0))
        #expect(firstQuerySelection.allSatisfy { $0 == 0 || $0 == -1 })
    }

    @Test("generation runs incrementally through MiniMaxM3KVCache without shape errors (shrunk block=2, topk=2)")
    func incrementalDecodeThroughSparseCacheProducesExpectedShapes() {
        let sparseConfig = MiniMaxM3SparseAttentionConfiguration(
            indexDim: 4, numIndexHeads: 2, topkBlocks: 2, blockSize: 2)
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 16, hiddenLayers: 2, attentionHeads: 4, kvHeads: 2, headDim: 8,
            vocabularySize: 32, denseIntermediateSize: 16, intermediateSize: 8,
            numLocalExperts: 2, numExpertsPerTok: 1, moeLayerFreq: [0, 1],
            sparseAttention: sparseConfig)

        MLXRandom.seed(21)
        let model = MiniMaxM3Model(config)
        eval(model)

        let cache = model.newCache(parameters: nil)
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is MiniMaxM3KVCache)

        // Prefill 5 tokens (threshold is topkBlocks * blockSize == 4, so
        // this prefill alone already crosses into the sparse-selection
        // path for the last couple of tokens), then decode 5 more tokens
        // one at a time -- each incremental step grows the sparse cache's
        // index-key history further past the threshold.
        let tokens = MLXArray(Array(0 ..< 10).map { Int32($0) }).reshaped(1, 10)

        let prefillLogits = model(tokens[0..., 0 ..< 5], cache: cache)
        eval(prefillLogits)
        #expect(prefillLogits.shape == [1, 5, config.vocabularySize])

        var lastLogits = prefillLogits
        for i in 5 ..< 10 {
            let token = tokens[0..., i ..< (i + 1)]
            lastLogits = model(token, cache: cache)
            eval(lastLogits)
            #expect(lastLogits.shape == [1, 1, config.vocabularySize])
        }
    }

    @Test("MiniMaxM3KVCache.isTrimmable is false")
    func miniMaxM3KVCacheIsNotTrimmable() {
        let cache = MiniMaxM3KVCache()
        #expect(cache.isTrimmable == false)
        #expect(cache.trim(5) == 0)
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

    /// A tiny synthetic vision configuration -- small enough for fast unit
    /// tests, but with a `hiddenSize`/`numAttentionHeads` ratio (`headDim` 4)
    /// that still exercises the same 3D-RoPE partial-rotation arithmetic as
    /// the verified real config (`headDim` 80 -> `axisDim` 26, `rotaryDim` 78).
    private func tinyVisionConfig() -> MiniMaxM3VisionConfiguration {
        MiniMaxM3VisionConfiguration(
            hiddenSize: 8, intermediateSize: 16, numAttentionHeads: 2, numHiddenLayers: 2,
            patchSize: 2, numChannels: 3)
    }

    /// A tiny synthetic vision-capable top-level configuration, pairing
    /// `tinyModelConfig()`'s text configuration with `tinyVisionConfig()`.
    private func tinyVisionModelConfig(hiddenLayers: Int = 4, moeLayerFreq: [Int] = [0, 0, 1, 1])
        -> MiniMaxM3Configuration
    {
        MiniMaxM3Configuration(
            textConfiguration: tinyModelConfig(hiddenLayers: hiddenLayers, moeLayerFreq: moeLayerFreq),
            visionConfiguration: tinyVisionConfig())
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

    @Test("newCache returns KVCacheSimple for dense layers, MiniMaxM3KVCache for MoE/sparse layers")
    func newCacheReturnsOneSimpleCachePerLayer() {
        let config = tinyModelConfig()  // moeLayerFreq: [0, 0, 1, 1]
        let model = MiniMaxM3Model(config)
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == config.hiddenLayers)
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[1] is KVCacheSimple)
        #expect(cache[2] is MiniMaxM3KVCache)
        #expect(cache[3] is MiniMaxM3KVCache)
    }

    @Test("newCache returns KVCacheSimple for layers 0..<3, MiniMaxM3KVCache for 3..<60 in the real schedule")
    func newCacheReturns60EntriesForRealSchedule() {
        let realSchedule = [0, 0, 0] + Array(repeating: 1, count: 57)
        let config = MiniMaxM3TextConfiguration(
            hiddenSize: 8, hiddenLayers: 60, attentionHeads: 2, kvHeads: 1, headDim: 4,
            vocabularySize: 16, denseIntermediateSize: 8, intermediateSize: 4,
            numLocalExperts: 2, numExpertsPerTok: 1, moeLayerFreq: realSchedule)
        let model = MiniMaxM3Model(config)

        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 60)
        #expect(cache[0..<3].allSatisfy { $0 is KVCacheSimple })
        #expect(cache[3...].allSatisfy { $0 is MiniMaxM3KVCache })
    }

    // MARK: - MiniMaxM3Vision (vision tower, 3D RoPE)

    @Test("tiny vision tower forward pass produces (tokens, hiddenSize), projector produces (tokens, projectionDim)")
    func visionTowerForwardProducesExpectedShape() {
        MLXRandom.seed(0)
        let visionConfig = tinyVisionConfig()
        let projectionDim = 12

        let visionTower = MiniMaxM3Vision.VisionModel(visionConfig)
        let projector = MiniMaxM3Projector(
            inputDimensions: visionConfig.hiddenSize, hiddenDimensions: 10,
            outputDimensions: projectionDim, bias: true, hiddenAct: "gelu")
        eval(visionTower, projector)

        // A single 4x4-patch image (spatialMergeSize 2 divides both dims).
        let grid = THW(1, 4, 4)
        let patchDimensions =
            visionConfig.numChannels * visionConfig.temporalPatchSize * visionConfig.patchSize
            * visionConfig.patchSize
        let pixelValues = MLXRandom.normal([grid.product, patchDimensions])

        let hidden = visionTower(pixelValues, gridTHW: [grid])
        eval(hidden)
        #expect(hidden.shape == [grid.product, visionConfig.hiddenSize])

        let projected = projector(hidden)
        eval(projected)
        #expect(projected.shape == [grid.product, projectionDim])
    }

    @Test("3D RoPE frequencies match a hand-built (t, h, w) position grid")
    func threeDRoPEMatchesHandBuiltGrid() {
        // hiddenSize 6 / 1 head -> headDim 6 -> rotaryDims 6 -> axisDim 2,
        // chosen so every head dimension rotates (no `x_pass` remainder) and
        // each axis contributes exactly one frequency: with `dim == 2`,
        // `axisFrequencies`'s inverse frequency is `theta^0 == 1`, so the
        // frequency table reduces to the raw (t, h, w) coordinates
        // themselves -- hand-computable without reimplementing the formula.
        let config = MiniMaxM3VisionConfiguration(
            hiddenSize: 6, intermediateSize: 8, numAttentionHeads: 1, numHiddenLayers: 1,
            patchSize: 2, numChannels: 3, spatialMergeSize: 1)
        let grid = THW(1, 2, 2)

        let (cosTable, sinTable) = MiniMaxM3Vision.rotaryPositionEmbedding(
            grids: [grid], config: config)
        eval(cosTable, sinTable)

        // Token order for a 1x2x2 grid with spatialMergeSize 1 matches
        // `QwenVL.patchify`'s (t, h/merge, w/merge, mergeRow, mergeCol)
        // ordering, row-major: (0,0,0), (0,0,1), (0,1,0), (0,1,1).
        let coordinates: [[Float]] = [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        ]
        let expected = MLXArray(coordinates.flatMap { $0 + $0 }, [4, 6])

        #expect(cosTable.shape == [4, 6])
        #expect(sinTable.shape == [4, 6])
        let cosDiff = MLX.abs(cosTable - MLX.cos(expected)).max().item(Float.self)
        let sinDiff = MLX.abs(sinTable - MLX.sin(expected)).max().item(Float.self)
        #expect(cosDiff < 1e-5)
        #expect(sinDiff < 1e-5)
    }

    @Test("a tiny vision-capable model's prepare() runs the image path end-to-end and returns logits")
    func prepareRunsImagePathEndToEnd() throws {
        MLXRandom.seed(0)
        let config = tinyVisionModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let visionConfig = config.visionConfiguration
        let grid = THW(1, 4, 4)
        let patchDimensions =
            visionConfig.numChannels * visionConfig.temporalPatchSize * visionConfig.patchSize
            * visionConfig.patchSize
        let pixelValues = MLXRandom.normal([grid.product, patchDimensions])
        let mergeSize = visionConfig.spatialMergeSize
        let numImageTokens = grid.product / (mergeSize * mergeSize)

        var tokens: [Int32] = [1]
        tokens.append(contentsOf: Array(repeating: Int32(config.imageTokenIndex), count: numImageTokens))
        tokens.append(2)
        let inputIds = MLXArray(tokens).reshaped(1, tokens.count)

        let input = LMInput(
            text: .init(tokens: inputIds),
            image: .init(pixels: pixelValues, frames: [grid]))

        let cache = model.newCache(parameters: nil)
        let result = try model.prepare(input, cache: cache, state: nil, windowSize: nil)

        guard case .logits(let output) = result else {
            Issue.record("expected .logits for an image prompt, got .tokens")
            return
        }
        eval(output.logits)
        #expect(output.logits.shape == [1, tokens.count, config.textConfiguration.vocabularySize])
    }

    // MARK: - sanitize(weights:)

    /// The full set of parameter paths a real `MiniMaxM3Model` expects,
    /// derived from the model's own (already `language_model.`-prefixed)
    /// parameter tree -- avoids hand-enumerating every shape.
    private func expectedParameterKeys(for model: MiniMaxM3Model) -> Set<String> {
        Set(model.parameters().flattened().map(\.0))
    }

    @Test("sanitize re-keys a flat text-only checkpoint and drops only MTP weights")
    func sanitizeDropsExtraneousWeightsAndProducesExactParameterSet() throws {
        let config = tinyModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let expectedKeys = expectedParameterKeys(for: model)

        // Simulate a flat (non-`language_model.`-prefixed) checkpoint by
        // stripping the prefix from the model's own real parameters --
        // includes the real self_attn.index_* weights for MoE-scheduled
        // layers, which sanitize must now load (not drop, since ^8dbc476
        // built the indexer that consumes them) -- then add an extraneous
        // MTP key that must still be dropped.
        var rawWeights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            precondition(key.hasPrefix("language_model."))
            let flatKey = String(key.dropFirst("language_model.".count))
            rawWeights[flatKey] = value
        }

        rawWeights["mtp.0.embed_tokens.weight"] = MLXArray.zeros([4, 4])

        let sanitized = model.sanitize(weights: rawWeights)

        #expect(Set(sanitized.keys) == expectedKeys)

        // The strongest proof: the sanitized weights actually load.
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    @Test(
        "sanitize keeps vision-tower/multi-modal-projector/patch-merge weights on a vision-capable model, with zero unconsumed keys"
    )
    func sanitizeKeepsVisionWeightsWithZeroUnconsumedKeys() throws {
        let config = tinyVisionModelConfig()
        let model = MiniMaxM3Model(config)
        eval(model)

        let expectedKeys = expectedParameterKeys(for: model)

        // Vision-tower/projector/patch-merge keys are already top-level
        // (unprefixed) in this model's own parameter tree -- only the
        // language-model keys carry the `language_model.` prefix that needs
        // stripping to simulate a flat-checkpoint-style raw weight dict.
        var rawWeights: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            if key.hasPrefix("language_model.") {
                rawWeights[String(key.dropFirst("language_model.".count))] = value
            } else {
                rawWeights[key] = value
            }
        }

        // Overwrite the patch-embedding weight with the checkpoint's real
        // Conv3d layout (5D: hidden, channels, temporalPatch, patch, patch)
        // instead of this model's own already-flattened 2D `Linear` weight,
        // to exercise `_remapVisionWeights`'s reshape step too.
        let visionConfig = config.visionConfiguration
        rawWeights["vision_tower.vision_model.embeddings.patch_embedding.weight"] =
            MLXArray.zeros([
                visionConfig.hiddenSize, visionConfig.numChannels, visionConfig.temporalPatchSize,
                visionConfig.patchSize, visionConfig.patchSize,
            ])

        rawWeights["mtp.0.embed_tokens.weight"] = MLXArray.zeros([4, 4])

        let sanitized = model.sanitize(weights: rawWeights)

        #expect(Set(sanitized.keys) == expectedKeys)
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

    // MARK: - VLMTypeRegistry / VLMProcessorTypeRegistry / VLMRegistry registration

    /// Tiny nested VL config (`minimax_m3_vl`, `text_config` wrapper) --
    /// mirrors `tinyModelConfig()`'s dimensions but as literal JSON, keeping
    /// registry-resolution tests fast (no full 60-layer/128-expert allocation).
    private static let tinyNestedVLConfigJSON = """
        {
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 48,
                "dense_intermediate_size": 64,
                "intermediate_size": 16,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "moe_layer_freq": [0, 1]
            }
        }
        """

    /// Tiny flat `minimax_m3` config (no VL nesting) -- same dimensions as
    /// `tinyNestedVLConfigJSON`'s `text_config`.
    private static let tinyFlatConfigJSON = """
        {
            "model_type": "minimax_m3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 48,
            "dense_intermediate_size": 64,
            "intermediate_size": 16,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_layer_freq": [0, 1]
        }
        """

    @Test("VLMTypeRegistry resolves minimax_m3_vl to MiniMaxM3Model")
    func vlmTypeRegistryResolvesNestedVLModelType() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(Self.tinyNestedVLConfigJSON.utf8), modelType: "minimax_m3_vl")
        #expect(model is MiniMaxM3Model)
    }

    @Test("VLMTypeRegistry resolves the flat minimax_m3 model type to the same MiniMaxM3Model")
    func vlmTypeRegistryResolvesFlatModelType() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(Self.tinyFlatConfigJSON.utf8), modelType: "minimax_m3")
        #expect(model is MiniMaxM3Model)
    }

    @Test("VLMProcessorTypeRegistry resolves MiniMaxM3VLProcessor to MiniMaxM3Processor")
    func vlmProcessorTypeRegistryResolvesMiniMaxM3Processor() async throws {
        let data = Data(
            """
            { "processor_class": "MiniMaxM3VLProcessor" }
            """.utf8)
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: data, processorType: "MiniMaxM3VLProcessor", tokenizer: TestTokenizer())
        #expect(processor is MiniMaxM3Processor)
    }

    @Test("VLMRegistry exposes a minimaxM34bit entry pointing at mlx-community/MiniMax-M3-4bit")
    func vlmRegistryHasMiniMaxM34bitEntry() {
        #expect(VLMRegistry.minimaxM34bit.name == "mlx-community/MiniMax-M3-4bit")
        #expect(VLMRegistry.all().contains { $0.name == VLMRegistry.minimaxM34bit.name })
    }

    // MARK: - loadProcessorConfig fallback (preprocessor_config.json missing processor_class)

    @Test(
        "loadProcessorConfig falls back to processor_config.json when preprocessor_config.json lacks processor_class"
    )
    func loadProcessorConfigFallsBackForMissingProcessorClass() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try
            """
            { "image_processor_type": "MiniMaxM3VLImageProcessor" }
            """
            .write(
                to: tempDir.appendingPathComponent("preprocessor_config.json"),
                atomically: true, encoding: .utf8)
        try
            """
            { "processor_class": "MiniMaxM3VLProcessor" }
            """
            .write(
                to: tempDir.appendingPathComponent("processor_config.json"),
                atomically: true, encoding: .utf8)

        let (_, config) = try await loadProcessorConfig(from: tempDir)
        #expect(config.processorClass == "MiniMaxM3VLProcessor")
    }

    // MARK: - MiniMaxM3Processor

    @Test("MiniMaxM3Processor tokenizes plain text prompts")
    func processorTokenizesTextPrompt() async throws {
        let processor = MiniMaxM3Processor(
            MiniMaxM3ProcessorConfiguration(), tokenizer: TestTokenizer())
        let input = try await processor.prepare(input: UserInput(prompt: "2+2="))
        #expect(input.text.tokens.size > 0)
        #expect(input.image == nil)
        #expect(input.video == nil)
    }

    /// Deterministic stand-in tokenizer for exercising MiniMax-M3's
    /// image-placeholder chat-template splicing without a real BPE tokenizer
    /// or Jinja renderer -- `TestTokenizer` can't be used here since its
    /// `encode(text:)`/`applyChatTemplate` are both random and
    /// content-independent, so a real placeholder-token occurrence could
    /// never be found in its output.
    ///
    /// `encode(text:)` maps each Unicode scalar to its integer value, so
    /// concatenated text always encodes to concatenated tokens (keeping
    /// placeholder subsequences reliably findable), and `applyChatTemplate`
    /// renders just enough of MiniMax-M3's real chat template
    /// (`processor_config.json`'s `chat_template`, `visible_text` macro) to
    /// turn `{"type": "image"}` content blocks (as produced by
    /// `MiniMaxM3MessageGenerator`) into the literal
    /// `MiniMaxM3Processor.imageToken` placeholder text.
    private struct MiniMaxM3FakeTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.unicodeScalars.map { Int($0.value) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(String.UnicodeScalarView(tokenIds.compactMap { Unicode.Scalar($0) }))
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var text = ""
            for message in messages {
                guard let content = message["content"] else { continue }
                if let string = content as? String {
                    text += string
                } else if let blocks = content as? [[String: String]] {
                    for block in blocks {
                        switch block["type"] {
                        case "text": text += block["text"] ?? ""
                        case "image": text += MiniMaxM3Processor.imageToken
                        default: break
                        }
                    }
                }
            }
            return encode(text: text, addSpecialTokens: false)
        }
    }

    @Test("MiniMaxM3Processor splices a synthetic image's tokens into the prompt")
    func processorSplicesImageTokens() async throws {
        let processor = MiniMaxM3Processor(
            MiniMaxM3ProcessorConfiguration(), tokenizer: MiniMaxM3FakeTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))

        let input = try await processor.prepare(
            input: UserInput(prompt: "describe", images: [.ciImage(image)]))

        guard let frames = input.image?.frames, let grid = frames.first else {
            Issue.record("expected image frames on the prepared LMInput")
            return
        }
        #expect(input.image?.pixels != nil)

        let mergeSize = MiniMaxM3ProcessorConfiguration().imageProcessor.mergeSize
        let expectedImageTokens = grid.product / (mergeSize * mergeSize)

        let promptTokens = input.text.tokens.asArray(Int32.self).map { Int($0) }
        let placeholderTokens = MiniMaxM3FakeTokenizer().encode(text: MiniMaxM3Processor.imageToken)
        let occurrences = promptTokens.ranges(of: placeholderTokens).count

        #expect(occurrences == expectedImageTokens)
    }

    @Test("MiniMaxM3Processor throws a descriptive error for image input when no chat template is configured")
    func processorThrowsForImageInputWithoutChatTemplate() async throws {
        // TestTokenizer's `applyChatTemplate` never throws
        // `TokenizerError.missingChatTemplate` (it just returns random
        // tokens), so this exercises the real "no template" failure via the
        // fake tokenizer's own default `Tokenizer` extension instead: a
        // tokenizer with no chat template configured can't render the image
        // placeholder markup, so the image path must propagate a clear
        // error rather than silently mis-tokenizing.
        struct NoTemplateTokenizer: MLXLMCommon.Tokenizer {
            func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? = nil
            var eosToken: String? = nil
            var unknownToken: String? = nil
            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                throw TokenizerError.missingChatTemplate
            }
        }

        let processor = MiniMaxM3Processor(
            MiniMaxM3ProcessorConfiguration(), tokenizer: NoTemplateTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))

        do {
            _ = try await processor.prepare(
                input: UserInput(prompt: "describe", images: [.ciImage(image)]))
            Issue.record("expected TokenizerError.missingChatTemplate to propagate")
        } catch {
            #expect(error is TokenizerError)
        }
    }

    @Test("MiniMaxM3Processor throws a descriptive error for video input instead of ignoring it")
    func processorThrowsForVideoInput() async throws {
        let processor = MiniMaxM3Processor(
            MiniMaxM3ProcessorConfiguration(), tokenizer: TestTokenizer())
        let video = UserInput.Video.url(URL(fileURLWithPath: "/nonexistent/video.mp4"))

        do {
            _ = try await processor.prepare(
                input: UserInput(prompt: "describe", videos: [video]))
            Issue.record("expected VLMError.mediaNotSupported to be thrown")
        } catch {
            #expect(error as? VLMError == VLMError.mediaNotSupported("video"))
        }
    }
}
