// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny model with a `MambaCache` for each recurrent layer.
///
/// Each of these models feeds its conv or SSM layer through a `MambaCache`. The executor prompt
/// cache keeps a cache only when the offset of each cache is the length of its token ledger, thus
/// the recurrent layer must move the offset of its `MambaCache` as an attention layer does.
///
/// BaichuanM1 and FalconH1 put a `MambaCache` and an attention cache in one `CacheList` for each
/// layer. The executor prompt cache reads the offset of the `CacheList`, thus the offset of the
/// list must also move.
enum HybridRecurrentModelFixture: String, CaseIterable, Sendable {
    case nemotronH
    case jamba
    case mamba2
    case graniteMoeHybrid
    case lfm2
    case lfm2MoE
    case lfm2VL
    case baichuanM1
    case baichuanM1ShortWindow
    case falconH1
    case qwen3Next
    case qwen35Text
    case qwen35
    case qwen35MoE
    case qwen35VL
    case qwen35MoEVL

    /// The seed of the initializer weights of each tiny model.
    private static let weightSeed: UInt64 = 43

    /// The number of decoder layers of each tiny Qwen 3.5 model.
    private static let qwen35LayerCount = 4

    /// One layer of each two is a full-attention layer, thus layers 0 and 2 of each tiny Qwen 3.5
    /// model are linear (recurrent) layers, and layers 1 and 3 are attention layers.
    private static let qwen35FullAttentionInterval = 2

    /// The number of experts of each MoE block of the tiny Qwen 3.5 MoE models.
    private static let qwen35ExpertCount = 4

    /// The number of experts that each token uses in the tiny Qwen 3.5 MoE models. The MoE block
    /// normalizes the scores of the chosen experts, and a sum over one expert stops the GPU
    /// reduction with an assertion, thus each token uses two experts.
    private static let qwen35ExpertsPerToken = 2

    /// The model type of the Qwen 3.5 checkpoints.
    private static let qwen35ModelType = "qwen3_5"

    /// Makes the tiny model with fixed initializer weights.
    ///
    /// - Returns: The model.
    /// - Throws: The error of the configuration decoder.
    func makeModel() throws -> any LanguageModel {
        try withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            try makeUnseededModel()
        }
    }

    /// Makes the tiny model with the current random state.
    ///
    /// - Returns: The model.
    /// - Throws: The error of the configuration decoder.
    private func makeUnseededModel() throws -> any LanguageModel {
        switch self {
        case .nemotronH:
            return NemotronHModel(try Self.decode(NemotronHConfiguration.self, Self.nemotronHJSON))
        case .jamba:
            return JambaModel(try Self.decode(JambaConfiguration.self, Self.jambaJSON))
        case .mamba2:
            return Mamba2Model(try Self.decode(Mamba2Configuration.self, Self.mamba2JSON))
        case .graniteMoeHybrid:
            return GraniteMoeHybridModel(
                try Self.decode(GraniteMoeHybridConfiguration.self, Self.graniteMoeHybridJSON))
        case .lfm2:
            return LFM2Model(try Self.decode(LFM2Configuration.self, Self.lfm2JSON))
        case .lfm2MoE:
            return LFM2MoEModel(try Self.decode(LFM2MoEConfiguration.self, Self.lfm2MoEJSON))
        case .lfm2VL:
            return LFM2VL(try Self.decode(LFM2VLConfiguration.self, Self.lfm2VLJSON))
        case .baichuanM1:
            return Self.makeBaichuanM1Model(
                try Self.decode(BaichuanM1Configuration.self, Self.baichuanM1JSON))
        case .baichuanM1ShortWindow:
            return Self.makeBaichuanM1Model(
                try Self.decode(BaichuanM1Configuration.self, Self.baichuanM1ShortWindowJSON))
        case .falconH1:
            return FalconH1Model(try Self.decode(FalconH1Configuration.self, Self.falconH1JSON))
        case .qwen3Next:
            return Qwen3NextModel(try Qwen3NextCompiledDecodeTests.configuration())
        case .qwen35Text:
            return MLXLLM.Qwen35TextModel(
                try Self.decode(MLXLLM.Qwen35TextConfiguration.self, Self.qwen35TextJSON()))
        case .qwen35:
            return MLXLLM.Qwen35Model(
                try Self.decode(MLXLLM.Qwen35Configuration.self, Self.qwen35WrappedJSON()))
        case .qwen35MoE:
            return MLXLLM.Qwen35MoEModel(
                try Self.decode(
                    MLXLLM.Qwen35Configuration.self,
                    Self.qwen35WrappedJSON(experts: Self.qwen35ExpertCount)))
        case .qwen35VL:
            return MLXVLM.Qwen35(
                try Self.decode(MLXVLM.Qwen35Configuration.self, Self.qwen35VLJSON()))
        case .qwen35MoEVL:
            return MLXVLM.Qwen35MoE(
                try Self.decode(
                    MLXVLM.Qwen35Configuration.self,
                    Self.qwen35VLJSON(experts: Self.qwen35ExpertCount)))
        }
    }

    /// The JSON of a tiny Qwen 3.5 text configuration with linear layers 0 and 2 and attention
    /// layers 1 and 3.
    ///
    /// - Parameter experts: The number of experts of each MoE block, or 0 for a dense model.
    /// - Returns: The JSON text.
    private static func qwen35TextJSON(experts: Int = 0) -> String {
        qwen35TextConfigJSON(
            mtpLayers: 0, numExperts: experts, expertsPerToken: qwen35ExpertsPerToken,
            hiddenLayers: qwen35LayerCount, fullAttentionInterval: qwen35FullAttentionInterval)
    }

    /// The JSON of a tiny `qwen3_5` text-only configuration, which wraps ``qwen35TextJSON(experts:)``.
    ///
    /// - Parameter experts: The number of experts of each MoE block, or 0 for a dense model.
    /// - Returns: The JSON text.
    private static func qwen35WrappedJSON(experts: Int = 0) -> String {
        qwen35WrappedTextConfigJSON(
            modelType: qwen35ModelType, textConfigJSON: qwen35TextJSON(experts: experts))
    }

    /// The JSON of a tiny `qwen3_5` vision-language configuration, which wraps
    /// ``qwen35TextJSON(experts:)`` and a tiny vision tower. Only the text runs.
    ///
    /// - Parameter experts: The number of experts of each MoE block, or 0 for a dense model.
    /// - Returns: The JSON text.
    private static func qwen35VLJSON(experts: Int = 0) -> String {
        qwen35VLMConfigJSON(textConfigJSON: qwen35TextJSON(experts: experts))
    }

    /// Makes a BaichuanM1 model whose short-convolution weights are not zero.
    ///
    /// The model starts each `conv_k` and `conv_v` weight with zeros, and a checkpoint then
    /// gives the real values. Zero weights make each key and each value zero, thus each
    /// attention output is zero, and no mask and no attention cache can change the logits. The
    /// tiny model thus gets random values from the current random state.
    ///
    /// - Parameter configuration: The configuration.
    /// - Returns: The model.
    private static func makeBaichuanM1Model(
        _ configuration: BaichuanM1Configuration
    ) -> BaichuanM1Model {
        let model = BaichuanM1Model(configuration)
        let convolutionWeights = model.parameters().flattened()
            .filter { key, _ in key.hasSuffix(".conv_k") || key.hasSuffix(".conv_v") }
            .map { key, weight in (key, MLXRandom.normal(weight.shape)) }
        model.update(parameters: ModuleParameters.unflattened(convolutionWeights))
        return model
    }

    /// Decodes a configuration from JSON text.
    ///
    /// - Parameters:
    ///   - type: The type of the configuration.
    ///   - json: The JSON text.
    /// - Returns: The configuration.
    /// - Throws: The error of the decoder.
    private static func decode<Configuration: Decodable>(
        _ type: Configuration.Type, _ json: String
    ) throws -> Configuration {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    // MARK: - Configurations

    /// Mamba, attention, Mamba and MLP layers. Only the Mamba and attention layers have a cache.
    private static let nemotronHJSON = """
        {
            "vocab_size": 32, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 16,
            "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
            "moe_intermediate_size": 64, "moe_shared_expert_intermediate_size": 64,
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "hybrid_override_pattern": "M*M-", "layer_norm_epsilon": 1e-5,
            "n_group": 2, "topk_group": 1
        }
        """

    /// Layers 1 and 3 are attention layers, and layers 0 and 2 are Mamba layers.
    private static let jambaJSON = """
        {
            "model_type": "jamba", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "attn_layer_offset": 1, "attn_layer_period": 2,
            "expert_layer_offset": 1, "expert_layer_period": 2,
            "mamba_d_conv": 4, "mamba_d_state": 8, "mamba_expand": 2,
            "num_experts": 4, "num_experts_per_tok": 2,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "vocab_size": 32,
            "tie_word_embeddings": true
        }
        """

    /// Two Mamba2 layers and no attention layer.
    private static let mamba2JSON = """
        {
            "model_type": "mamba2", "num_heads": 4, "head_dim": 4, "vocab_size": 32,
            "hidden_size": 16, "state_size": 8, "num_hidden_layers": 2,
            "layer_norm_epsilon": 1e-5, "conv_kernel": 4, "n_groups": 1,
            "use_bias": false, "use_conv_bias": true, "tie_word_embeddings": false,
            "time_step_limit": [0.0, 100.0]
        }
        """

    /// Mamba2 layers 0 and 2, and attention layers 1 and 3.
    private static let graniteMoeHybridJSON = """
        {
            "model_type": "granitemoehybrid", "vocab_size": 32, "hidden_size": 8,
            "intermediate_size": 16, "num_hidden_layers": 4,
            "max_position_embeddings": 128, "num_attention_heads": 2,
            "num_key_value_heads": 1, "attention_bias": false,
            "embedding_multiplier": 1.0, "attention_multiplier": 1.0,
            "logits_scaling": 1.0, "residual_multiplier": 1.0,
            "layer_types": ["mamba", "attention", "mamba", "attention"],
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "num_local_experts": 4, "num_experts_per_tok": 2,
            "shared_intermediate_size": 8,
            "mamba_n_heads": 2, "mamba_d_head": 4, "mamba_d_state": 8,
            "mamba_d_conv": 4, "mamba_n_groups": 1,
            "mlp_bias": false, "position_embedding_type": "rope",
            "tie_word_embeddings": true
        }
        """

    /// Short-convolution layers 0 and 2, and attention layers 1 and 3.
    private static let lfm2JSON = """
        {
            "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "max_position_embeddings": 128, "norm_eps": 1e-5, "conv_L_cache": 3,
            "block_dim": 16, "block_ff_dim": 32, "block_multiple_of": 8,
            "full_attn_idxs": [1, 3], "rope_theta": 10000.0
        }
        """

    /// Short-convolution layers 0 and 2, and attention layers 1 and 3, each with an MoE block.
    private static let lfm2MoEJSON = """
        {
            "model_type": "lfm2_moe", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 32, "moe_intermediate_size": 16, "num_hidden_layers": 4,
            "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": true,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "max_position_embeddings": 128, "use_expert_bias": false,
            "num_dense_layers": 0, "norm_eps": 1e-5, "conv_bias": false,
            "conv_L_cache": 3, "full_attn_idxs": [1, 3], "rope_theta": 10000.0
        }
        """

    /// The LFM2 text layout of `lfm2JSON` with a tiny vision tower. Only the text runs.
    private static let lfm2VLJSON = """
        {
            "model_type": "lfm2-vl",
            "text_config": {
                "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
                "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
                "norm_eps": 1e-5, "conv_L_cache": 3,
                "block_dim": 16, "block_ff_dim": 32, "block_multiple_of": 8,
                "full_attn_idxs": [1, 3], "rope_theta": 10000.0
            },
            "vision_config": {
                "model_type": "siglip2_vision_model", "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                "image_size": 8, "patch_size": 4, "num_patches": 4
            },
            "projector_hidden_size": 16
        }
        """

    /// Four layers, each with a short-convolution `MambaCache` and an attention cache in one
    /// `CacheList`. Layers 0 and 2 are sliding-window layers. The window is longer than each
    /// prompt of the tests, thus the rotating caches keep each token.
    private static let baichuanM1JSON = """
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "rope_theta": 10000.0, "sliding_window": 64, "sliding_window_layers": [0, 2],
            "conv_window": 2, "rms_norm_eps": 1e-6, "tie_word_embeddings": true
        }
        """

    /// The BaichuanM1 layout of `baichuanM1JSON` with a window of 4 tokens. Each prompt of the
    /// tests is longer than the window, thus the sliding-window layers 0 and 2 must attend only
    /// to the last 4 tokens in a prefill as in a decode.
    private static let baichuanM1ShortWindowJSON = """
        {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
            "rope_theta": 10000.0, "sliding_window": 4, "sliding_window_layers": [0, 2],
            "conv_window": 2, "rms_norm_eps": 1e-6, "tie_word_embeddings": true
        }
        """

    /// Two layers, each with a Mamba2 `MambaCache` and an attention cache in one `CacheList`.
    private static let falconH1JSON = """
        {
            "model_type": "falcon_h1", "hidden_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 32, "mamba_d_ssm": 16, "mamba_d_state": 8, "mamba_d_head": 8,
            "mamba_n_heads": 2, "mamba_n_groups": 1, "mamba_d_conv": 4,
            "mamba_chunk_size": 64
        }
        """
}

/// The caches of one run of a tiny model, and the model state that goes with them.
///
/// The vision-language Qwen 3.5 models keep their M-RoPE anchor in the model state, and they
/// refuse to continue warm caches without it. Each call thus gives the state of the last call
/// back to the model, as the token iterator does.
private final class HybridRun {
    /// The caches, one for each layer that has a cache.
    let caches: [KVCache]

    /// The model state that the last call gave, or the state that a prompt cache file held.
    var state: LMOutput.State?

    /// Makes a run.
    ///
    /// - Parameters:
    ///   - caches: The caches of the run.
    ///   - state: The model state that goes with the caches.
    init(caches: [KVCache], state: LMOutput.State? = nil) {
        self.caches = caches
        self.state = state
    }
}

/// Tests that each recurrent layer of a tiny hybrid model moves the offset of its `MambaCache`.
///
/// Before, the recurrent layers of these models moved only the batch bookkeeping of their
/// `MambaCache`. The offset of each `MambaCache` thus stayed at 0, and the executor prompt cache
/// refused the caches of these models in memory and on disk. Now `ArraysCache.advance(_:)` also
/// moves the offset. Each test runs on each model of ``HybridRecurrentModelFixture``.
///
/// The sliding-window tests run on the BaichuanM1 models. BaichuanM1 gave the mask of the
/// global-attention layers also to its sliding-window layers, thus a prefill longer than the
/// window attended to more tokens than a decode.
///
/// The models compute in float32. The `MLXTestPrecision` target sets `MLX_ENABLE_TF32=0` when
/// this bundle loads, thus a split forward and a single forward agree closely also on a GPU with
/// neural accelerators.
struct HybridRecurrentCacheOffsetTests {

    // MARK: - Fixture values

    /// The prompt that the prefill puts in the caches. Each token is in the vocabulary of each
    /// tiny model.
    private static let prompt: [Int32] = [1, 7, 3, 9, 2, 11, 5, 13, 4, 8]

    /// The tokens that a next prompt adds after `prompt`.
    private static let promptTail: [Int32] = [6, 10, 12]

    /// The number of greedy tokens that each decode adds after the prompt.
    private static let decodeStepCount = 4

    /// The largest difference between the logits of the restored caches and the logits of the
    /// live caches. The two decodes run the same arrays, thus they agree to the float noise.
    private static let warmTolerance: Float = 1e-6

    /// The largest difference between the logits of a split forward (restored caches, live
    /// caches or a decode of one token at a time) and the logits of a cold prefill.
    ///
    /// A split forward adds rounding differences. The SSM layers of Mamba2 and NemotronH also
    /// send a prompt of many tokens through a chunked scan, and a single token through a step
    /// kernel. On 2026-09-25 the two paths differed by 3.1e-3 (Mamba2) and 1.8e-3 (NemotronH),
    /// with logits of at most 1.5 in absolute value. The differences were the same before and
    /// after the offset fix, thus they do not come from the offsets.
    private static let coldTolerance: Float = 5e-3

    // MARK: - Helpers

    /// Makes a run with the fresh caches of the model and no model state.
    ///
    /// - Parameter model: The model.
    /// - Returns: The run.
    /// - Throws: The error of the cache factory of the model.
    private static func newRun(_ model: any LanguageModel) throws -> HybridRun {
        HybridRun(caches: try model.newCache(parameters: nil))
    }

    /// Runs tokens into the caches of a run and gives back the logits of the last position. The
    /// model state that the call gives goes back into the run.
    ///
    /// - Parameters:
    ///   - model: The model.
    ///   - tokens: The tokens.
    ///   - run: The run whose caches the call fills.
    /// - Returns: The logits, with shape `(1, vocabulary)`.
    private static func lastLogits(
        _ model: any LanguageModel, _ tokens: [Int32], run: HybridRun
    ) -> MLXArray {
        let input = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        let output = model(input, cache: run.caches, state: run.state)
        run.state = output.state
        let last = output.logits[0..., -1, 0...]
        eval(last)
        return last
    }

    /// The token with the largest logit.
    ///
    /// - Parameter logits: The logits, with shape `(1, vocabulary)`.
    /// - Returns: The token.
    private static func greedyToken(_ logits: MLXArray) -> Int32 {
        argMax(logits, axis: -1).item(Int32.self)
    }

    /// The largest absolute difference between two arrays.
    ///
    /// - Parameters:
    ///   - lhs: The first array.
    ///   - rhs: The second array.
    /// - Returns: The difference.
    private static func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }

    /// Records an issue for each cache whose offset is not the expected position.
    ///
    /// The check reads the top-level cache of each layer, which the executor prompt cache reads,
    /// and each leaf in a `CacheList`, so that the children of a list also agree.
    ///
    /// - Parameters:
    ///   - caches: The caches, one for each layer that has a cache.
    ///   - position: The expected offset of each cache.
    ///   - label: The text that names the check in each issue.
    private static func expectOffsets(_ caches: [KVCache], _ position: Int, _ label: String) {
        for (layer, cache) in caches.enumerated() {
            #expect(
                cache.offset == position,
                "\(label), layer \(layer) (\(type(of: cache))): offset \(cache.offset)")
        }
        for leaf in KVCacheTree.leaves(in: caches) {
            #expect(
                leaf.cache.offset == position,
                "\(label), leaf \(leaf.path) (\(type(of: leaf.cache))): offset \(leaf.cache.offset)"
            )
        }
    }

    /// Records an issue unless the plan for the next prompt reuses exactly the whole ledger of
    /// `prompt` from `caches` and feeds only `promptTail`.
    ///
    /// - Parameters:
    ///   - caches: The caches after a prefill of `prompt`.
    ///   - label: The text that names the check in the issue.
    private static func expectReuseExtendsTheLedger(_ caches: [KVCache], _ label: String) {
        let nextPrompt = (prompt + promptTail).map(Int.init)

        let reuse = reconcilePromptCache(
            promptTokens: nextPrompt, cachedTokens: prompt.map(Int.init), caches: caches)

        #expect(
            reuse
                == PromptCacheReuse(
                    suffixStart: prompt.count, representedTokens: nextPrompt, kind: .extend),
            "\(label)")
    }

    /// Writes the caches and the model state of a run to a prompt cache file, and reads the file
    /// into fresh caches of the model.
    ///
    /// - Parameters:
    ///   - run: The run to write.
    ///   - model: The model that makes the fresh caches.
    /// - Returns: The restored run, with the caches and the model state of the file.
    /// - Throws: The error of the write or of the read.
    private static func savedAndRestored(
        _ run: HybridRun, model: any LanguageModel
    ) throws -> HybridRun {
        let url = PromptCacheTemplateRestoreTests.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: run.caches, state: run.state)
        let snapshot = try loadPromptCacheSnapshot(
            url: url, into: model.newCache(parameters: nil))
        return HybridRun(caches: snapshot.cache, state: snapshot.state)
    }

    // MARK: - Offsets

    @Test(
        "A prefill moves the offset of each cache, the recurrent caches too",
        arguments: HybridRecurrentModelFixture.allCases)
    func prefillMovesEveryOffset(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let run = try Self.newRun(model)

        _ = Self.lastLogits(model, Self.prompt, run: run)

        #expect(
            KVCacheTree.leaves(in: run.caches).contains { $0.cache is MambaCache },
            "\(fixture): has a recurrent cache")
        Self.expectOffsets(run.caches, Self.prompt.count, "\(fixture) after the prefill")
    }

    @Test(
        "Each decode step moves the offset of each cache by one token",
        arguments: HybridRecurrentModelFixture.allCases)
    func decodeStepsMoveEveryOffset(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let run = try Self.newRun(model)
        var token = Self.greedyToken(Self.lastLogits(model, Self.prompt, run: run))

        for step in 1 ... Self.decodeStepCount {
            token = Self.greedyToken(Self.lastLogits(model, [token], run: run))
            Self.expectOffsets(
                run.caches, Self.prompt.count + step, "\(fixture) after decode step \(step)")
        }
    }

    // MARK: - Sliding window

    /// A decode of one token at a time puts each token through the rotating cache of each
    /// sliding-window layer, thus each query attends only to the window. A cold prefill must
    /// attend to the same tokens. The mask of the prefill makes the window, and the test fails
    /// when the sliding-window layers get the mask of the global-attention layers.
    @Test(
        "A cold prefill agrees with a decode of one token at a time",
        arguments: [HybridRecurrentModelFixture.baichuanM1, .baichuanM1ShortWindow])
    func coldPrefillAgreesWithSingleTokenDecode(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let tokens = Self.prompt + Self.promptTail
        let stepRun = try Self.newRun(model)
        let stepLogits = try #require(
            tokens.map { Self.lastLogits(model, [$0], run: stepRun) }.last,
            "\(fixture): logits of the last decode step")
        let coldLogits = Self.lastLogits(model, tokens, run: try Self.newRun(model))

        #expect(
            Self.maxAbsDifference(stepLogits, coldLogits) <= Self.coldTolerance,
            "\(fixture): single-token and cold logits")
    }

    // MARK: - Warm continuation

    @Test(
        "The live caches of a prefill extend the ledger for the next prompt",
        arguments: HybridRecurrentModelFixture.allCases)
    func liveCachesExtendTheLedger(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let run = try Self.newRun(model)
        _ = Self.lastLogits(model, Self.prompt, run: run)

        Self.expectReuseExtendsTheLedger(run.caches, "\(fixture): reuse of the live caches")
    }

    @Test(
        "The live caches continue the next prompt as a cold prefill of the whole prompt",
        arguments: HybridRecurrentModelFixture.allCases)
    func liveCachesContinueTheNextPrompt(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let run = try Self.newRun(model)
        _ = Self.lastLogits(model, Self.prompt, run: run)
        let nextPrompt = Self.prompt + Self.promptTail
        let reuse = try #require(
            reconcilePromptCache(
                promptTokens: nextPrompt.map(Int.init), cachedTokens: Self.prompt.map(Int.init),
                caches: run.caches),
            "\(fixture): reuse of the live caches")

        let warmLogits = Self.lastLogits(
            model, Array(nextPrompt[reuse.suffixStart...]), run: run)
        let coldLogits = Self.lastLogits(model, nextPrompt, run: try Self.newRun(model))

        #expect(reuse.suffixStart == Self.prompt.count, "\(fixture): suffix start")
        #expect(
            Self.maxAbsDifference(warmLogits, coldLogits) <= Self.coldTolerance,
            "\(fixture): warm and cold logits")
        Self.expectOffsets(run.caches, nextPrompt.count, "\(fixture) after the warm continuation")
    }

    @Test(
        "A restored prefill has each offset and decodes as the live caches and a cold prefill",
        arguments: HybridRecurrentModelFixture.allCases)
    func restoredPrefillDecodesAsTheLiveCaches(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let liveRun = try Self.newRun(model)
        var token = Self.greedyToken(Self.lastLogits(model, Self.prompt, run: liveRun))
        let restoredRun = try Self.savedAndRestored(liveRun, model: model)
        let coldLogits = Self.lastLogits(
            model, Self.prompt + [token], run: try Self.newRun(model))

        Self.expectOffsets(restoredRun.caches, Self.prompt.count, "\(fixture) after the restore")
        for step in 0 ..< Self.decodeStepCount {
            let liveLogits = Self.lastLogits(model, [token], run: liveRun)
            let restoredLogits = Self.lastLogits(model, [token], run: restoredRun)
            #expect(
                Self.maxAbsDifference(restoredLogits, liveLogits) <= Self.warmTolerance,
                "\(fixture) step \(step): restored and live logits")
            if step == 0 {
                #expect(
                    Self.maxAbsDifference(restoredLogits, coldLogits) <= Self.coldTolerance,
                    "\(fixture) step 0: restored and cold logits")
            }
            token = Self.greedyToken(liveLogits)
        }
        Self.expectOffsets(
            restoredRun.caches, Self.prompt.count + Self.decodeStepCount,
            "\(fixture) after the restored decode")
    }

    /// The executor prompt cache restores a spilled turn from its file and then plans the next
    /// prompt. The plan must reuse the whole ledger of turn 1, not only a part of it.
    @Test(
        "A restored prefill extends exactly the ledger of turn 1 for the next prompt",
        arguments: HybridRecurrentModelFixture.allCases)
    func restoredPrefillExtendsExactlyTheLedger(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let liveRun = try Self.newRun(model)
        _ = Self.lastLogits(model, Self.prompt, run: liveRun)
        let restoredRun = try Self.savedAndRestored(liveRun, model: model)

        Self.expectReuseExtendsTheLedger(
            restoredRun.caches, "\(fixture): reuse of the restored caches")
    }
}
