// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@testable import MLXLLM

/// A tiny model with a `MambaCache` for each recurrent layer.
///
/// Each of these models feeds its conv or SSM layer through a `MambaCache`. The executor prompt
/// cache keeps a cache only when the offset of each cache is the length of its token ledger, thus
/// the recurrent layer must move the offset of its `MambaCache` as an attention layer does.
enum HybridRecurrentModelFixture: String, CaseIterable, Sendable {
    case nemotronH
    case jamba
    case mamba2
    case graniteMoeHybrid
    case lfm2
    case lfm2MoE
    case lfm2VL

    /// The seed of the initializer weights of each tiny model.
    private static let weightSeed: UInt64 = 43

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
            return NemotronHModel(Self.nemotronHConfiguration)
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
        }
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
    private static let nemotronHConfiguration = NemotronHConfiguration(
        vocabSize: 32, hiddenSize: 64, numHiddenLayers: 4, numAttentionHeads: 4,
        numKeyValueHeads: 2, mambaNumHeads: 4, mambaHeadDim: 16, ssmStateSize: 16,
        convKernel: 4, nGroups: 2, intermediateSize: 128, moeIntermediateSize: 64,
        moeSharedExpertIntermediateSize: 64, nRoutedExperts: 4, numExpertsPerTok: 2,
        hybridOverridePattern: "M*M-", layerNormEpsilon: 1e-5, nGroup: 2, topkGroup: 1)

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
}

/// Tests that each recurrent layer of a tiny hybrid model moves the offset of its `MambaCache`.
///
/// The recurrent layers called `ArraysCache.advance(_:)`, which moves only the batch bookkeeping.
/// The offset of each `MambaCache` thus stayed at 0, and the executor prompt cache refused the
/// caches of these models in memory and on disk. Each test runs on each model of
/// ``HybridRecurrentModelFixture``.
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

    /// The largest difference between the logits of the restored caches and the logits of a
    /// cold prefill of the prompt and the first token.
    ///
    /// A split forward adds rounding differences. The SSM layers of Mamba2 and NemotronH also
    /// send a prompt of many tokens through a chunked scan, and a single token through a step
    /// kernel. On 2026-09-25 the two paths differed by 3.1e-3 (Mamba2) and 1.8e-3 (NemotronH),
    /// with logits of at most 1.5 in absolute value. The differences were the same before and
    /// after the offset fix, thus they do not come from the offsets.
    private static let coldTolerance: Float = 5e-3

    // MARK: - Helpers

    /// Runs tokens into caches and gives back the logits of the last position.
    ///
    /// - Parameters:
    ///   - model: The model.
    ///   - tokens: The tokens.
    ///   - caches: The caches to fill.
    /// - Returns: The logits, with shape `(1, vocabulary)`.
    private static func lastLogits(
        _ model: any LanguageModel, _ tokens: [Int32], caches: [KVCache]
    ) -> MLXArray {
        let logits = model(MLXArray(tokens).expandedDimensions(axis: 0), cache: caches)
        let last = logits[0..., -1, 0...]
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
    }

    // MARK: - Offsets

    @Test(
        "A prefill moves the offset of each cache, the recurrent caches too",
        arguments: HybridRecurrentModelFixture.allCases)
    func prefillMovesEveryOffset(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let caches = try model.newCache(parameters: nil)

        _ = Self.lastLogits(model, Self.prompt, caches: caches)

        #expect(caches.contains { $0 is MambaCache }, "\(fixture): has a recurrent cache")
        Self.expectOffsets(caches, Self.prompt.count, "\(fixture) after the prefill")
    }

    @Test(
        "Each decode step moves the offset of each cache by one token",
        arguments: HybridRecurrentModelFixture.allCases)
    func decodeStepsMoveEveryOffset(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let caches = try model.newCache(parameters: nil)
        var token = Self.greedyToken(Self.lastLogits(model, Self.prompt, caches: caches))

        for step in 1 ... Self.decodeStepCount {
            token = Self.greedyToken(Self.lastLogits(model, [token], caches: caches))
            Self.expectOffsets(
                caches, Self.prompt.count + step, "\(fixture) after decode step \(step)")
        }
    }

    // MARK: - Warm continuation

    @Test(
        "The live caches of a prefill extend the ledger for the next prompt",
        arguments: HybridRecurrentModelFixture.allCases)
    func liveCachesExtendTheLedger(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let caches = try model.newCache(parameters: nil)
        _ = Self.lastLogits(model, Self.prompt, caches: caches)
        let nextPrompt = (Self.prompt + Self.promptTail).map(Int.init)

        let reuse = reconcilePromptCache(
            promptTokens: nextPrompt, cachedTokens: Self.prompt.map(Int.init), caches: caches)

        #expect(
            reuse
                == PromptCacheReuse(
                    suffixStart: Self.prompt.count, representedTokens: nextPrompt, kind: .extend),
            "\(fixture): reuse of the live caches")
    }

    @Test(
        "A restored prefill has each offset and decodes as the live caches and a cold prefill",
        arguments: HybridRecurrentModelFixture.allCases)
    func restoredPrefillDecodesAsTheLiveCaches(_ fixture: HybridRecurrentModelFixture) throws {
        let model = try fixture.makeModel()
        let liveCaches = try model.newCache(parameters: nil)
        var token = Self.greedyToken(Self.lastLogits(model, Self.prompt, caches: liveCaches))
        let url = PromptCacheTemplateRestoreTests.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: liveCaches)
        let restoredCaches = try loadPromptCacheSnapshot(
            url: url, into: model.newCache(parameters: nil)
        ).cache
        let coldLogits = Self.lastLogits(
            model, Self.prompt + [token], caches: try model.newCache(parameters: nil))

        Self.expectOffsets(restoredCaches, Self.prompt.count, "\(fixture) after the restore")
        for step in 0 ..< Self.decodeStepCount {
            let liveLogits = Self.lastLogits(model, [token], caches: liveCaches)
            let restoredLogits = Self.lastLogits(model, [token], caches: restoredCaches)
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
            restoredCaches, Self.prompt.count + Self.decodeStepCount,
            "\(fixture) after the restored decode")
    }
}
