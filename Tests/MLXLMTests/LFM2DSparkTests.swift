// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny LFM2 target that the DSpark tests and the directory-load tests build.
let tinyLFM2DSparkTargetJSON = """
    {
      "model_type": "lfm2",
      "vocab_size": 32,
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "max_position_embeddings": 128,
      "norm_eps": 1e-5,
      "conv_bias": false,
      "conv_L_cache": 3,
      "block_dim": 8,
      "intermediate_size": 24,
      "block_multiple_of": 8,
      "block_ffn_dim_multiplier": 1.0,
      "block_auto_adjust_ff_dim": false,
      "layer_types": ["conv", "full_attention"],
      "rope_theta": 1000000.0
    }
    """

private let tinyLFM2DSparkJSON = """
    {
      "architectures": ["Lfm2DSparkDraftModel"],
      "model_type": "qwen3",
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "intermediate_size": 16,
      "hidden_act": "silu",
      "rms_norm_eps": 1e-6,
      "vocab_size": 32,
      "rope_theta": 1000000.0,
      "max_position_embeddings": 128,
      "layer_types": ["full_attention", "full_attention"],
      "block_size": 3,
      "dflash_config": {
        "mask_token_id": 31,
        "target_layer_ids": [0, 1],
        "num_target_layers": 2
      },
      "markov_rank": 4,
      "rope_is_neox_style": false,
      "enable_confidence_head": true,
      "markov_head_type": "vanilla"
    }
    """

private func tinyDSparkConfiguration() throws -> LFM2DSparkConfiguration {
    try JSONDecoder().decode(
        LFM2DSparkConfiguration.self, from: Data(tinyLFM2DSparkJSON.utf8))
}

private func tinyDSparkTarget() throws -> LFM2Model {
    LFM2Model(
        try JSONDecoder().decode(
            LFM2Configuration.self, from: Data(tinyLFM2DSparkTargetJSON.utf8)))
}

private func tinyDSparkMoETarget() throws -> LFM2MoEModel {
    var object =
        try JSONSerialization.jsonObject(with: Data(tinyLFM2DSparkTargetJSON.utf8))
        as! [String: Any]
    object["model_type"] = "lfm2_moe"
    object["num_experts"] = 4
    object["num_experts_per_tok"] = 2
    object["moe_intermediate_size"] = 16
    object["num_dense_layers"] = 1
    object["norm_topk_prob"] = true
    object["use_expert_bias"] = true
    return LFM2MoEModel(
        try JSONDecoder().decode(
            LFM2MoEConfiguration.self, from: JSONSerialization.data(withJSONObject: object)))
}

@Test
func lfm2DSparkConfigurationAndCheckpointNamespaceMatchReleasedArchitecture() throws {
    let configuration = try tinyDSparkConfiguration()
    try configuration.validateModelConfiguration()
    let drafter = LFM2DSparkDraftModel(configuration)
    let parameters = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())

    #expect(drafter.maximumBlockSize == 4)
    #expect(drafter.targetLayerIds == [0, 1])
    #expect(drafter.requiresPromptPrefill)
    #expect(!drafter.requiresSharedTargetKV)
    #expect(drafter.requiresGreedySampling)
    #expect(parameters["fc.weight"]?.shape == [8, 16])
    #expect(parameters["hidden_norm.weight"]?.shape == [8])
    #expect(parameters["layers.0.self_attn.q_proj.weight"]?.shape == [8, 8])
    #expect(parameters["markov_head.markov_w1.weight"]?.shape == [32, 4])
    #expect(parameters["markov_head.markov_w2.weight"]?.shape == [32, 4])
    #expect(parameters["confidence_head.proj.weight"]?.shape == [1, 12])
    #expect(parameters["confidence_head.proj.bias"]?.shape == [1])
}

@Test
func lfm2DSparkRegistrationDisambiguatesTheQwen3ModelType() async throws {
    await LFM2DSparkRegistration.register()
    let model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: Data(tinyLFM2DSparkJSON.utf8), modelType: "qwen3")

    #expect(model is LFM2DSparkDraftModel)
    #expect(
        await MTPDrafterTypeRegistry.shared.contains(
            "qwen3", configuration: Data(tinyLFM2DSparkJSON.utf8)))
    #expect(
        await !MTPDrafterTypeRegistry.shared.contains(
            "qwen3",
            configuration: Data(
                #"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#.utf8)))
}

@Test
func lfm2DSparkRegistrationValidatesTheDecodedConfiguration() async throws {
    await LFM2DSparkRegistration.register()
    var object =
        try JSONSerialization.jsonObject(with: Data(tinyLFM2DSparkJSON.utf8)) as! [String: Any]
    object["num_key_value_heads"] = 3
    await #expect(throws: ModelFactoryError.self) {
        _ = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: object), modelType: "qwen3")
    }
}

@Test
func unconditionalDrafterRegistrationValidatesTheDecodedConfiguration() async throws {
    let registry = ModelTypeRegistry<any MTPDrafterModel>()
    await registry.registerDrafter(
        modelTypes: ["qwen3"], configuration: LFM2DSparkConfiguration.self,
        create: { LFM2DSparkDraftModel($0) })
    #expect(
        try await registry.createModel(
            configuration: Data(tinyLFM2DSparkJSON.utf8), modelType: "qwen3")
            is LFM2DSparkDraftModel)

    var object =
        try JSONSerialization.jsonObject(with: Data(tinyLFM2DSparkJSON.utf8)) as! [String: Any]
    object["architectures"] = ["Qwen3ForCausalLM"]
    await #expect(throws: ModelFactoryError.self) {
        _ = try await registry.createModel(
            configuration: JSONSerialization.data(withJSONObject: object), modelType: "qwen3")
    }
}

@Test
func lfm2DSparkIncompatibleTargetNamesTheDrafterAndTheTarget() throws {
    let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
    let target = Qwen3Model(
        try JSONDecoder.json5().decode(
            Qwen3Configuration.self,
            from: Data(
                #"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,"vocab_size":32,"rope_theta":1000000.0,"max_position_embeddings":128,"tie_word_embeddings":true}"#
                    .utf8)))
    #expect(
        throws: MTPDrafterCompatibilityError.incompatibleTarget(
            drafter: "LFM2DSparkDraftModel",
            expected: "an LFM2 target with matching dimensions and RoPE parameters",
            actual: "Qwen3Model")
    ) {
        try drafter.validateCompatibility(with: target)
    }
}

@Test
func lfm2DSparkRegistryContainsEveryReleasedCheckpoint() {
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-1.2B-Instruct-DSpark"))
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-2.6B-DSpark"))
    #expect(MTPDrafterRegistry.shared.contains(id: "LiquidAI/LFM2.5-8B-A1B-DSpark"))
}

@Suite(.serialized)
struct LFM2DSparkMetalTests {
    @Test
    func drafterCompatibilityAcceptsDenseAndMoEAndRejectsMismatchedTargets() throws {
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        #expect(drafter.isCompatible(with: try tinyDSparkTarget()))
        #expect(drafter.isCompatible(with: try tinyDSparkMoETarget()))
        for (key, value) in [
            ("hidden_size", 16), ("vocab_size", 64),
            ("num_hidden_layers", 3), ("rope_theta", 10000),
        ] {
            var object =
                try JSONSerialization.jsonObject(with: Data(tinyLFM2DSparkTargetJSON.utf8))
                as! [String: Any]
            object[key] = value
            if key == "hidden_size" {
                object["block_dim"] = value
            }
            if key == "num_hidden_layers" {
                object["layer_types"] = ["conv", "full_attention", "conv"]
            }
            let target = LFM2Model(
                try JSONDecoder().decode(
                    LFM2Configuration.self,
                    from: JSONSerialization.data(withJSONObject: object)))
            #expect(!drafter.isCompatible(with: target))
            #expect(throws: MTPDrafterCompatibilityError.self) {
                _ = try MTPSpeculativeTokenIterator(
                    input: LMInput(tokens: MLXArray([Int32(1), 2])),
                    mainModel: target, drafter: drafter,
                    parameters: GenerateParameters(temperature: 0), blockSize: 4)
            }
        }
    }

    @Test
    func targetEmitsOrderedDecoderFeatureTaps() throws {
        let target = try tinyDSparkTarget()
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerIdsKey] = [0, 1]
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let output = target(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(output.state?[mtpSharedKVStatesKey])
        eval(output.logits, hidden)

        #expect(output.logits.shape == [1, 4, 32])
        #expect(hidden.shape == [1, 4, 16])
        #expect(sharedKV.isEmpty)
    }

    @Test
    func draftProposalIsTransientAndCommitAppendsOnlyVerifiedTargetFeatures() throws {
        MLXRandom.seed(41)
        let target = try tinyDSparkTarget()
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let prompt = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        var emitState = LMOutput.State()
        emitState[mtpEmitFlagKey] = true
        emitState[mtpTargetLayerIdsKey] = [0, 1]
        let prefill = target(LMInput.Text(tokens: prompt), cache: nil, state: emitState)
        let promptHidden = try #require(prefill.state?[mtpLastHiddenStatesKey])
        let bonus = MLXArray([Int32(5)])
        var state = drafter.makeState(parameters: nil)

        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: promptHidden,
            firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
        eval(promptHidden)
        #expect(state.cache.allSatisfy { $0.offset == 4 })

        let proposal = drafter.draftBlock(
            target: target, lastToken: bonus,
            lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 4, blockSize: 4,
            state: &state, sampler: sampler)
        eval(proposal)
        #expect(proposal.shape == [1, 3])
        #expect(state.cache.allSatisfy { $0.offset == 4 })

        let verifyTokens = concatenated([bonus, proposal.flattened()])
        let verify = target(
            LMInput.Text(tokens: verifyTokens[.newAxis]), cache: nil, state: emitState)
        let verifyHidden = try #require(verify.state?[mtpLastHiddenStatesKey])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: MLXArray([Int32(6)]),
            positionDeltas: nil, state: &state, sampler: sampler)
        eval(verifyHidden)

        #expect(state.cache.allSatisfy { $0.offset == 6 })
        #expect(state.nextPosition == 6)
    }

    @Test(arguments: [3, 4, 7, 8, 12], [false, true])
    func rotatingHybridCacheMatchesGreedyAcrossWindow(promptLength: Int, useMoE: Bool) throws {
        MLXRandom.seed(41)
        let target: any LanguageModel
        if useMoE {
            target = try tinyDSparkMoETarget()
        } else {
            target = try tinyDSparkTarget()
        }
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        let input = LMInput(tokens: MLXArray((0 ..< promptLength).map { Int32($0 % 31) }))
        let parameters = GenerateParameters(maxTokens: 20, maxKVSize: 8, temperature: 0)
        let baselineCache = try target.newCache(parameters: parameters)
        let speculativeCache = try target.newCache(parameters: parameters)
        var baseline = try TokenIterator(
            input: input, model: target, cache: baselineCache, parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target, drafter: drafter,
            mainCache: speculativeCache, parameters: parameters, blockSize: 4)

        var expected: [Int] = []
        while let token = baseline.next() { expected.append(token) }
        var output: [Int] = []
        while let token = iterator.next() { output.append(token) }

        #expect(output == expected)
        #expect(output.count == 20)
        // In this fork the sliding window rewinds a verify write after it wraps, thus the
        // stream speculates across the window, where upstream falls back to plain decoding.
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        // TokenIterator commits its last emitted token while computing lookahead.
        let lastToken = try #require(output.last)
        let last = LMInput.Text(tokens: MLXArray([Int32(lastToken)])[.newAxis])
        eval(target(last, cache: speculativeCache, state: nil).logits)
        for (actual, expected) in zip(speculativeCache, baselineCache) {
            #expect(actual.offset == expected.offset)
        }
        let next = LMInput.Text(tokens: MLXArray([Int32(1)])[.newAxis])
        let actualLogits = target(next, cache: speculativeCache, state: nil).logits
        let expectedLogits = target(next, cache: baselineCache, state: nil).logits
        let difference = abs(actualLogits - expectedLogits).max().item(Float.self)
        // TF32 matmuls (neural accelerators) leave ~2e-4 here; see MatmulPrecision.
        #expect(difference < MatmulPrecision.tolerance(float32: 1e-4, reduced: 1e-3))
    }

    @Test
    func iteratorUsesLFMHybridNativeRollbackPath() throws {
        let target = try tinyDSparkTarget()
        let drafter = LFM2DSparkDraftModel(try tinyDSparkConfiguration())
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4]))
        let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
        var baseline = try TokenIterator(
            input: input, model: target, parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target, drafter: drafter,
            parameters: parameters, blockSize: 4)

        var expected: [Int] = []
        while let token = baseline.next() { expected.append(token) }
        var output: [Int] = []
        while let token = iterator.next() { output.append(token) }

        #expect(output == expected)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
    }
}
