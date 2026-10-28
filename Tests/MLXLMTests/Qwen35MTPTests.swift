import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Test
func testQwen35TextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35VLMTextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Suite
struct Qwen35MTPCompatibilityTests {
    @Test
    func validatesTargetFamily() throws {
        let textConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let textTarget = MLXLLM.Qwen35TextModel(textConfig)
        let textDrafter = MLXLLM.Qwen35MTPDraftModel(textConfig)
        let wrappedTextConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8))

        let vlmConfig = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let vlmTarget = MLXVLM.Qwen35(vlmConfig)
        let vlmDrafter = MLXVLM.Qwen35VLMNextNDraftModel(vlmConfig)

        try textDrafter.validateCompatibility(with: textTarget)
        try textDrafter.validateCompatibility(with: MLXLLM.Qwen35Model(wrappedTextConfig))
        try vlmDrafter.validateCompatibility(with: vlmTarget)

        #expect(throws: MTPDrafterCompatibilityError.self) {
            try textDrafter.validateCompatibility(with: vlmTarget)
        }
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try vlmDrafter.validateCompatibility(with: textTarget)
        }
    }

    @Test
    func textDrafterRejectsSameFamilyDimensionMismatches() throws {
        let drafterConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let drafter = MLXLLM.Qwen35MTPDraftModel(drafterConfig)

        let hiddenSizeMismatch = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1, hiddenSize: 32).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXLLM.Qwen35TextModel(hiddenSizeMismatch))
        }

        let vocabularySizeMismatch = try JSONDecoder().decode(
            MLXLLM.Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.wrappedText(
                    modelType: "qwen3_5",
                    textConfigJSON: Qwen35FixtureJSON.text(mtpLayers: 1, vocabularySize: 17)
                ).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXLLM.Qwen35Model(vocabularySizeMismatch))
        }
    }

    @Test
    func vlmDrafterRejectsSameFamilyDimensionMismatches() throws {
        let drafterConfig = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(drafterConfig)

        let hiddenSizeMismatch = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1, hiddenSize: 32).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXVLM.Qwen35(hiddenSizeMismatch))
        }

        let vocabularySizeMismatch = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1, vocabularySize: 17).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXVLM.Qwen35(vocabularySizeMismatch))
        }
    }
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let sanitized = try drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 32, 16]),
        "mtp.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 16, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.gate_up_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.down_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
    let norm = try #require(sanitized["mtp.norm.weight"])
    let pre = try #require(sanitized["mtp.pre_fc_norm_embedding.weight"])
    eval(norm, pre)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(pre, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test(arguments: [false, true])
func testQwen35MTPDraftSanitizeRejectsCompetingComponents(vision: Bool) throws {
    let data = Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8)
    let drafter: any BaseLanguageModel
    if vision {
        drafter = MLXVLM.Qwen35VLMNextNDraftModel(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.TextConfiguration.self, from: data))
    } else {
        drafter = MLXLLM.Qwen35MTPDraftModel(
            try JSONDecoder().decode(MLXLLM.Qwen35TextConfiguration.self, from: data))
    }
    let weights = [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "language_model.mtp.norm.weight": MLXArray.ones([16]),
    ]
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights)
    }
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights, metadata: [:])
    }
}

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(Qwen35FixtureJSON.standaloneMTP().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = try drafter.sanitize(weights: ["mtp.norm.weight": weight])
    let norm = try #require(sanitized["mtp.norm.weight"])
    eval(norm)
    #expect(allClose(norm, weight, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35MTPDraftSanitizeStacksPerExpertMoEWeights() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1, numExperts: 2).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let weights: [String: MLXArray] = [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": MLXArray.ones([16, 16]),
    ]

    let sanitized = try drafter.sanitize(weights: weights)

    #expect(sanitized["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
}

@Test
func testQwen35MTPDraftInstantiatesDedicatedEmbeddingWhenConfigured() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            Qwen35FixtureJSON.text(mtpLayers: 1, mtpUseDedicatedEmbeddings: true).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    #expect(drafter.mtp.embedTokens != nil)
    let sanitized = try drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

/// Bound for float32 values compared at full precision: on machines whose float32 matmuls run
/// in IEEE float32, and for values every run reaches through matmuls of the same shapes.
private let ieeeFloat32Tolerance = 1e-5

/// Bound for float32 values that the same inputs reach through matmuls of different shapes on
/// machines whose float32 matmuls run as TF32; see ``MatmulPrecision``.
private let tf32ReshapedMatmulTolerance = 2e-3

/// Bound for comparing a gated-delta-net state reached two ways. Wider on machines whose
/// float32 matmuls run as TF32; see ``MatmulPrecision``.
private let gdnStateTolerance = MatmulPrecision.tolerance(
    float32: ieeeFloat32Tolerance, reduced: tf32ReshapedMatmulTolerance)

/// Bound for comparing a gated-delta-net output computed with and without a checkpoint. Both
/// runs go through matmuls of the same shapes, so full float32 precision holds on every machine.
private let gdnCheckpointOutputTolerance = ieeeFloat32Tolerance

/// `count` token ids below `vocabularySize`, shape `[count]`, drawn from `seed`: the same
/// tokens for the same seed on every run.
func seededTokens(_ count: Int, vocabularySize: Int, seed: UInt64) -> MLXArray {
    MLXRandom.randInt(Int32(0) ..< Int32(vocabularySize), [count], key: MLXRandom.key(seed))
}

@Suite(.serialized)
struct Qwen35MTPMetalTests {
    @Test
    func testQwen35MTPPredictorAdvancesEveryLayerCachePerToken() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 2).utf8))
        let predictor = MLXLLM.Qwen35MTPPredictor(cfg)
        let cache = predictor.newCache()
        let embeds = MLXArray.zeros([1, 1, 16])
        let hidden = MLXArray.zeros([1, 1, 16])

        let first = predictor(
            inputsEmbeds: embeds, hiddenStates: hidden, cache: cache,
            positionOffset: 128)
        eval(first)
        #expect(cache[0].offset == 1)
        #expect(cache[1].offset == 1)

        let second = predictor(
            inputsEmbeds: embeds, hiddenStates: first, cache: cache,
            positionOffset: 129)
        eval(second)
        #expect(cache[0].offset == 2)
        #expect(cache[1].offset == 2)
    }

    @Test
    func testQwen35VLMMTPPositionIdsApplyMultimodalDelta() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 2,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 2, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[1, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[2, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[1, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[2, 1, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35VLMMTPPositionIdsRepeatAndTrimShortBatchDeltas() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 4,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 4, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[0, 2, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 3, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35TextModelEmitDrafterStateBySynthetic() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let cache = try model.newCache(parameters: nil as GenerateParameters?)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        let out = model(input, cache: cache, state: state)

        let hidden = try #require(out.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(out.state?[mtpSharedKVStatesKey])
        let sharedKVOffsets = try #require(out.state?[mtpSharedKVOffsetsKey])
        eval(out.logits, hidden)
        #expect(out.logits.shape == [1, 4, 16])
        #expect(hidden.shape == [1, 4, 16])
        #expect(Set(sharedKV.keys) == ["full_attention"])
        #expect(sharedKVOffsets == ["full_attention": 4])
        let full = try #require(sharedKV["full_attention"])
        eval(full.0, full.1)
        #expect(full.0.shape.count == 4)
        #expect(full.1.shape.count == 4)
    }

    @Test
    func testQwen35TextModelEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let expected = model.model.forward(tokens, applyFinalNorm: false)
        let normalized = model.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35VLMEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let model = MLXVLM.Qwen35(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let base = MLXArray(0 ..< 4).asType(.int32).reshaped([1, 1, 4])
        let positionIds = broadcast(base, to: [3, 1, 4])
        let expected = model.languageModel.model(
            tokens, positionIds: positionIds, applyFinalNorm: false)
        let normalized = model.languageModel.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model.languageModel(
            tokens, cache: nil, state: state, positionIds: positionIds)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35DrafterCacheTracksVerifiedSequenceAcrossAcceptRejectPatterns() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let targetOutput = target(LMInput.Text(tokens: prompt), cache: nil, state: targetState)
        let promptHidden = try #require(targetOutput.state?[mtpLastHiddenStatesKey])

        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            var state = drafter.makeState(parameters: nil)
            var bonus = MLXArray([Int32(4)])
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt, targetHidden: promptHidden,
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
            eval(state.seedToken!, state.seedHidden!)
            #expect(state.cache.allSatisfy { $0.offset == 3 })
            #expect(state.nextPosition == 3)

            var expectedPosition = 3
            for accepted in pattern {
                let proposal = drafter.draftBlock(
                    target: target, lastToken: bonus,
                    lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
                    positionDeltas: nil, queryOffset: expectedPosition, blockSize: 2,
                    state: &state, sampler: sampler)
                eval(proposal)
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })

                let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
                let finalToken = MLXArray([Int32(8 + accepted)])
                drafter.commitDrafterState(
                    target: target, targetHidden: verifyHidden, draftTokens: proposal,
                    acceptedCount: accepted, finalToken: finalToken, positionDeltas: nil,
                    state: &state, sampler: sampler)
                eval(state.seedToken!, state.seedHidden!)
                expectedPosition += accepted + 1
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })
                #expect(state.nextPosition == expectedPosition)
                bonus = finalToken
            }
        }
    }

    @Test
    func testQwen35TextDrafterFinalizesAndResumesCanonicalBoundary() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let targetOutput = target(LMInput.Text(tokens: prompt), cache: nil, state: targetState)
        let promptHidden = try #require(targetOutput.state?[mtpLastHiddenStatesKey])

        var state = drafter.makeState(parameters: nil)
        var bonus = MLXArray([Int32(4)])
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: promptHidden,
            firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
        let proposal = drafter.draftBlock(
            target: target, lastToken: bonus,
            lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 3, blockSize: 2,
            state: &state, sampler: sampler)
        let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        bonus = MLXArray([Int32(5)])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: bonus, positionDeltas: nil,
            state: &state, sampler: sampler)
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 5)
        #expect(state.cache.allSatisfy { $0.offset == 5 })

        let boundaryHidden = verifyHidden[0..., 1 ..< 2, 0...]
        #expect(
            drafter.finalizeDrafterState(
                target: target, targetBoundaryHidden: boundaryHidden,
                targetProcessedTokenCount: 5, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == 4)
        #expect(state.cache.allSatisfy { $0.offset == 4 })
        #expect(state.seedToken == nil)
        #expect(state.seedHidden == nil)

        let suffix = MLXArray([Int32(6), 7]).reshaped([1, 2])
        let suffixHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        #expect(
            drafter.resumeDrafterState(
                target: target, suffixTokens: suffix,
                suffixTargetHidden: suffixHidden,
                targetBoundaryHidden: boundaryHidden,
                firstBonus: MLXArray([Int32(8)]), positionDeltas: nil,
                state: &state, sampler: sampler))
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 7)
        #expect(state.cache.allSatisfy { $0.offset == 7 })
    }

    @Test
    func testQwen35TextDrafterFinalizationAlsoDiscardsTargetLookahead() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, cfg.hiddenSize])
        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(4)]), positionDeltas: nil,
            state: &state, sampler: sampler)
        let proposal = drafter.draftBlock(
            target: target, lastToken: MLXArray([Int32(4)]),
            lastHidden: hidden[0..., (-1)..., 0...], sharedKV: [:],
            positionDeltas: nil, queryOffset: 3, blockSize: 2,
            state: &state, sampler: sampler)
        let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: proposal,
            acceptedCount: 1, finalToken: MLXArray([Int32(5)]),
            positionDeltas: nil, state: &state, sampler: sampler)

        #expect(state.nextPosition == 5)
        #expect(
            drafter.finalizeDrafterState(
                target: target,
                targetBoundaryHidden: verifyHidden[0..., ..<1, 0...],
                targetProcessedTokenCount: 4, discardedTargetTokens: 1,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == 3)
        #expect(state.cache.allSatisfy { $0.offset == 3 })
    }

    @Test
    func testQwen35VLMDrafterResumeKeepsPositionDeltaAlignment() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(cfg)
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, cfg.textConfiguration.hiddenSize])
        let positionDeltas = MLXArray([Int32(2)])
        var state = drafter.makeState(parameters: nil)

        drafter.prepareDrafterState(
            target: target, promptTokens: prompt, targetHidden: hidden,
            firstBonus: MLXArray([Int32(4)]), positionDeltas: positionDeltas,
            state: &state, sampler: sampler)
        #expect(state.nextPosition == 3)
        #expect(
            drafter.finalizeDrafterState(
                target: target,
                targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: 3, discardedTargetTokens: 0,
                positionDeltas: positionDeltas, state: &state))
        #expect(state.nextPosition == 2)

        #expect(
            drafter.resumeDrafterState(
                target: target,
                suffixTokens: MLXArray([Int32(5), 6]).reshaped([1, 2]),
                suffixTargetHidden: MLXArray.zeros(
                    [1, 2, cfg.textConfiguration.hiddenSize]),
                targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                firstBonus: MLXArray([Int32(7)]),
                positionDeltas: positionDeltas, state: &state, sampler: sampler))
        eval(state.seedToken!, state.seedHidden!)
        #expect(state.nextPosition == 5)
        #expect(state.cache.allSatisfy { $0.offset == 5 })
    }

    @Test
    func testQwen35TextChatSessionResumesActualDrafterState() async throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let speculative = try fixture.makeChatSession(speculative: true, maxTokens: 4)
        let targetOnly = try fixture.makeChatSession(speculative: false, maxTokens: 4)

        let firstPrompt = fixture.text(for: [1])
        let secondPrompt = fixture.text(for: [2])
        let speculativeFirst = try await collectQwenChatDetails(
            speculative.streamDetails(to: firstPrompt))
        let targetFirst = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: firstPrompt))
        let fork = await speculative.fork()
        let forkSecond = try await collectQwenChatDetails(
            fork.streamDetails(to: secondPrompt))
        let speculativeSecond = try await collectQwenChatDetails(
            speculative.streamDetails(to: secondPrompt))
        let targetSecond = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: secondPrompt))

        #expect(speculativeFirst.text == targetFirst.text)
        #expect(speculativeSecond.text == targetSecond.text)
        #expect(speculativeSecond.info.promptTokenCount == 2)
        #expect(speculativeSecond.info.cachedPromptTokenCount > 0)
        #expect((speculativeSecond.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculativeSecond.info.passthroughReason == nil)
        #expect(forkSecond.text == targetSecond.text)
        #expect(forkSecond.info.promptTokenCount == 2)
        #expect(forkSecond.info.cachedPromptTokenCount > 0)
        #expect((forkSecond.info.proposedDraftTokens ?? 0) > 0)
        #expect(forkSecond.info.passthroughReason == nil)
    }

    @Test
    func testQwen35VLMChatSessionResumesActualDrafterState() async throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(cfg)
        let tokenizer = ScalarTokenizer(vocabularySize: Qwen35FixtureJSON.defaultVocabularySize)
        let processor = ScalarTokenizerInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "tiny-qwen-mtp-chat-session"))
        let context = ModelContext(
            configuration: processor.configuration,
            model: target,
            processor: processor,
            tokenizer: tokenizer)
        let drafterContainer = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: processor.configuration,
                model: MLXVLM.Qwen35VLMNextNDraftModel(cfg)))
        let speculative = ChatSession(
            context,
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: drafterContainer, blockSize: 2),
            generateParameters: .init(maxTokens: 4, temperature: 0))
        let targetOnly = ChatSession(
            context,
            generateParameters: .init(maxTokens: 4, temperature: 0))

        let firstPrompt = tokenizer.text(for: 1)
        let secondPrompt = tokenizer.text(for: 2)
        let speculativeFirst = try await collectQwenChatDetails(
            speculative.streamDetails(to: firstPrompt))
        let targetFirst = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: firstPrompt))
        let speculativeSecond = try await collectQwenChatDetails(
            speculative.streamDetails(to: secondPrompt))
        let targetSecond = try await collectQwenChatDetails(
            targetOnly.streamDetails(to: secondPrompt))

        #expect(speculativeFirst.text == targetFirst.text)
        #expect(speculativeSecond.text == targetSecond.text)
        #expect(speculativeSecond.info.promptTokenCount == 2)
        #expect(speculativeSecond.info.cachedPromptTokenCount > 0)
        #expect((speculativeSecond.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculativeSecond.info.passthroughReason == nil)
    }

    @Test
    func testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let layer = MLXLLM.Qwen35GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, keepsSnapshots: true)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(
            allClose(
                speculative, full, rtol: gdnCheckpointOutputTolerance,
                atol: gdnCheckpointOutputTolerance
            ).item(Bool.self))
        #expect(speculativeCache.speculativeSlotCount == input.dim(1))
        #expect(speculativeCache.selectSpeculativeSlot(rollingBack: input.dim(1) - 1))

        let restored = speculativeCache.state
        let expected = prefixCache.state
        #expect(restored.count == expected.count)
        for (actual, reference) in zip(restored, expected) {
            eval(actual, reference)
            // The checkpoint and the prefix run reach the same state through
            // differently shaped matmuls, so the bound follows the machine's
            // float32 matmul precision; see MatmulPrecision.
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35VLMGDNCheckpointMatchesPrefix() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, keepsSnapshots: true)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(
            allClose(
                speculative, full, rtol: gdnCheckpointOutputTolerance,
                atol: gdnCheckpointOutputTolerance
            ).item(Bool.self))
        #expect(speculativeCache.selectSpeculativeSlot(rollingBack: input.dim(1) - 1))
        for (actual, reference) in zip(speculativeCache.state, prefixCache.state) {
            eval(actual, reference)
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically() {
        let attention = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 3, 2])
        _ = attention.update(keys: keys, values: keys)

        let recurrent = MambaCache()
        let checkpointConv = MLXArray.ones([1, 1, 4])
        let checkpointState = MLXArray.ones([1, 2, 2, 2])
        // The slots of a two-token verify: the states after its first token are the checkpoint.
        recurrent.saveSpeculativeSnapshots(
            GatedDeltaDecodeKernel.Snapshots(
                recurrent: stacked([checkpointState, zeros(like: checkpointState)]),
                convWindow: concatenated(
                    [zeros(like: checkpointConv), checkpointConv, zeros(like: checkpointConv)],
                    axis: 1)))
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: 1)
        #expect(rewound == 1)
        #expect(attention.offset == 2)
        #expect(recurrent.speculativeSlotCount == 0)

        let restored = recurrent.state
        #expect(restored.count == 2)
        eval(restored[0], restored[1])
        #expect(allClose(restored[0], checkpointConv, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(restored[1], checkpointState, rtol: 0, atol: 0).item(Bool.self))
    }
}

/// Bound for comparing drafter cache entries that the same tokens reach through matmuls of
/// different shapes. Wider on machines whose float32 matmuls run as TF32; see
/// ``MatmulPrecision``.
private let drafterCacheTolerance = MatmulPrecision.tolerance(
    float32: ieeeFloat32Tolerance, reduced: tf32ReshapedMatmulTolerance)

/// The drafter state API beyond a cold prefill at position zero: a cold start at a position
/// the target cache already holds, and trimming and appending entries.
@Suite(.serialized)
struct Qwen35MTPDrafterStateTests {

    /// Prompt tokens of the fixture vocabulary.
    private let prompt = [3, 9, 14, 22, 5, 40]

    /// The token the target sampled after ``prompt``.
    private let bonusToken: Int32 = 17

    /// ``bonusToken`` as the iterator hands it to the drafter.
    private var bonus: MLXArray { MLXArray([bonusToken]) }

    /// The prompt token the next turn appends after ``bonusToken``.
    private let nextTurnPromptToken: Int32 = 8

    /// The token the target sampled after the next turn's suffix.
    private let nextTurnBonusToken: Int32 = 11

    /// Suffix tokens the VLM cold start feeds after the cached prefix.
    private let vlmSuffixTokens: [Int32] = [5, 6]

    /// The token the VLM target sampled after ``vlmSuffixTokens``.
    private let vlmBonusToken: Int32 = 7

    /// The multimodal RoPE position delta the VLM cold start applies.
    private let vlmPositionDelta: Int32 = 2

    /// Entries the trim and append test removes and then appends again.
    private let replayedEntryCount = 3

    /// Tokens of the prompt that a warm target cache already holds.
    private let cachedPrefixLength = 4

    /// Tokens each greedy decode generates.
    private let generatedTokenCount = 8

    /// The target hidden rows of `tokens`, emitted as the MTP iterator asks for them.
    private func targetHidden(
        _ fixture: Qwen35MTPFixture, tokens: [Int], cache: [KVCache]? = nil
    ) throws -> MLXArray {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let output = fixture.target(
            LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis, 0...]),
            cache: cache, state: state)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(hidden)
        return hidden
    }

    /// A drafter state prefilled over ``prompt`` from position zero.
    private func preparedState(
        _ fixture: Qwen35MTPFixture, hidden: MLXArray, sampler: any LogitSampler
    ) -> MTPDrafterState {
        var state = fixture.drafter.makeState(parameters: nil)
        fixture.drafter.prepareDrafterState(
            target: fixture.target,
            promptTokens: MLXArray(prompt.map(Int32.init))[.newAxis, 0...],
            targetHidden: hidden, firstBonus: bonus, positionDeltas: nil, state: &state,
            sampler: sampler)
        return state
    }

    /// Consumes the seed proposal, then drafts one tentative entry past it.
    private func appendTentativeEntry(
        _ fixture: Qwen35MTPFixture, hidden: MLXArray, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        for _ in 0 ..< Qwen35MTPFixture.blockSize {
            let proposal = fixture.drafter.draftBlock(
                target: fixture.target, lastToken: bonus,
                lastHidden: hidden[0..., (-1)..., 0...], sharedKV: [:], positionDeltas: nil,
                queryOffset: state.nextPosition, blockSize: Qwen35MTPFixture.blockSize,
                state: &state, sampler: sampler)
            eval(proposal)
        }
    }

    @Test("trimming entries and appending them again matches a cold prefill")
    func trimAndAppendMatchColdPrefill() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).sampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        let cold = preparedState(fixture, hidden: hidden, sampler: sampler)
        var replayed = preparedState(fixture, hidden: hidden, sampler: sampler)
        appendTentativeEntry(fixture, hidden: hidden, state: &replayed, sampler: sampler)
        #expect(replayed.proposalAppended == 1)

        let removed = replayed.removeLastEntries(replayed.proposalAppended + replayedEntryCount)
        #expect(removed)
        #expect(replayed.proposalAppended == 0)
        #expect(replayed.seedToken == nil)
        let keptEntries = prompt.count - replayedEntryCount
        let tokens = MLXArray(prompt[(keptEntries + 1)...].map(Int32.init))[.newAxis, 0...]
        #expect(
            fixture.drafter.appendDrafterState(
                target: fixture.target,
                tokens: concatenated([tokens, bonus[.newAxis, 0...]], axis: 1),
                targetHidden: hidden[0..., keptEntries..., 0...], positionDeltas: nil,
                state: &replayed, sampler: sampler))

        #expect(replayed.nextPosition == cold.nextPosition)
        #expect(replayed.isAligned)
        let replayedArrays = replayed.cache.flatMap(\.state)
        let coldArrays = cold.cache.flatMap(\.state)
        #expect(replayedArrays.count == coldArrays.count)
        for (actual, expected) in zip(replayedArrays, coldArrays) {
            eval(actual, expected)
            try #require(actual.shape == expected.shape)
            #expect(
                allClose(
                    actual, expected, rtol: drafterCacheTolerance, atol: drafterCacheTolerance
                ).item(Bool.self))
        }
        let replayedSeed = try #require(replayed.seedToken)
        let coldSeed = try #require(cold.seedToken)
        #expect(replayedSeed.asArray(Int32.self) == coldSeed.asArray(Int32.self))
    }

    @Test("finalization drops tentative proposal entries before the target boundary")
    func finalizationDropsTentativeEntries() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).sampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        var state = preparedState(fixture, hidden: hidden, sampler: sampler)
        appendTentativeEntry(fixture, hidden: hidden, state: &state, sampler: sampler)

        #expect(
            fixture.drafter.finalizeDrafterState(
                target: fixture.target, targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: prompt.count, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == prompt.count - 1)
        #expect(state.proposalAppended == 0)
        #expect(state.isAligned)
    }

    @Test("a cold start at a cached position resumes as a continuation")
    func coldStartAtCachedPositionResumes() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).sampler()
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)
        let suffix = Array(prompt.dropFirst(cachedPrefixLength))
        let suffixHidden = try targetHidden(fixture, tokens: suffix, cache: cache)
        var state = fixture.drafter.makeState(parameters: nil)

        #expect(
            fixture.drafter.coldStartDrafterState(
                target: fixture.target,
                suffixTokens: MLXArray(suffix.map(Int32.init))[.newAxis, 0...],
                suffixTargetHidden: suffixHidden, firstBonus: bonus,
                position: cachedPrefixLength, positionDeltas: nil, state: &state,
                sampler: sampler))
        let seed = try #require(state.seedToken)
        eval(seed)
        #expect(state.cacheStartPosition == cachedPrefixLength)
        #expect(state.nextPosition == prompt.count)
        #expect(state.cache.allSatisfy { $0.offset == suffix.count })

        #expect(
            fixture.drafter.finalizeDrafterState(
                target: fixture.target, targetBoundaryHidden: suffixHidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: prompt.count, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == prompt.count - 1)

        let nextSuffix = MLXArray([bonusToken, nextTurnPromptToken])[.newAxis, 0...]
        #expect(
            fixture.drafter.resumeDrafterState(
                target: fixture.target, suffixTokens: nextSuffix,
                suffixTargetHidden: MLXArray.zeros([
                    1, nextSuffix.dim(1), Qwen35MTPFixture.hiddenSize,
                ]),
                targetBoundaryHidden: suffixHidden[0..., (-1)..., 0...],
                firstBonus: MLXArray([nextTurnBonusToken]), positionDeltas: nil, state: &state,
                sampler: sampler))
        #expect(state.nextPosition == prompt.count + nextSuffix.dim(1))
        #expect(state.isAligned)
    }

    @Test("the VLM drafter cold-starts at a cached position with its position deltas")
    func vlmColdStartAtCachedPosition() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(cfg)
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let suffix = MLXArray(vlmSuffixTokens)[.newAxis, 0...]
        var state = drafter.makeState(parameters: nil)

        #expect(
            drafter.coldStartDrafterState(
                target: target, suffixTokens: suffix,
                suffixTargetHidden: MLXArray.zeros(
                    [1, suffix.dim(1), cfg.textConfiguration.hiddenSize]),
                firstBonus: MLXArray([vlmBonusToken]), position: cachedPrefixLength,
                positionDeltas: MLXArray([vlmPositionDelta]), state: &state, sampler: sampler))
        let seed = try #require(state.seedToken)
        eval(seed)
        #expect(state.cacheStartPosition == cachedPrefixLength)
        #expect(state.nextPosition == cachedPrefixLength + suffix.dim(1))
        #expect(state.isAligned)
    }

    @Test("a warm target cache without a continuation speculates and matches greedy decode")
    func warmCacheWithoutContinuationSpeculates() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)

        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(
                tokens: MLXArray(prompt.dropFirst(cachedPrefixLength).map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, mainCache: cache,
            parameters: GenerateParameters(maxTokens: generatedTokenCount, temperature: 0),
            blockSize: Qwen35MTPFixture.blockSize)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == plain.tokens)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        #expect(iterator.acceptedCount > 0)
    }

    /// What an MTP decode with a carried continuation left: its tokens, where the drafter
    /// entries began and ended right after the prefill, and whether the stream speculated.
    private struct ContinuedDecode {
        let tokens: [Int]
        let drafterStartPosition: Int
        let drafterNextPosition: Int
        let drafterIsAligned: Bool
        let passthroughReason: String?
        let proposedCount: Int
    }

    /// A continuation whose drafter holds ``prompt`` and whose boundary witnesses
    /// `targetProcessedTokenCount` target tokens.
    private func continuation(
        _ fixture: Qwen35MTPFixture, targetProcessedTokenCount: Int
    ) throws -> MTPDrafterContinuation {
        let sampler = GenerateParameters(temperature: 0).sampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        return MTPDrafterContinuation(
            state: preparedState(fixture, hidden: hidden, sampler: sampler),
            targetBoundaryHidden: hidden[0..., (-1)..., 0...],
            targetProcessedTokenCount: targetProcessedTokenCount,
            drafterID: fixture.drafter.drafterID)
    }

    /// Decodes greedily with MTP after the tokens of ``prompt`` that `cache` does not hold,
    /// carrying `continuation`. The prompt is shorter than one prefill chunk, so the target
    /// prefills it without calling the chunk handler.
    private func continuedDecode(
        _ fixture: Qwen35MTPFixture, cache: [KVCache], cachedTokenCount: Int,
        continuation: MTPDrafterContinuation
    ) throws -> ContinuedDecode {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.dropFirst(cachedTokenCount).map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, mainCache: cache,
            mtpContinuation: continuation,
            parameters: GenerateParameters(maxTokens: generatedTokenCount, temperature: 0),
            blockSize: Qwen35MTPFixture.blockSize)
        let primed = try #require(iterator.drafterState)
        let drafterIsAligned = primed.isAligned
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return ContinuedDecode(
            tokens: tokens, drafterStartPosition: primed.cacheStartPosition,
            drafterNextPosition: primed.nextPosition, drafterIsAligned: drafterIsAligned,
            passthroughReason: iterator.passthroughReason, proposedCount: iterator.proposedCount)
    }

    @Test(
        "an unaligned continuation on a warm cache cold-starts the drafter at the cached position")
    func unalignedContinuationColdStartsAtCachedPosition() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)

        let decode = try continuedDecode(
            fixture, cache: cache, cachedTokenCount: cachedPrefixLength,
            continuation: try continuation(fixture, targetProcessedTokenCount: prompt.count))

        #expect(decode.passthroughReason == nil)
        #expect(decode.drafterStartPosition == cachedPrefixLength)
        #expect(decode.drafterNextPosition == prompt.count)
        #expect(decode.drafterIsAligned)
        #expect(decode.proposedCount > 0)
        #expect(decode.tokens == plain.tokens)
    }

    @Test("an unaligned continuation on an empty cache prefills the drafter from position zero")
    func unalignedContinuationPrefillsFromPositionZero() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)

        let decode = try continuedDecode(
            fixture, cache: try fixture.target.newCache(parameters: nil), cachedTokenCount: 0,
            continuation: try continuation(fixture, targetProcessedTokenCount: cachedPrefixLength))

        #expect(decode.passthroughReason == nil)
        #expect(decode.drafterStartPosition == 0)
        #expect(decode.drafterNextPosition == prompt.count)
        #expect(decode.drafterIsAligned)
        #expect(decode.proposedCount > 0)
        #expect(decode.tokens == plain.tokens)
    }
}

/// The MTP prefill of a prompt in chunks, each of which feeds the drafter, against the reference
/// prefill of the whole prompt in one forward.
@Suite(.serialized)
struct Qwen35MTPChunkedPrefillTests {

    /// A Qwen 3.5 target with its MTP drafter.
    enum Target: String, CaseIterable, CustomTestStringConvertible {

        /// The MLXLLM text target of ``Qwen35MTPFixture``.
        case text

        /// The MLXVLM target of ``Qwen35VLMMTPFixture``.
        case visionLanguage

        var testDescription: String { rawValue }

        /// The target and its drafter, on seeded weights.
        func models() throws -> (target: any LanguageModel, drafter: any MTPDrafterModel) {
            switch self {
            case .text:
                let fixture = try Qwen35MTPFixture()
                return (fixture.target, fixture.drafter)
            case .visionLanguage:
                let fixture = try Qwen35VLMMTPFixture()
                return (fixture.target, fixture.drafter)
            }
        }
    }

    /// What one MTP decode left: its tokens, the drafter cache right after the prefill, and
    /// whether it speculated.
    private struct MTPDecode {
        let tokens: [Int]
        let drafterCache: [MLXArray]
        let passthroughReason: String?
        let proposedCount: Int
    }

    /// Prompt tokens of the chunked prefill: more than 300, so a step of ``stepSize`` cuts
    /// several chunks.
    private let promptLength = 320

    /// The chunk ceiling of the chunked prefill.
    private let stepSize = 64

    /// The chunks the balanced prefill cuts from the 319 positions before the reserved logits
    /// position at a ceiling of 64: five chunks of at most 64.
    private let expectedChunkCount = 5

    /// Prompt tokens a warm target cache already holds before the continuation.
    private let cachedPrefixLength = 96

    /// Tokens each decode generates.
    private let generatedTokenCount = 8

    /// Prompt tokens of the prefill whose peak memory is measured.
    private let longPromptLength = 4096

    /// The chunk ceiling of the prefill whose peak memory is measured.
    private let longPromptStepSize = 512

    /// The seed of the prompt tokens.
    private let promptSeed: UInt64 = 7

    /// `length` prompt tokens of the fixture vocabulary, from ``promptSeed``.
    private func prompt(length: Int) -> MLXArray {
        seededTokens(length, vocabularySize: Qwen35MTPFixture.vocabularySize, seed: promptSeed)
    }

    /// Generation parameters of a greedy decode with `prefill`.
    private func parameters(prefill: PrefillParameters, maxTokens: Int) -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefill = prefill
        return parameters
    }

    /// Decodes greedily with MTP after `prompt`, on `cache` when it is given.
    private func decode(
        _ models: (target: any LanguageModel, drafter: any MTPDrafterModel), prompt: MLXArray,
        prefill: PrefillParameters, cache: [KVCache]? = nil, state: LMOutput.State? = nil
    ) throws -> MTPDecode {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: models.target, drafter: models.drafter,
            mainCache: cache, mainState: state,
            parameters: parameters(prefill: prefill, maxTokens: generatedTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        let drafterCache = try #require(iterator.drafterState).cache.flatMap(\.state)
        eval(drafterCache)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return MTPDecode(
            tokens: tokens, drafterCache: drafterCache,
            passthroughReason: iterator.passthroughReason, proposedCount: iterator.proposedCount)
    }

    /// A cache of `target` that holds `prefix`, and the model state of that cache.
    private func warmCache(_ target: any LanguageModel, prefix: MLXArray) throws -> (
        cache: [KVCache], state: LMOutput.State
    ) {
        let cache = try target.newCache(parameters: nil)
        var state: LMOutput.State?
        if case .logits(let output) = try target.prepare(
            LMInput(tokens: prefix), cache: cache, state: nil, prefill: .init())
        {
            state = output.state
        }
        return (cache, try #require(state))
    }

    @Test("a chunked MTP prefill matches the prefill in one forward", arguments: Target.allCases)
    func chunkedPrefillMatchesUnchunked(target: Target) throws {
        let models = try target.models()
        let prompt = prompt(length: promptLength)

        let unchunked = try decode(models, prompt: prompt, prefill: .init(chunking: .unchunked))
        let chunked = try decode(models, prompt: prompt, prefill: .init(stepSize: stepSize))

        #expect(chunked.tokens == unchunked.tokens)
        #expect(chunked.passthroughReason == nil)
        #expect(chunked.proposedCount > 0)
        try #require(chunked.drafterCache.count == unchunked.drafterCache.count)
        for (actual, expected) in zip(chunked.drafterCache, unchunked.drafterCache) {
            try #require(actual.shape == expected.shape)
            #expect(
                allClose(
                    actual, expected, rtol: drafterCacheTolerance, atol: drafterCacheTolerance
                ).item(Bool.self))
        }
    }

    @Test(
        "the target prepare calls the chunk handler once for each chunk", arguments: Target.allCases
    )
    func prepareCallsChunkHandlerForEachChunk(target: Target) throws {
        let models = try target.models()
        let prompt = prompt(length: promptLength)
        var chunks: [(tokens: [Int32], hiddenRows: Int)] = []
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpPrefillChunkHandlerKey] = { tokens, hiddenStates, _ in
            chunks.append((tokens.asArray(Int32.self), hiddenStates.dim(1)))
        }

        _ = try models.target.prepare(
            LMInput(tokens: prompt), cache: try models.target.newCache(parameters: nil),
            state: state, prefill: .init(stepSize: stepSize))

        #expect(chunks.count == expectedChunkCount)
        #expect(chunks.allSatisfy { $0.tokens.count == $0.hiddenRows })
        #expect(chunks.allSatisfy { $0.tokens.count <= stepSize })
        #expect(chunks.flatMap(\.tokens) == Array(prompt.asArray(Int32.self).dropLast()))
    }

    @Test("a warm VLM continuation prefilled in chunks speculates and matches greedy decode")
    func warmVLMContinuationSpeculates() throws {
        let fixture = try Qwen35VLMMTPFixture()
        let prompt = prompt(length: promptLength)
        let prefix = prompt[..<cachedPrefixLength]
        let suffix = prompt[cachedPrefixLength...]

        let plainStart = try warmCache(fixture.target, prefix: prefix)
        var plainIterator = try TokenIterator(
            input: LMInput(tokens: suffix), model: fixture.target, cache: plainStart.cache,
            state: plainStart.state,
            parameters: parameters(
                prefill: .init(stepSize: stepSize), maxTokens: generatedTokenCount))
        var plainTokens: [Int] = []
        while let token = plainIterator.next() {
            plainTokens.append(token)
        }

        let mtpStart = try warmCache(fixture.target, prefix: prefix)
        let mtp = try decode(
            (fixture.target, fixture.drafter), prompt: suffix, prefill: .init(stepSize: stepSize),
            cache: mtpStart.cache, state: mtpStart.state)

        #expect(mtp.tokens == plainTokens)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
    }

    @Test("a chunked MTP prefill of a long prompt peaks lower than the prefill in one forward")
    func chunkedPrefillPeaksLower() throws {
        let fixture = try Qwen35MTPFixture()
        let prompt = prompt(length: longPromptLength)

        let unchunked = try peakPrefillMemory(
            fixture, prompt: prompt, prefill: .init(chunking: .unchunked))
        let chunked = try peakPrefillMemory(
            fixture, prompt: prompt, prefill: .init(stepSize: longPromptStepSize))

        #expect(chunked < unchunked)
    }

    /// The peak memory of an MTP prefill of `prompt`, with the drafter state it builds.
    private func peakPrefillMemory(
        _ fixture: Qwen35MTPFixture, prompt: MLXArray, prefill: PrefillParameters
    ) throws -> Int {
        Memory.clearCache()
        Memory.peakMemory = 0
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters(prefill: prefill, maxTokens: 1),
            blockSize: Qwen35MTPFixture.blockSize)
        eval(try #require(iterator.drafterState).cache.flatMap(\.state))
        return Memory.peakMemory
    }
}

private func collectQwenChatDetails(
    _ stream: AsyncThrowingStream<Generation, Error>
) async throws -> (text: String, info: GenerateCompletionInfo) {
    var text = ""
    var info: GenerateCompletionInfo?
    for try await event in stream {
        text += event.chunk ?? ""
        if let completion = event.info { info = completion }
    }
    return (text, try #require(info))
}

@Suite(.serialized)
struct Qwen35MTPRegistrationTests {
    @Test
    func registrationsCreateTextAndVLMDrafters() async throws {
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8),
            modelType: "qwen3_5_text")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)

        await MLXVLM.Qwen35VLMMTPRegistration.register()

        let wrappedTextModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(wrappedTextModel is MLXLLM.Qwen35MTPDraftModel)

        let vlmModel = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.standaloneMTP().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standalone is MLXLLM.Qwen35MTPDraftModel)
        #expect(standalone.maximumBlockSize == 2)
        #expect(standalone.requiresPromptPrefill)
        #expect(!standalone.requiresSharedTargetKV)
        #expect(standalone.requiresGreedySampling)

        let standaloneVLM = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.standaloneMTP().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standaloneVLM is MLXVLM.Qwen35VLMNextNDraftModel)

        let textConfig = try JSONDecoder.json5().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let vlmConfig = try JSONDecoder.json5().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let textTarget = MLXLLM.Qwen35TextModel(textConfig)
        let vlmTarget = MLXVLM.Qwen35(vlmConfig)

        #expect(standalone.isCompatible(with: textTarget))
        #expect(!standalone.isCompatible(with: vlmTarget))
        #expect(standaloneVLM.isCompatible(with: vlmTarget))
        #expect(!standaloneVLM.isCompatible(with: textTarget))
    }

    @Test
    func registrationsAreIsolatedByTargetArchitecture() async throws {
        await MLXVLM.Qwen35VLMMTPRegistration.register()
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let vlmModel = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)
    }
}

@Suite(.serialized)
struct Qwen35CheckpointLoadingTests {
    @Test(arguments: [false, true])
    func convertedEmbeddedFactoryPreservesMixedPrecision(vision: Bool) async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: false, prefix: "language_model.mtp.", bits: 4)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let fc = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fc.bits == 8)
        #expect(
            context.model.modules().compactMap { $0 as? QuantizedLinear }.contains { $0.bits == 4 })
    }

    /// A head embedded in a target checkpoint reads only its own tensors; a standalone head
    /// reads every tensor.
    @Test(arguments: [false, true])
    func drafterDoesNotReadTargetTensors(vision: Bool) async throws {
        for standalone in [false, true] {
            let fixture = try await Qwen35MTPCheckpointFixture(
                vision: vision, standalone: standalone, prefix: standalone ? "" : "mtp.")
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let context = try await fixture.factory.load(
                from: fixture.directory, using: UnusedTokenizerLoader())

            let url = fixture.directory.appendingPathComponent("model.safetensors")
            let all = try loadModelCheckpoint(urls: [url]).weights.keys
            let read = try loadModelCheckpoint(
                urls: [url], excludedNamespaces: context.model.excludedCheckpointNamespaces
            ).weights.keys
            if standalone {
                #expect(Set(read) == Set(all))
            } else {
                #expect(all.contains("model.embed_tokens.weight"))
                #expect(Set(read) == Set(all.filter { $0.hasPrefix("mtp.") }))
            }
        }
    }

    @Test(arguments: [false, true])
    func targetLoaderExcludesConvertedMTPComponent(vision: Bool) async throws {
        let data = try Qwen35MTPCheckpointFixture.configuration(
            vision: vision, standalone: false, bits: nil)
        let target: any BaseLanguageModel
        if vision {
            target = MLXVLM.Qwen35(
                try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
        } else {
            target = MLXLLM.Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
        var weights = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        let expectedKeys = Set(weights.keys)
        weights["language_model.mtp.norm.weight"] = MLXArray.zeros([64])
        weights["language_model.mtp.fc.weight"] = MLXArray.zeros([64, 128])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        try await loadWeights(
            modelDirectory: directory, model: target, componentLoading: .immediate)
        #expect(Set(target.parameters().flattened().map(\.0)) == expectedKeys)
    }

    @Test
    func conversionRoundTripRetainsNormConvention() async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: false, standalone: false, prefix: "mtp.")
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: fixture.directory)
            try? FileManager.default.removeItem(at: output)
        }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        _ = try await convert(
            modelDirectory: fixture.directory, model: context.model, to: output,
            bits: 4, groupSize: 32)
        let (_, metadata) = try loadArraysAndMetadata(
            url: output.appendingPathComponent("model.safetensors"))
        #expect(metadata[Qwen35CheckpointPolicy.normMetadataKey] == "scale")
        let reloaded = try await fixture.factory.load(from: output, using: UnusedTokenizerLoader())
        let norm = try #require(
            reloaded.model.parameters().flattened().first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func expertParametersAndSettingsAreTransformedTogether() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp"
        var weights = [String: MLXArray]()
        for expert in 0 ..< 2 {
            for (parameter, shape) in [
                ("weight", [64, 8]), ("scales", [64, 2]), ("biases", [64, 2]),
            ] {
                weights["\(prefix).experts.\(expert).up_proj.\(parameter)"] = MLXArray.zeros(shape)
            }
        }
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).experts.0.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                    "\(prefix).experts.1.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                ]))
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.weight"]?.shape == [2, 64, 8])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.scales"]?.shape == [2, 64, 2])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.biases"]?.shape == [2, 64, 2])
        #expect(
            prepared.perLayerQuantization?.quantization(layer: "\(prefix).switch_mlp.up_proj")?.bits
                == 8)
        #expect(
            prepared.perLayerQuantization?.perLayerQuantization["\(prefix).experts.0.up_proj"]
                == nil)
    }

    @Test(arguments: [false, true], [4, 5, 8])
    func standaloneFactoryLoadsPublishedLayout(vision: Bool, bits: Int) async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: true, bits: bits)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let parameters = context.model.parameters().flattened()
        #expect(parameters.count == 31)
        let norm = try #require(parameters.first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
        let fc = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fc.bits == 8)
        let projections = context.model.modules().compactMap { $0 as? MLXNN.QuantizedLinear }
        #expect(projections.contains { $0.bits == bits })
    }

    @Test(arguments: [false, true], [false, true])
    func registeredFactoriesLoadQwen38Components(vision: Bool, standalone: Bool) async throws {
        await Qwen35TextMTPRegistration.register()
        await Qwen35VLMMTPRegistration.register()
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: standalone,
            prefix: standalone ? "" : "language_model.mtp.", bits: 4, family: "qwen3_8")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let factory = vision ? MTPDrafterModelFactory.visionLanguage : .shared
        let context = try await factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let parameters = context.model.parameters().flattened()
        #expect(parameters.count == 31)
        let norm = try #require(parameters.first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
        let fc = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fc.bits == 8)
        #expect(
            context.model.modules().compactMap { $0 as? QuantizedLinear }.contains { $0.bits == 4 })
    }

    @Test(arguments: ["qwen3_5", "qwen3_8"])
    func standaloneVLMRegistrationUsesOuterModelType(family: String) async throws {
        await Qwen35VLMMTPRegistration.register()
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: true, standalone: true, bits: 4, family: family)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await MTPDrafterModelFactory.visionLanguage.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        #expect(context.model is MLXVLM.Qwen35VLMNextNDraftModel)
        #expect(context.model.parameters().flattened().count == 31)
    }

    @Test(arguments: [false, true], ["mtp.", "language_model.mtp."])
    func embeddedFactoryLoadsComponentWithoutTargetWeights(vision: Bool, prefix: String)
        async throws
    {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: false, prefix: prefix)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let weights = Dictionary(uniqueKeysWithValues: context.model.parameters().flattened())
        #expect(weights.count == 15)
        #expect(weights["model.embed_tokens.weight"] == nil)
        #expect(weights["mtp.norm.weight"]?.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func targetMetadataDoesNotClassifyTheRawMTPShard() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let checkpoint = ModelCheckpoint(
            weights: ["mtp.norm.weight": MLXArray.zeros([64])],
            metadata: ["format": "mlx"], weightMetadata: ["mtp.norm.weight": [:]])
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(
            prepared.weights["mtp.norm.weight"]?.asArray(Float.self)
                == Array(repeating: 1, count: 64))
    }

    @Test
    func explicitNormConventionOverridesLegacyLayout() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let raw = ModelCheckpoint(
            weights: ["language_model.mtp.norm.weight": MLXArray.zeros([4])],
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "offset"])
        let prepared = try policy.prepare(raw, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(prepared.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
        let scale = ModelCheckpoint(
            weights: prepared.weights,
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "scale"])
        let repeated = try policy.prepare(scale, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(repeated.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
    }

    @Test
    func missingAndCompetingComponentsAreRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        for weights in [
            ["model.layers.0.weight": MLXArray(1)],
            ["mtp.norm.weight": MLXArray(1), "language_model.mtp.fc.weight": MLXArray(1)],
        ] {
            #expect(throws: CheckpointComponent.SelectionError.self) {
                try policy.prepare(.init(weights: weights), mtpNumHiddenLayers: 1, numExperts: 0)
            }
        }
    }

    @Test
    func conflictingAndUnknownNormMetadataIsRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let weights = [
            "mtp.norm.weight": MLXArray.zeros([4]),
            "mtp.layers.0.input_layernorm.weight": MLXArray.zeros([4]),
        ]
        for metadata in [
            ["mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "unknown"]],
            [
                "mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "scale"],
                "mtp.layers.0.input_layernorm.weight": [
                    Qwen35CheckpointPolicy.normMetadataKey: "offset"
                ],
            ],
        ] {
            #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
                try policy.prepare(
                    .init(weights: weights, weightMetadata: metadata), mtpNumHiddenLayers: 1,
                    numExperts: 0)
            }
        }
    }

    @Test
    func targetLayoutsShareComponentSelectionAndQuantizationMapping() throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray(1),
                "mtp.fc.weight": MLXArray(2),
                "lm_head.weight": MLXArray(3),
                "notmtp.weight": MLXArray(4),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "model.language_model.layers.0.self_attn.q_proj": .quantize(
                    .init(groupSize: 32, bits: 8))
            ]))
        for layout in [Qwen35CheckpointPolicy.TargetLayout.text, .wrappedText, .vision] {
            let prepared = try Qwen35CheckpointPolicy.prepareTarget(
                checkpoint, layout: layout, tiedWordEmbeddings: true)
            let prefix = layout == .text ? "model." : "language_model.model."
            let projection = prefix + "layers.0.self_attn.q_proj"
            #expect(prepared.weights[projection + ".weight"] != nil)
            #expect(prepared.perLayerQuantization?.quantization(layer: projection)?.bits == 8)
            #expect(prepared.weights.count == 2)
            #expect(prepared.weights["notmtp.weight"] != nil)
        }
    }

    @Test(arguments: ["qwen3_5", "qwen3_8"])
    func unknownAndMissingWeightsStillFailStrictFactoryLoading(family: String) async throws {
        for missing in [false, true] {
            let fixture = try await Qwen35MTPCheckpointFixture(
                vision: false, standalone: true, family: family)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let url = fixture.directory.appendingPathComponent("model.safetensors")
            var weights = try loadArrays(url: url)
            eval(Array(weights.values))
            if missing {
                weights.removeValue(forKey: "norm.weight")
            } else {
                weights["unexpected.weight"] = MLXArray(1)
            }
            try save(arrays: weights, metadata: ["format": "mlx"], url: url)
            await #expect(throws: (any Error).self) {
                try await fixture.factory.load(
                    from: fixture.directory, using: UnusedTokenizerLoader())
            }
        }
    }

    @Test
    func expertPrecisionCannotChangeDuringStacking() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp.experts"
        let weights = [
            "\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64]),
            "\(prefix).1.up_proj.weight": MLXArray.zeros([64, 64]),
        ]
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).0.up_proj": .quantize(.init(groupSize: 32, bits: 8))
                ]))
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        }
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(
                .init(weights: ["\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64])]),
                mtpNumHiddenLayers: 1, numExperts: 2)
        }
    }
}

/// Where a token of an MTP stream of the fixture sits in its round.
enum Qwen35MTPRoundSlot: CaseIterable, Sendable {

    /// An accepted draft. The verify pass of its round fed it to the main cache.
    case draft

    /// The verifier's correction after a rejected draft. No forward pass has fed it yet.
    case correction

    /// The verifier's bonus after every draft of its round was accepted. No forward pass has
    /// fed it yet.
    case bonus
}

/// How the first call of a continued decode ends.
enum Qwen35MTPFirstCall: CaseIterable, Sendable {

    /// The first call generates its whole token budget, over several MTP rounds.
    case complete

    /// The first call generates the prefill bonus only, and runs no MTP round.
    case withoutRounds

    /// The consumer stops reading the first call early, which cancels it.
    case cancelled

    /// The token budget of a complete first call: several rounds of ``Qwen35MTPFixture/blockSize``.
    private static let completeTokenBudget = 6

    /// The token budget of a first call that ends at its prefill bonus.
    private static let prefillBonusOnlyTokenBudget = 1

    /// The token budget of a cancelled first call: more than it generates before the
    /// cancellation reaches it.
    private static let cancelledTokenBudget = 24

    /// The tokens the consumer reads before it cancels the first call.
    private static let tokensReadBeforeCancellation = 2

    /// The tokens the first call may generate.
    var tokenBudget: Int {
        switch self {
        case .complete: Self.completeTokenBudget
        case .withoutRounds: Self.prefillBonusOnlyTokenBudget
        case .cancelled: Self.cancelledTokenBudget
        }
    }

    /// The tokens the consumer reads before it stops, or `nil` to read the whole stream.
    var consumedTokenCount: Int? {
        switch self {
        case .complete, .withoutRounds: nil
        case .cancelled: Self.tokensReadBeforeCancellation
        }
    }
}

/// The contract between the MTP iterator and the shared generation loop: the cache state that a
/// stop token reports, the drafter continuation that the token-task overload gives back, the
/// state the iterator exports, and plain decode for log probabilities.
@Suite(.serialized)
struct Qwen35MTPGenerateLoopContractTests {

    /// Prompt tokens of the fixture vocabulary.
    private let prompt = [3, 9, 14, 22, 5, 40]

    /// Tokens the slot classification decodes. Enough for each slot to hold a token that does
    /// not occur earlier in the stream.
    private let classifiedTokenCount = 32

    /// Prompt tokens a second call appends after the sequence the cache holds.
    private let followUpPrompt = [8, 30, 2]

    /// Tokens a second call generates.
    private let secondCallTokenCount = 8

    /// The top candidates a stream asks the log probabilities of.
    private let logProbabilityCandidateCount = 2

    /// Plain decode and the plain fallback of the MTP iterator run the same forward passes over
    /// the same cache, thus their log probabilities differ by float rounding at most.
    private let logProbabilityTolerance: Float = 1e-4

    /// The prompt as model input.
    private var promptInput: LMInput { LMInput(tokens: MLXArray(prompt.map(Int32.init))) }

    /// A greedy generation of `maxTokens` tokens.
    private func greedyParameters(maxTokens: Int) -> GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0)
    }

    /// The tokens a cache holds, as the iterator infers them.
    private func cachedTokenCount(_ cache: [KVCache]) -> Int {
        KVCacheStorage(cache, plan: .disabled).processedTokenCount
    }

    /// Decodes ``classifiedTokenCount`` greedy MTP tokens after ``prompt`` and names the slot
    /// of each, or `nil` for a token no round produced (the prefill bonus).
    private func classifiedTokens(_ fixture: Qwen35MTPFixture) throws
        -> [(token: Int, slot: Qwen35MTPRoundSlot?)]
    {
        var iterator = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: greedyParameters(maxTokens: classifiedTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        var classified: [(token: Int, slot: Qwen35MTPRoundSlot?)] = []
        var proposedCount = iterator.proposedCount
        var acceptedCount = iterator.acceptedCount
        var followsAcceptedDraft = false
        while let token = iterator.next() {
            let ranRound = iterator.proposedCount > proposedCount
            let acceptedDraft = iterator.acceptedCount > acceptedCount
            proposedCount = iterator.proposedCount
            acceptedCount = iterator.acceptedCount
            let roundSlot: Qwen35MTPRoundSlot = acceptedDraft ? .draft : .correction
            classified.append((token, ranRound ? roundSlot : followsAcceptedDraft ? .bonus : nil))
            followsAcceptedDraft = acceptedDraft
        }
        return classified
    }

    /// The first index of `slot` whose token does not occur earlier in `classified`, so that
    /// a stop on that token stops there.
    private func firstUnrepeatedIndex(
        of slot: Qwen35MTPRoundSlot, in classified: [(token: Int, slot: Qwen35MTPRoundSlot?)]
    ) -> Int? {
        classified.indices.first { index in
            classified[index].slot == slot
                && !classified[..<index].contains { $0.token == classified[index].token }
        }
    }

    @Test(
        "a stop token reports whether the main cache holds it",
        arguments: Qwen35MTPRoundSlot.allCases)
    func stopTokenReportsCacheState(slot: Qwen35MTPRoundSlot) async throws {
        let fixture = try Qwen35MTPFixture()
        let classified = try classifiedTokens(fixture)
        let stopIndex = try #require(firstUnrepeatedIndex(of: slot, in: classified))
        let stopToken = classified[stopIndex].token
        let cache = try fixture.target.newCache(parameters: nil)

        let decode = try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: promptInput, cache: cache,
                parameters: greedyParameters(maxTokens: classifiedTokenCount),
                context: fixture.makeModelContext(eosTokenIds: [stopToken]),
                mtpDrafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize))

        #expect(decode.tokens == classified.prefix(stopIndex).map(\.token))
        #expect(decode.info.stopReason == .stop)
        #expect(decode.info.stopTokenFedToCache == (slot == .draft ? stopToken : nil))
        let ledger =
            decode.info.promptTokenCount + decode.info.generationTokenCount
            + (decode.info.stopTokenFedToCache == nil ? 0 : 1)
        #expect(cachedTokenCount(cache) == ledger)
    }

    /// What a decode in two calls leaves: the whole prompt of the second call, the tokens the
    /// cache held between the calls, the continuation the first call gave back, and the second
    /// call's tokens.
    private struct ContinuedDecode {
        let sequence: [Int]
        let cachedTokenCount: Int
        let continuation: MTPDrafterContinuation?
        let second: Qwen35MTPFixture.GreedyDecode
    }

    /// Reads `stream` until it ends, or until it yielded `limit` tokens.
    private func read(_ stream: AsyncStream<TokenGeneration>, upTo limit: Int?) async {
        var received = 0
        for await event in stream {
            if case .token = event { received += 1 }
            if received == limit { break }
        }
    }

    /// Decodes after ``prompt`` in a first call, then continues the main cache and the drafter
    /// continuation of that call with ``followUpPrompt`` in a second call.
    private func continuedDecode(_ fixture: Qwen35MTPFixture, firstCall: Qwen35MTPFirstCall)
        async throws -> ContinuedDecode
    {
        let reference = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: firstCall.tokenBudget, speculative: true)
        let context = fixture.makeModelContext()
        let cache = try fixture.target.newCache(parameters: nil)
        let (firstStream, firstTask) = try generateTokensTask(
            input: promptInput, cache: cache,
            parameters: greedyParameters(maxTokens: firstCall.tokenBudget), context: context,
            mtpDrafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize)
        await read(firstStream, upTo: firstCall.consumedTokenCount)
        let continuation = await firstTask.value

        let cached = cachedTokenCount(cache)
        let sequence = Array((prompt + reference.tokens).prefix(cached)) + followUpPrompt
        let (secondStream, _) = try generateTokensTask(
            input: LMInput(tokens: MLXArray(sequence[cached...].map(Int32.init))), cache: cache,
            parameters: greedyParameters(maxTokens: secondCallTokenCount), context: context,
            mtpDrafter: fixture.drafter, mtpContinuation: continuation,
            blockSize: Qwen35MTPFixture.blockSize)
        return ContinuedDecode(
            sequence: sequence, cachedTokenCount: cached, continuation: continuation,
            second: try await Qwen35MTPFixture.collect(secondStream))
    }

    @Test(
        "a second call that continues the cache and the drafter matches a cold call and plain decode",
        arguments: Qwen35MTPFirstCall.allCases)
    func continuedCallMatchesColdCall(firstCall: Qwen35MTPFirstCall) async throws {
        let fixture = try Qwen35MTPFixture()
        let continued = try await continuedDecode(fixture, firstCall: firstCall)
        let cold = try await fixture.decodeGreedily(
            prompt: continued.sequence, maxTokens: secondCallTokenCount, speculative: true)
        let plain = try await fixture.decodeGreedily(
            prompt: continued.sequence, maxTokens: secondCallTokenCount, speculative: false)

        let continuation = try #require(continued.continuation)
        #expect(continuation.targetProcessedTokenCount == continued.cachedTokenCount)
        #expect(continued.second.tokens == cold.tokens)
        #expect(continued.second.tokens == plain.tokens)
        #expect(continued.second.info.passthroughReason == nil)
    }

    @Test("the exported state of an MTP stream saves to a prompt cache")
    func exportedStateSavesToPromptCache() throws {
        let fixture = try Qwen35MTPFixture()
        let cache = try fixture.target.newCache(parameters: nil)
        var iterator = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            mainCache: cache, parameters: greedyParameters(maxTokens: secondCallTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        while iterator.next() != nil {}

        // The exchange values the target hands the drafter are not arrays, so a prompt cache
        // cannot carry them; the exported state leaves them out.
        #expect(iterator.mainState?[mtpSharedKVStatesKey] != nil)
        #expect(throws: (any Error).self) {
            try preparePromptCacheSave(cache: cache, state: iterator.mainState)
        }
        _ = try preparePromptCacheSave(cache: cache, state: iterator.state)
    }

    /// Reads the log probabilities of each of the `count` tokens `iterator` generates.
    private func logProbabilities<Iterator: TokenIteratorProtocol>(
        of iterator: inout Iterator, count: Int
    ) throws -> [GenerateTokenLogProbabilities] {
        try (0 ..< count).map { _ in
            let token = iterator.next()
            try #require(token != nil)
            return try #require(iterator.lastLogProbabilities).materialize()
        }
    }

    @Test("a stream that asks for log probabilities decodes plainly and reports them")
    func logProbabilitiesDecodePlainly() throws {
        let fixture = try Qwen35MTPFixture()
        var parameters = greedyParameters(maxTokens: secondCallTokenCount)
        parameters.logProbabilities = logProbabilityCandidateCount
        var plain = try TokenIterator(
            input: promptInput, model: fixture.target, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters, blockSize: Qwen35MTPFixture.blockSize)

        let expected = try logProbabilities(of: &plain, count: secondCallTokenCount)
        let actual = try logProbabilities(of: &speculative, count: secondCallTokenCount)

        #expect(
            speculative.passthroughReason
                == MTPSpeculativeTokenIterator.logProbabilitiesReason)
        #expect(speculative.proposedCount == 0)
        #expect(actual.map(\.chosen.token) == expected.map(\.chosen.token))
        #expect(
            actual.map { $0.topLogProbabilities.map(\.token) }
                == expected.map { $0.topLogProbabilities.map(\.token) })
        for (reported, reference) in zip(actual, expected) {
            #expect(
                abs(reported.chosen.logProbability - reference.chosen.logProbability)
                    <= logProbabilityTolerance)
        }
    }
}

/// The compiled verify schedule of the Qwen 3.5 text target: a verify of two to
/// ``GatedDeltaDecodeKernel/mostTokens`` tokens runs as traced segments, as a decode step does,
/// and gives what the general path gives.
@Suite(.serialized)
struct Qwen35CompiledVerifyTests {

    /// The verify lengths of the compiled schedule: the bonus token and one draft, up to the most
    /// tokens the GDN kernel holds.
    static let verifyTokenCounts = Array(2 ... GatedDeltaDecodeKernel.mostTokens)

    /// The decoder layers of the test model: GDN, attention, GDN, attention. The schedule then
    /// holds each kind of segment: one that opens the step, one between two attention layers,
    /// and one that closes it.
    private static let hiddenLayers = 4

    /// The dtype of the test model: one the GDN kernel reads, so the verify takes snapshots.
    static let dtype = DType.bfloat16

    /// The seed of the weights of the test model.
    private static let weightSeed: UInt64 = 20_261_028

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_029

    /// The seed of the verify tokens.
    private static let verifySeed: UInt64 = 20_261_030

    /// The prompt tokens the cache holds before the verify. More than
    /// ``GatedDeltaDecodeKernel/mostTokens``, so the prefill itself takes the general path.
    private static let promptLength = 8

    /// The prompt of each greedy decode of these tests.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each greedy decode of these tests generates.
    private static let generatedTokenCount = 48

    /// What one verify leaves: its logits and hidden rows, the cache after its rollback, and
    /// the snapshot slots each GDN cache kept before the rollback.
    private struct Verify {
        let logits: MLXArray
        let hidden: MLXArray
        let states: [MLXArray]
        let offsets: [Int]
        let slotCounts: [Int]
    }

    /// A text target whose GDN layers the kernel runs, on seeded weights, in ``dtype``.
    static func kernelModel() throws -> Qwen35TextModel {
        let configuration = try GDNDecodeKernelFixture.configuration(
            Qwen35TextConfiguration.self, GDNDecodeKernelFixture.smallHeadCase,
            hiddenLayers: Self.hiddenLayers)
        let model = withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            Qwen35TextModel(configuration)
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(Self.dtype) })
        model.train(false)
        eval(model)
        return model
    }

    /// `count` seeded tokens of the vocabulary of `model`, shape `[count]`.
    private func tokens(_ count: Int, of model: Qwen35TextModel, seed: UInt64) -> MLXArray {
        seededTokens(count, vocabularySize: model.vocabularySize, seed: seed)
    }

    /// Prefills the prompt, runs one verify of `tokenCount` tokens on the compiled or the
    /// general path, then rolls the verify back by `rollback` tokens as a rejected round does.
    private func verify(
        _ model: Qwen35TextModel, tokenCount: Int, rollingBack rollback: Int, compiled: Bool
    ) throws -> Verify {
        let cache = try model.newCache(parameters: nil)
        let prompt = tokens(Self.promptLength, of: model, seed: Self.promptSeed)
        eval(model(LMInput.Text(tokens: prompt[.newAxis]), cache: cache, state: nil).logits)

        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        let output = model.forward(
            LMInput.Text(tokens: tokens(tokenCount, of: model, seed: Self.verifySeed)[.newAxis]),
            cache: cache, state: state, lastTokenOnly: false, useCompiledDecode: compiled)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        let slotCounts = cache.compactMap { ($0 as? MambaCache)?.speculativeSlotCount }

        if rollback > 0 {
            for entry in cache {
                if let mamba = entry as? MambaCache {
                    #expect(mamba.selectSpeculativeSlot(rollingBack: rollback))
                } else {
                    #expect(entry.trim(rollback) == rollback)
                }
            }
        }
        let states = cache.flatMap(\.state)
        eval(output.logits, hidden, states)
        return Verify(
            logits: output.logits, hidden: hidden, states: states, offsets: cache.map(\.offset),
            slotCounts: slotCounts)
    }

    /// Checks that `got` stays within `tolerance` of `expected`.
    private func expectWithin(
        _ got: MLXArray, _ expected: MLXArray, _ tolerance: Float, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == expected.shape, "\(what)", sourceLocation: sourceLocation)
        let gap = AttentionKernelTolerance.worstGap(got, expected)
        #expect(gap <= tolerance, "\(what): worst gap \(gap)", sourceLocation: sourceLocation)
    }

    @Test(
        "a compiled verify stays within tolerance of the general path, for each rollback",
        arguments: verifyTokenCounts)
    func compiledVerifyMatchesGeneralPath(tokenCount: Int) throws {
        let model = try Self.kernelModel()
        let tolerance = try #require(AttentionKernelTolerance.of(Self.dtype))

        for rollback in 0 ..< tokenCount {
            let general = try verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: false)
            let compiled = try verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: true)
            let round = "\(tokenCount) tokens, rollback \(rollback)"

            #expect(compiled.slotCounts.allSatisfy { $0 == tokenCount }, "\(round)")
            #expect(compiled.slotCounts == general.slotCounts, "\(round)")
            #expect(compiled.offsets == general.offsets, "\(round)")
            expectWithin(compiled.logits, general.logits, tolerance, "\(round) logits")
            expectWithin(compiled.hidden, general.hidden, tolerance, "\(round) hidden rows")
            try #require(compiled.states.count == general.states.count)
            for (index, (got, expected)) in zip(compiled.states, general.states).enumerated() {
                expectWithin(got, expected, tolerance, "\(round) cache state \(index)")
            }
        }
        #expect(model.model.compiledVerifySegmentCount > 0)
    }

    @Test("a verify on the general path compiles no verify segment")
    func generalPathCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        _ = try verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: false)
        #expect(model.model.compiledVerifySegmentCount == 0)
    }

    @Test("invalidating the compiled traces of the model drops its verify segments")
    func invalidationDropsVerifySegments() throws {
        let model = try Self.kernelModel()
        _ = try verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: true)
        #expect(model.model.compiledVerifySegmentCount > 0)

        model.invalidateCompiledTraces()

        #expect(model.model.compiledVerifySegmentCount == 0)
    }

    @Test("a verify shape holds two to the most kernel tokens")
    func verifyShapeBounds() {
        let most = GatedDeltaDecodeKernel.mostTokens
        #expect(CompiledVerifyShape(tokens: CompiledVerifyShape.fewestTokens) != nil)
        #expect(CompiledVerifyShape(tokens: most) != nil)
        #expect(CompiledVerifyShape(tokens: GatedDeltaDecodeKernel.decodeTokens) == nil)
        #expect(CompiledVerifyShape(tokens: most + 1) == nil)
    }

    @Test("snapshot slots keep their arrays through the trace arrays")
    func speculativeStatesRoundTripThroughTraceArrays() throws {
        let headCase = GDNDecodeKernelFixture.smallHeadCase
        let recurrent = MLXRandom.normal(
            [headCase.batch, headCase.valueHeads, headCase.valueHeadDim, headCase.keyHeadDim])
        let conv = MLXRandom.normal([headCase.batch, GDNDecodeKernelFixture.convTaps - 1, 1])
        let slots = stacked([recurrent, recurrent + 1])
        let window = concatenated([conv, conv], axis: 1)

        let snapshots = GatedDeltaDecodeKernel.Snapshots(
            traceArrays: GatedDeltaDecodeKernel.Snapshots(recurrent: slots, convWindow: window)
                .traceArrays)

        #expect(arrayEqual(snapshots.recurrent, slots).item(Bool.self))
        #expect(arrayEqual(snapshots.convWindow, window).item(Bool.self))
    }

    @Test(
        "greedy MTP decode of the text target equals plain decode through the compiled verify",
        arguments: Qwen35MTPFixture.DrafterMode.allCases)
    func textTargetMTPEqualsPlainDecode(mode: Qwen35MTPFixture.DrafterMode) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)

        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true)

        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
        #expect(fixture.target.model.compiledVerifySegmentCount > 0)
    }

    @Test("greedy MTP decode of the VLM target equals plain decode")
    func visionLanguageTargetMTPEqualsPlainDecode() throws {
        let fixture = try Qwen35VLMMTPFixture()
        let input = LMInput(tokens: MLXArray(Self.prompt.map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0)

        var plain = try TokenIterator(input: input, model: fixture.target, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters, blockSize: Qwen35MTPFixture.blockSize)
        var plainTokens: [Int] = []
        while let token = plain.next() {
            plainTokens.append(token)
        }
        var speculativeTokens: [Int] = []
        while let token = speculative.next() {
            speculativeTokens.append(token)
        }

        #expect(plainTokens.count == Self.generatedTokenCount)
        #expect(speculativeTokens == plainTokens)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
    }
}
