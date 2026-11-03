import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Bound for comparing a gated-delta-net state reached two ways. Wider on machines whose
/// float32 matmuls run as TF32; see ``MatmulPrecision``.
private let gdnStateTolerance = MatmulPrecision.tolerance(
    float32: ieeeFloat32Tolerance, reduced: tf32ReshapedMatmulTolerance)

/// Bound for comparing a gated-delta-net output computed with and without a checkpoint. Both
/// runs go through matmuls of the same shapes, so full float32 precision holds on every machine.
private let gdnCheckpointOutputTolerance = ieeeFloat32Tolerance

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
struct Qwen35MTPMetalTests {
    @Test
    func testQwen35MTPPredictorAdvancesEveryLayerCachePerToken() throws {
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 2).utf8))
        let predictor = MLXLLM.Qwen35MTPPredictor(configuration)
        let cache = predictor.newCache()
        let embeddings = MLXArray.zeros([1, 1, 16])
        let hidden = MLXArray.zeros([1, 1, 16])

        let first = predictor(
            inputsEmbeds: embeddings, hiddenStates: hidden, cache: cache,
            positionOffset: 128)
        eval(first)
        #expect(cache[0].offset == 1)
        #expect(cache[1].offset == 1)

        let second = predictor(
            inputsEmbeds: embeddings, hiddenStates: first, cache: cache,
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(configuration)
        let cache = try model.newCache(parameters: nil as GenerateParameters?)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        let output = model(input, cache: cache, state: state)

        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(output.state?[mtpSharedKVStatesKey])
        let sharedKVOffsets = try #require(output.state?[mtpSharedKVOffsetsKey])
        eval(output.logits, hidden)
        #expect(output.logits.shape == [1, 4, 16])
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(configuration)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let expected = model.model.forward(tokens, applyFinalNorm: false)
        let normalized = model.model.norm(expected)

        let emitted = try targetHiddenStates(model, tokens: tokens)
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35VLMEmitsPostFinalNormHiddenState() throws {
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let model = MLXVLM.Qwen35(configuration)
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(configuration)
        let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        let promptHidden = try targetHiddenStates(target, tokens: prompt)

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

                let verifyHidden = MLXArray.zeros([1, 2, configuration.hiddenSize])
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(configuration)
        let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        let promptHidden = try targetHiddenStates(target, tokens: prompt)

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
        let verifyHidden = MLXArray.zeros([1, 2, configuration.hiddenSize])
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
        let suffixHidden = MLXArray.zeros([1, 2, configuration.hiddenSize])
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(configuration)
        let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, configuration.hiddenSize])
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
        let verifyHidden = MLXArray.zeros([1, 2, configuration.hiddenSize])
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
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(configuration)
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(configuration)
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let hidden = MLXArray.zeros([1, 3, configuration.textConfiguration.hiddenSize])
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
                    [1, 2, configuration.textConfiguration.hiddenSize]),
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
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(configuration)
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
                model: MLXVLM.Qwen35VLMNextNDraftModel(configuration)))
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
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let layer = MLXLLM.Qwen35GatedDeltaNet(configuration)
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
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(configuration)
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
        let verifiedTokens = 3
        let attention = ZeroFilledKVCache.make(tokens: verifiedTokens)

        let recurrent = MambaCache()
        let checkpointConv = MLXArray.ones([1, 1, 4])
        let checkpointState = MLXArray.ones([1, 2, 2, 2])
        // The slot of a two-token verify: the states after its first token are the checkpoint.
        recurrent.saveSpeculativeSnapshots(
            GatedDeltaDecodeKernel.Snapshots(
                recurrent: stacked([checkpointState]),
                convWindow: concatenated(
                    [zeros(like: checkpointConv), checkpointConv, zeros(like: checkpointConv)],
                    axis: 1)))
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: 1)
        #expect(rewound == 1)
        #expect(attention.offset == verifiedTokens - 1)
        #expect(recurrent.speculativeSlotCount == 0)

        let restored = recurrent.state
        #expect(restored.count == 2)
        eval(restored[0], restored[1])
        #expect(allClose(restored[0], checkpointConv, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(restored[1], checkpointState, rtol: 0, atol: 0).item(Bool.self))
    }
}
