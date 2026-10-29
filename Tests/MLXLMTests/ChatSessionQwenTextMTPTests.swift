// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct ChatSessionQwenTextMTPTests {
    @Test(
        "MTP applies tool validation and authorization before automatic dispatch",
        arguments: ToolCallValidationPolicy.allCases, [true, false])
    func mtpValidatesToolsBeforeDispatch(
        validation: ToolCallValidationPolicy, declaresTool: Bool
    ) async throws {
        let probe = ToolRestartProbe()
        let tokenizer = ToolRestartMTPTokenizer(probe: probe)
        let processor = ToolRestartMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer)
        let restrictedTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string", "enum": ["Rome"]] as [String: any Sendable]
                    ],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(
                maxTokens: 24, temperature: 0,
                toolCallPolicy: .init(recovery: .disabled, validation: validation)),
            tools: declaresTool ? [restrictedTool] : nil,
            toolDispatch: { call in
                probe.recordDispatch(call)
                return "sunny"
            })

        do {
            _ = try await session.respond(to: "weather")
            #expect(declaresTool && validation == .permissive)
            #expect(probe.dispatchedToolCalls.count == 1)
            #expect(probe.dispatchedToolCalls.first?.function.arguments["city"] == .string("Paris"))
        } catch let error as RejectedToolCallError {
            #expect(!declaresTool || validation == .strict)
            #expect(error.rejection.reason == (declaresTool ? .invalidArguments : .undeclaredTool))
            #expect(probe.dispatchedToolCalls.isEmpty)
        }
    }

    @Test("single text turn reports MTP speculation")
    func singleTextTurnReportsMTPTelemetry() async throws {
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: parameters
        )
        let greedySession = ChatSession(
            makeModelContext(),
            generateParameters: parameters
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let greedyResult = try await collect(greedySession.streamDetails(to: "a"))

        #expect(result.text == greedyResult.text)
        let telemetry = try #require(result.info.speculativeDecodingTelemetry)
        #expect(telemetry.roundCount > 0)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.acceptedDraftTokens == result.info.proposedDraftTokens)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
        #expect(greedyResult.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("one-token MTP generation has no proposals and no session fallback")
    func oneTokenMTPGenerationDoesNotReportFallback() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 1, temperature: 0)
        )

        let result = try await collect(session.streamDetails(to: "a"))

        #expect(result.text == "x")
        #expect(result.info.generationTokenCount == 1)
        #expect(result.info.proposedDraftTokens == 0)
        #expect(result.info.acceptedDraftTokens == 0)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("cold MTP staging refusal falls back before target prefill")
    func coldStagingRefusalUsesOrdinaryGeneration() async throws {
        let probe = MTPStagingProbe()
        probe.cache.refusesSpeculativeStaging = true
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: parameters
        )
        let baseline = ChatSession(makeModelContext(), generateParameters: parameters)

        let result = try await collect(session.streamDetails(to: "a"))
        let expected = try await collect(baseline.streamDetails(to: "a"))
        let preparation = try #require(probe.preparations.first)
        let status = try await session.cacheStatus()

        #expect(result.text == expected.text)
        #expect(result.info.promptTokenCount == expected.info.promptTokenCount)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.speculativeDecodingFallbackReason == .unsupportedCache)
        #expect(probe.cache.refusedWidths == [2])
        #expect(probe.cacheCreationCount == 1)
        #expect(probe.preparations.count == 1)
        #expect(preparation.tokens == [7])
        #expect(preparation.cachedTokens.isEmpty)
        #expect(!preparation.emitsMTPState)
        #expect(probe.draftCount == 0)
        #expect(status.processedTokenCount == 5)
        #expect(probe.cache.offset == status.processedTokenCount)
    }

    @Test("warm MTP staging refusal preserves target cache and carried state")
    func warmStagingRefusalPreservesTargetContinuation() async throws {
        let probe = MTPStagingProbe()
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let firstStatus = try await session.cacheStatus()
        let firstProcessedTokenCount = try #require(firstStatus.processedTokenCount)
        let prefix = probe.cache.cachedTokens
        let draftsBeforeRefusal = probe.draftCount
        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(probe.finalizedPositions == [firstProcessedTokenCount])
        #expect(prefix == [7, 4, 4, 4])

        probe.cache.refusesSpeculativeStaging = true
        let second = try await collect(session.streamDetails(to: "b"))
        let status = try await session.cacheStatus()
        #expect(probe.preparations.count == 2)
        let preparation = try #require(probe.preparations.last)

        #expect(second.text == first.text)
        #expect(probe.cache.refusedWidths == [2])
        #expect(probe.cacheCreationCount == 1)
        #expect(preparation.cacheID == ObjectIdentifier(probe.cache))
        #expect(preparation.cachedTokens == prefix)
        #expect(preparation.tokens == [4, 7])
        #expect(preparation.targetMarker == 42)
        #expect(!preparation.emitsMTPState)
        #expect(Array(probe.cache.cachedTokens.prefix(prefix.count)) == prefix)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount == firstProcessedTokenCount)
        #expect(
            status.processedTokenCount
                == second.info.cachedPromptTokenCount + second.info.promptTokenCount
                + second.info.generationTokenCount)
        #expect(probe.cache.offset == status.processedTokenCount)
        #expect(probe.draftCount == draftsBeforeRefusal)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.acceptedDraftTokens == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unsupportedCache)
    }

    @Test("MTP preparation errors after cache mutation propagate without retry")
    func preparationFailureAfterCacheMutationDoesNotFallBack() async throws {
        let probe = MTPStagingProbe()
        probe.failAfterPreparationWrite = true
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        do {
            _ = try await collect(session.streamDetails(to: "a"))
            Issue.record("Expected the target preparation error")
        } catch let error as KVCacheError {
            #expect(error.message == "Target preparation failed after a cache write")
        }

        #expect(probe.preparations.count == 1)
        #expect(probe.preparations.first?.emitsMTPState == true)
        #expect(probe.cache.cachedTokens == [7])
        #expect(probe.cacheCreationCount == 1)
        #expect(probe.cache.refusedWidths.isEmpty)
        #expect(probe.draftCount == 0)
    }

    @Test("default sampling bypasses Qwen MTP before cache policy")
    func defaultSamplingUsesOrdinaryWarmCacheGeneration() async throws {
        // Keep GenerateParameters' default temperature (0.6), which Qwen MTP
        // cannot speculate with until stochastic acceptance is implemented.
        let parameters = GenerateParameters(maxTokens: 4)
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: parameters
        )
        let regularSession = ChatSession(
            makeModelContext(),
            generateParameters: parameters
        )

        _ = try await collect(session.streamDetails(to: "a"))
        _ = try await collect(regularSession.streamDetails(to: "a"))
        let result = try await collect(session.streamDetails(to: "b"))
        let regularResult = try await collect(regularSession.streamDetails(to: "b"))

        #expect(result.text == regularResult.text)
        #expect(result.info.promptTokenCount == regularResult.info.promptTokenCount)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == .unsupportedSampling)
        #expect(regularResult.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("a target-only sampling turn drops the continuation and the next turn cold-starts MTP")
    func samplingChangeDropsMTPContinuationWithoutDroppingTargetCache() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        session.generateParameters.temperature = 0.6
        let targetOnly = try await collect(session.streamDetails(to: "b"))
        session.generateParameters.temperature = 0
        let afterDrop = try await collect(session.streamDetails(to: "c"))

        #expect(targetOnly.info.promptTokenCount == 2)
        #expect(targetOnly.info.cachedPromptTokenCount > 0)
        #expect(targetOnly.info.proposedDraftTokens == nil)
        #expect(targetOnly.info.speculativeDecodingFallbackReason == .unsupportedSampling)
        // A target-only pass commits its final token directly, so the next
        // append has only the new user token left to prefill. With no
        // continuation to resume, the drafter cold-starts at the cached position.
        #expect(afterDrop.info.promptTokenCount == 1)
        #expect(afterDrop.info.cachedPromptTokenCount > 0)
        #expect((afterDrop.info.proposedDraftTokens ?? 0) > 0)
        #expect(afterDrop.info.passthroughReason == nil)
        #expect(afterDrop.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("a refused continuation cold-starts the drafter at the cached position")
    func refusedContinuationColdStartsDrafter() async throws {
        let probe = MTPStagingProbe()
        let session = ChatSession(
            makeModelContext(probe: probe),
            speculativeDecoding: try makeMTPConfiguration(model: QwenStyleMTPDrafter(probe: probe)),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        let firstProcessedTokenCount = try #require(
            try await session.cacheStatus().processedTokenCount)
        probe.refusesResumption = true
        let second = try await collect(session.streamDetails(to: "b"))

        // The first turn's prompt prefill is itself a cold start, at position zero.
        #expect(probe.coldStartPositions == [0, firstProcessedTokenCount])
        #expect(second.info.cachedPromptTokenCount == firstProcessedTokenCount)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("a warm Qwen cache without a continuation cold-starts MTP, then the next turn resumes")
    func qwenWarmCacheColdStartsThenResumes() async throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = GenerateParameters(maxTokens: 6, temperature: 0)
        let speculative = ChatSession(
            fixture.makeModelContext(), cache: try prefilledQwenCache(fixture),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: parameters)
        let plain = ChatSession(
            fixture.makeModelContext(), cache: try prefilledQwenCache(fixture),
            generateParameters: parameters)

        for prompt in [[2], [11]] {
            let text = fixture.text(for: prompt)
            let result = try await collect(speculative.streamDetails(to: text))
            let expected = try await collect(plain.streamDetails(to: text))

            #expect(result.text == expected.text)
            #expect(result.info.promptTokenCount == prompt.count)
            #expect((result.info.proposedDraftTokens ?? 0) > 0)
            #expect((result.info.acceptedDraftTokens ?? 0) > 0)
            #expect(result.info.passthroughReason == nil)
            #expect(result.info.speculativeDecodingFallbackReason == nil)
        }
    }

    @Test("a cache-plan change rebuilds target and MTP state together")
    func cachePlanChangeRebuildsMTPContinuation() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        session.generateParameters.maxKVSize = 64
        let rebuilt = try await collect(session.streamDetails(to: "b"))

        #expect(rebuilt.info.cachedPromptTokenCount == 0)
        #expect(rebuilt.info.promptTokenCount > 2)
        #expect((rebuilt.info.proposedDraftTokens ?? 0) > 0)
        #expect(rebuilt.info.passthroughReason == nil)
        #expect(rebuilt.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("second text turn resumes MTP from the warm prompt cache")
    func secondTextTurnResumesWarmMTP() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect(first.text == "xxxx")
        // The fixture's transcript render is append-only: the fourth generated
        // token and the new user token are the only uncached suffix.
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("stateless MTP reuses the warm main cache on the second text turn")
    func statelessMTPUsesWarmMainCache() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(requiresPromptPrefill: false),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        _ = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect(second.info.promptTokenCount < 13)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.passthroughReason == nil)
    }

    @Test("non-resumable stateful MTP preserves target cache and falls back target-only")
    func nonResumableStatefulMTPFallsBackWithoutRebuildingTarget() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(
                model: NonResumableMTPDrafter()),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.acceptedDraftTokens == nil)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(first.info.speculativeDecodingFallbackReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unavailableContinuation)
    }

    @Test("raw prompt cache keeps its prefix and cold-starts the MTP drafter")
    func rawPromptCacheColdStartsWithoutDiscardingPrefix() async throws {
        let prefixTokens = 3
        let rawCache = ZeroFilledKVCache.make(tokens: prefixTokens)
        let session = ChatSession(
            makeModelContext(),
            cache: [rawCache],
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let status = try await session.cacheStatus()

        #expect(result.info.promptTokenCount == 1)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
        // Existing raw prefix (3) + rendered fragment (1) + generated tokens (4), less the final
        // verifier sample a speculative pass leaves uncommitted.
        #expect(status.processedTokenCount == 7)
    }

    @Test(
        "prepared media reports the configured fallback and retains the main cache",
        arguments: [true, false], [Float(0), Float(0.6)])
    func preparedMediaFallsBackAndRetainsMainCache(
        speculationEnabled: Bool, temperature: Float
    ) async throws {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = PreparedMediaMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let configuration = try makeMTPConfiguration()
        let session = ChatSession(
            context,
            speculativeDecoding: speculationEnabled ? configuration : nil,
            generateParameters: .init(maxTokens: 4, temperature: temperature)
        )

        let result = try await collect(session.streamDetails(to: "a"))
        let status = try await session.cacheStatus()

        #expect(result.info.promptTokenCount == 1)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.acceptedDraftTokens == nil)
        #expect(result.info.speculativeDecodingTelemetry == nil)
        if speculationEnabled {
            #expect(
                result.info.speculativeDecodingFallbackReason
                    == (temperature == 0 ? .unsupportedMedia : .unsupportedSampling))
        } else {
            #expect(result.info.speculativeDecodingFallbackReason == nil)
        }
        #expect(status.processedTokenCount == 5)
    }

    @Test("prepared media discards an existing MTP continuation target-only")
    func preparedMediaDropsWarmMTPContinuation() async throws {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = SecondTurnMediaMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))
        let second = try await collect(session.streamDetails(to: "b"))

        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.promptTokenCount == 2)
        #expect(second.info.cachedPromptTokenCount > 0)
        #expect(second.info.proposedDraftTokens == nil)
        #expect(second.info.speculativeDecodingTelemetry == nil)
        #expect(first.info.speculativeDecodingFallbackReason == nil)
        #expect(second.info.speculativeDecodingFallbackReason == .unsupportedMedia)
    }

    @Test("consumer cancellation finalizes MTP before a clean next turn")
    func consumerCancellationLeavesCleanNextTurn() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 50, temperature: 0)
        )

        for try await event in session.streamDetails(to: "a") {
            if event.chunk != nil {
                break
            }
        }
        await session.synchronize()

        session.generateParameters = .init(maxTokens: 4, temperature: 0)
        let result = try await collect(session.streamDetails(to: "b"))
        let status = try await session.cacheStatus()

        #expect(result.text == "xxxx")
        // The cancelled assistant turn is not retained; the new user turn is
        // rendered from a clean transcript and cold-prefilled with MTP.
        #expect(result.info.promptTokenCount == 1)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.passthroughReason == nil)
        #expect(
            status.processedTokenCount
                == result.info.promptTokenCount + result.info.generationTokenCount - 1)
    }

    @Test("automatic tool restart reports MTP continuation eligibility", arguments: [true, false])
    func automaticToolRestartReportsMTPContinuation(resumable: Bool) async throws {
        let probe = ToolRestartProbe()
        let tokenizer = ToolRestartMTPTokenizer(probe: probe)
        let processor = ToolRestartMTPInputProcessor(tokenizer: tokenizer)
        let context = ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
        let weatherTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let drafter: any MTPDrafterModel =
            resumable ? QwenStyleMTPDrafter() : NonResumableMTPDrafter()
        let session = ChatSession(
            context,
            speculativeDecoding: try makeMTPConfiguration(model: drafter),
            generateParameters: .init(maxTokens: 24, temperature: 0),
            tools: [weatherTool],
            toolDispatch: { call in
                probe.recordDispatch(call)
                return #"{"forecast":"sunny"}"#
            }
        )

        let result = try await collectAll(session.streamDetails(to: "weather"))
        let status = try await session.cacheStatus()

        let dispatched = probe.dispatchedToolCalls
        #expect(dispatched.count == 1)
        let call = try #require(dispatched.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["city"] == .string("Paris"))

        let renderPasses = probe.renderPasses
        #expect(renderPasses.count == 2)
        let restart = try #require(renderPasses.count == 2 ? renderPasses[1] : nil)
        #expect(restart.map(\.role) == ["user", "assistant", "tool"])
        #expect(restart[0].content == "weather")
        #expect(restart[1].content.isEmpty)
        #expect(restart[1].toolCallNames == ["get_weather"])
        #expect(restart[2].content == #"{"forecast":"sunny"}"#)
        #expect(restart[2].toolResultName == "get_weather")

        #expect(result.infos.count == 2)
        let firstInfo = try #require(result.infos.first)
        let restartInfo = try #require(result.infos.last)
        #expect(firstInfo.promptTokenCount == 10)
        #expect(
            restartInfo.cachedPromptTokenCount
                == firstInfo.promptTokenCount + firstInfo.generationTokenCount - 1)
        for info in result.infos.prefix(resumable ? 2 : 1) {
            #expect(info.speculativeDecodingTelemetry != nil)
            #expect((info.proposedDraftTokens ?? 0) > 0)
            #expect(info.passthroughReason == nil)
            #expect(info.speculativeDecodingFallbackReason == nil)
        }
        if !resumable {
            #expect(restartInfo.speculativeDecodingTelemetry == nil)
            #expect(restartInfo.proposedDraftTokens == nil)
            #expect(restartInfo.acceptedDraftTokens == nil)
            #expect(restartInfo.passthroughReason == nil)
            #expect(restartInfo.speculativeDecodingFallbackReason == .unavailableContinuation)
        }
        // A speculative pass leaves its final verifier sample uncommitted.
        #expect(
            status.processedTokenCount
                == restartInfo.cachedPromptTokenCount + restartInfo.promptTokenCount
                + restartInfo.generationTokenCount - (resumable ? 1 : 0))
    }

    // MARK: - Checkpoints with MTP

    @Test(
        "a two-turn MTP chat takes checkpoints, writes the plain text, and resumes the drafter the restored checkpoint keeps"
    )
    func mtpChatTakesCheckpointsAndResumesTheRestoredDrafterContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let speculative = checkpointSession(
            fixture, speculativeDecoding: try fixture.makeSpeculativeDecodingConfig())
        let plain = checkpointSession(fixture, speculativeDecoding: nil)

        var lastTurn: GenerateCompletionInfo?
        for prompt in checkpointPrompts {
            let result = try await collect(speculative.streamDetails(to: prompt))
            let expected = try await collect(plain.streamDetails(to: prompt))

            #expect(result.text == expected.text)
            #expect(result.info.cachedPromptTokenCount == expected.info.cachedPromptTokenCount)
            #expect(result.info.promptTokenCount == expected.info.promptTokenCount)
            #expect((result.info.proposedDraftTokens ?? 0) > 0)
            #expect(result.info.passthroughReason == nil)
            #expect(result.info.speculativeDecodingFallbackReason == nil)
            lastTurn = result.info
        }

        // The second render closes the think block the first prompt opened, so the second turn
        // goes back to the checkpoint at the end of the first user message. The drafter resumes
        // the continuation that checkpoint keeps, which the first turn started at position zero,
        // instead of a cold start at the checkpoint.
        let second = try #require(lastTurn)
        #expect(second.cachedPromptTokenCount > 0)
        let continuation = try #require(await speculative.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == 0)
    }

    @Test(
        "an edit-and-regenerate turn writes the plain text and resumes the drafter from a copy of the checkpoint continuation"
    )
    func editAndRegenerateResumesACopyOfTheCheckpointContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let speculative = checkpointSession(
            fixture, speculativeDecoding: try fixture.makeSpeculativeDecodingConfig())
        let plain = checkpointSession(fixture, speculativeDecoding: nil)
        var history: [Chat.Message] = []
        for prompt in checkpointPrompts {
            let reply = try await collect(speculative.streamDetails(to: prompt))
            _ = try await collect(plain.streamDetails(to: prompt))
            history += [.user(prompt), .assistant(reply.text)]
        }
        let keptHistory = Array(history.prefix(messagesOfOneTurn))

        let speculativeEdit = await speculative.fork(history: keptHistory)
        let edit = try await collect(speculativeEdit.streamDetails(to: editPrompt))
        let expected = try await collect(
            plain.fork(history: keptHistory).streamDetails(to: editPrompt))

        #expect(edit.text == expected.text)
        #expect(edit.info.cachedPromptTokenCount == expected.info.cachedPromptTokenCount)
        #expect(edit.info.cachedPromptTokenCount > 0)
        #expect((edit.info.proposedDraftTokens ?? 0) > 0)
        #expect(edit.info.passthroughReason == nil)
        #expect(edit.info.speculativeDecodingFallbackReason == nil)
        // The restored checkpoint keeps the continuation the first turn started at position zero.
        let continuation = try #require(await speculativeEdit.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == 0)
        // The edit resumed a copy: the checkpoints the source session keeps are as they were.
        for checkpoint in await speculative.checkpointSummaries() {
            let stored = try #require(checkpoint.drafterContinuation)
            #expect(
                stored.isAligned(
                    withMainTokenCount: checkpoint.tokenCount, drafterID: fixture.drafter.drafterID)
            )
        }
    }

    @Test("after a target-only turn, the next turn of a chat that takes checkpoints runs MTP again")
    func mtpRunsAgainAfterATargetOnlyCheckpointTurn() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = checkpointSession(
            fixture, speculativeDecoding: try fixture.makeSpeculativeDecodingConfig())

        _ = try await collect(session.streamDetails(to: checkpointPrompts[0]))
        session.generateParameters.logProbabilities = targetOnlyLogProbabilities
        let targetOnly = try await collect(session.streamDetails(to: checkpointPrompts[1]))
        session.generateParameters.logProbabilities = nil
        let recovered = try await collect(session.streamDetails(to: checkpointFollowUpPrompt))

        #expect(
            targetOnly.info.passthroughReason
                == MTPSpeculativeTokenIterator.logProbabilitiesReason)
        #expect(targetOnly.info.proposedDraftTokens == 0)
        #expect(recovered.info.cachedPromptTokenCount > 0)
        #expect((recovered.info.proposedDraftTokens ?? 0) > 0)
        #expect(recovered.info.passthroughReason == nil)
        #expect(recovered.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("an edit that rewinds a trimmable target keeps MTP on and cold-starts the drafter")
    func trimmableTargetRewindKeepsMTP() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )
        let first = try await collect(session.streamDetails(to: "a"))
        _ = try await collect(session.streamDetails(to: "b"))

        // The render of the kept turn is the common prefix: the edit leaves the second user
        // message, thus the cache rewinds to the end of the first reply.
        let edit = try await collect(
            session.fork(history: [.user("a"), .assistant(first.text)])
                .streamDetails(to: "xb"))

        #expect(edit.info.cachedPromptTokenCount == 1 + first.text.count)
        #expect((edit.info.proposedDraftTokens ?? 0) > 0)
        #expect(edit.info.passthroughReason == nil)
        #expect(edit.info.speculativeDecodingFallbackReason == nil)
    }

    @Test("each checkpoint of an MTP turn keeps a drafter continuation aligned at its position")
    func mtpCheckpointsKeepAlignedDrafterContinuations() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = checkpointSession(
            fixture, speculativeDecoding: try fixture.makeSpeculativeDecodingConfig())

        _ = try await collect(session.streamDetails(to: checkpointPrompts[0]))
        let checkpoints = await session.checkpointSummaries()

        // The end of the instructions, the end of the user message, and the end of the prompt.
        #expect(checkpoints.count == checkpointsOfAFirstTurn)
        for checkpoint in checkpoints {
            let continuation = try #require(checkpoint.drafterContinuation)
            #expect(
                continuation.isAligned(
                    withMainTokenCount: checkpoint.tokenCount, drafterID: fixture.drafter.drafterID)
            )
            #expect(checkpoint.byteCount > continuation.byteCount)
        }
    }

    @Test("a checkpoint keeps a compact copy of the drafter continuation and counts its bytes")
    func checkpointKeepsACompactDrafterContinuation() throws {
        let live = makeLiveDrafterContinuation(aheadBy: 0)
        let withoutDrafter = try #require(
            PromptCacheCheckpoint(caches: [live.mainCache], tokens: live.tokens, state: nil))
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: [live.mainCache], tokens: live.tokens, state: nil,
                drafterContinuation: live.continuation))
        let stored = try #require(checkpoint.drafterContinuation)

        #expect(checkpoint.byteCount == withoutDrafter.byteCount + stored.byteCount)
        #expect(stored.byteCount == live.heldByteCount)
        let storedArrays = stored.state.cache.flatMap(\.state) + [stored.targetBoundaryHidden]
        let liveArrays =
            live.continuation.state.cache.flatMap(\.state) + [
                live.continuation.targetBoundaryHidden
            ]
        #expect(storedArrays.count == liveArrays.count)
        for (storedArray, liveArray) in zip(storedArrays, liveArrays) {
            #expect(bufferAddress(of: storedArray) != bufferAddress(of: liveArray))
        }
    }

    @Test("a checkpoint keeps no drafter continuation that ends at another position")
    func checkpointDropsAMisalignedDrafterContinuation() throws {
        let live = makeLiveDrafterContinuation(aheadBy: 1)
        let withoutDrafter = try #require(
            PromptCacheCheckpoint(caches: [live.mainCache], tokens: live.tokens, state: nil))
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: [live.mainCache], tokens: live.tokens, state: nil,
                drafterContinuation: live.continuation))

        #expect(checkpoint.drafterContinuation == nil)
        #expect(checkpoint.byteCount == withoutDrafter.byteCount)
    }

    @Test("a checkpoint hands its drafter continuation to the same drafter as a copy of its own")
    func checkpointHandsOutACopyOfItsDrafterContinuation() throws {
        let live = makeLiveDrafterContinuation(aheadBy: 0)
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: [live.mainCache], tokens: live.tokens, state: nil,
                drafterContinuation: live.continuation))
        let stored = try #require(checkpoint.drafterContinuation)

        let restored = try #require(
            checkpoint.drafterContinuation(resumableBy: live.continuation.drafterID))

        #expect(
            restored.isAligned(
                withMainTokenCount: live.tokens.count, drafterID: live.continuation.drafterID))
        // A generation writes into the drafter caches it resumes, which must leave the stored
        // continuation aligned for the next restore.
        let drafterRow = MLXRandom.normal([1, liveDrafterHeads, 1, liveDrafterHeadWidth])
        for cache in restored.state.cache {
            _ = cache.update(keys: drafterRow, values: drafterRow)
        }
        #expect(
            stored.isAligned(
                withMainTokenCount: live.tokens.count, drafterID: live.continuation.drafterID))
    }

    @Test("a checkpoint hands no drafter continuation to another drafter")
    func checkpointHandsNoDrafterContinuationToAnotherDrafter() throws {
        let live = makeLiveDrafterContinuation(aheadBy: 0)
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: [live.mainCache], tokens: live.tokens, state: nil,
                drafterContinuation: live.continuation))

        #expect(checkpoint.drafterContinuation != nil)
        #expect(checkpoint.drafterContinuation(resumableBy: MTPDrafterID("another-drafter")) == nil)
    }

    // MARK: - Incompatible drafter and memory policy

    @Test("an incompatible MTP drafter decodes plainly with a fallback reason instead of throwing")
    func incompatibleDrafterFallsBackToPlainDecode() async throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = GenerateParameters(maxTokens: fixtureTurnTokenCount, temperature: 0)
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(model: try makeNarrowDrafter()),
            generateParameters: parameters)
        let plain = ChatSession(fixture.makeModelContext(), generateParameters: parameters)
        let prompt = fixture.text(for: fixturePrompt)

        let result = try await collect(session.streamDetails(to: prompt))
        let expected = try await collect(plain.streamDetails(to: prompt))

        #expect(result.text == expected.text)
        #expect(result.info.speculativeDecodingFallbackReason == .incompatibleDrafter)
        #expect(result.info.proposedDraftTokens == nil)
        #expect(result.info.speculativeDecodingTelemetry == nil)
    }

    @Test("validating an incompatible MTP drafter against its target throws")
    func incompatibleDrafterValidationThrows() async throws {
        let fixture = try Qwen35MTPFixture()
        let target = fixture.makeModelContainer()
        let narrow = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration, model: try makeNarrowDrafter()))
        let matching = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration, model: fixture.drafter))

        await #expect(throws: MTPDrafterCompatibilityError.self) {
            try await narrow.validateCompatibility(with: target)
        }
        try await matching.validateCompatibility(with: target)
    }

    @Test("a memory policy that does not admit the MTP drafter falls back to plain decode")
    func memoryPolicyFallbackDecodesPlainly() async throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = GenerateParameters(maxTokens: fixtureTurnTokenCount, temperature: 0)
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(
                memoryPolicy: .init(limitBytes: rejectingMemoryLimit)),
            generateParameters: parameters)
        let plain = ChatSession(fixture.makeModelContext(), generateParameters: parameters)
        let prompt = fixture.text(for: fixturePrompt)

        let result = try await collect(session.streamDetails(to: prompt))
        let expected = try await collect(plain.streamDetails(to: prompt))

        #expect(result.text == expected.text)
        #expect(result.info.speculativeDecodingFallbackReason == .memoryBudgetExceeded)
        #expect(result.info.proposedDraftTokens == nil)
    }

    @Test("a failing memory policy throws with the target and drafter weight bytes")
    func memoryPolicyFailureThrows() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(
                memoryPolicy: .init(limitBytes: rejectingMemoryLimit, action: .fail)),
            generateParameters: GenerateParameters(
                maxTokens: fixtureTurnTokenCount, temperature: 0))

        let error = await #expect(throws: SpeculativeDecodingMemoryError.self) {
            _ = try await collect(session.streamDetails(to: fixture.text(for: fixturePrompt)))
        }

        #expect(
            error?.evaluation.mainModelBytes
                == SpeculativeDecodingMemoryPolicy.modelWeightBytes(fixture.target))
        #expect(
            error?.evaluation.draftModelBytes
                == SpeculativeDecodingMemoryPolicy.modelWeightBytes(fixture.drafter))
    }

    @Test("the memory policy counts the drafter continuation the session carries")
    func memoryPolicyCountsTheCarriedContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let weightBytes =
            SpeculativeDecodingMemoryPolicy.modelWeightBytes(fixture.target)
            + SpeculativeDecodingMemoryPolicy.modelWeightBytes(fixture.drafter)
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(
                memoryPolicy: .init(limitBytes: weightBytes)),
            generateParameters: GenerateParameters(
                maxTokens: fixtureTurnTokenCount, temperature: 0))

        let first = try await collect(session.streamDetails(to: fixture.text(for: fixturePrompt)))
        let second = try await collect(
            session.streamDetails(to: fixture.text(for: fixtureFollowUpPrompt)))

        #expect(first.info.speculativeDecodingFallbackReason == nil)
        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.speculativeDecodingFallbackReason == .memoryBudgetExceeded)
        #expect(second.info.proposedDraftTokens == nil)
    }

    // MARK: - Prompt cache files with MTP

    @Test(
        "a session restored from a saved prompt cache resumes the saved drafter continuation and writes the text of the source"
    )
    func restoredPromptCacheResumesTheSavedDrafterContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let source = try await savedRawCacheSession(fixture)
        defer { try? FileManager.default.removeItem(at: source.url) }
        let saved = try #require(await source.session.drafterContinuation())
        let restored = try restoredMTPSession(
            fixture, promptCache: try loadPromptCacheSnapshot(url: source.url))

        let cachedPosition = try await expectFollowUpTurn(
            of: restored, writesTheTextOf: source.session, fixture)

        // A resume keeps the entries the source drafter began at the prefilled prefix. A cold
        // start begins them at the position this turn found cached.
        let continuation = try #require(await restored.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == saved.state.cacheStartPosition)
        #expect(continuation.state.cacheStartPosition < cachedPosition)
    }

    @Test(
        "a session restored from a prompt cache file saved without a drafter continuation cold-starts the drafter and writes the text of the source"
    )
    func restoredPromptCacheWithoutADrafterContinuationColdStarts() async throws {
        let fixture = try Qwen35MTPFixture()
        let source = try await savedRawCacheSession(fixture)
        defer { try? FileManager.default.removeItem(at: source.url) }
        // The file a save wrote before drafter continuations: the main caches and model state.
        let saved = try loadPromptCacheSnapshot(url: source.url)
        let legacyURL = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: legacyURL) }
        try savePromptCache(url: legacyURL, cache: saved.cache, state: saved.state)
        let restored = try restoredMTPSession(
            fixture, promptCache: try loadPromptCacheSnapshot(url: legacyURL))

        let cachedPosition = try await expectFollowUpTurn(
            of: restored, writesTheTextOf: source.session, fixture)

        try await expectColdStart(of: restored, at: cachedPosition)
    }

    @Test(
        "a session restored with a damaged or foreign drafter continuation cold-starts the drafter and writes the text of the source",
        arguments: [
            MTPDrafterContinuationTamper.drafterID, .formatVersion, .boundaryHidden, .cacheBatch,
        ])
    func restoredPromptCacheWithADamagedDrafterContinuationColdStarts(
        tamper: MTPDrafterContinuationTamper
    ) async throws {
        let fixture = try Qwen35MTPFixture()
        let source = try await savedRawCacheSession(fixture)
        defer { try? FileManager.default.removeItem(at: source.url) }
        let saved = try loadPromptCacheSnapshot(url: source.url)
        let serialized = try #require(saved.drafterContinuation)
        var arrays = serialized.arrays
        var metadata = serialized.metadata
        try tamper.apply(arrays: &arrays, metadata: &metadata)
        let restored = try restoredMTPSession(
            fixture,
            promptCache: PromptCacheSnapshot(
                cache: saved.cache, metadata: saved.metadata, state: saved.state,
                drafterContinuation: SerializedMTPDrafterContinuation(
                    arrays: arrays, metadata: metadata)))

        let cachedPosition = try await expectFollowUpTurn(
            of: restored, writesTheTextOf: source.session, fixture)

        try await expectColdStart(of: restored, at: cachedPosition)
    }

    /// A greedy MTP session over ``prefilledQwenCache(_:)`` that wrote one turn of
    /// ``fixturePrompt``, and the prompt cache file it saved after that turn.
    ///
    /// - Parameter fixture: The Qwen fixture.
    /// - Returns: The session and the file. The caller removes the file.
    private func savedRawCacheSession(_ fixture: Qwen35MTPFixture) async throws -> (
        session: ChatSession, url: URL
    ) {
        let session = ChatSession(
            fixture.makeModelContext(), cache: try prefilledQwenCache(fixture),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: GenerateParameters(
                maxTokens: fixtureTurnTokenCount, temperature: 0))
        _ = try await collect(session.streamDetails(to: fixture.text(for: fixturePrompt)))
        let url = TemporaryPromptCacheFile.makeURL()
        try await session.saveCache(to: url)
        return (session, url)
    }

    /// A greedy MTP session over the Qwen fixture that starts from `promptCache`.
    ///
    /// - Parameters:
    ///   - fixture: The Qwen fixture.
    ///   - promptCache: The snapshot the session starts from.
    /// - Returns: The session.
    private func restoredMTPSession(
        _ fixture: Qwen35MTPFixture, promptCache: PromptCacheSnapshot
    ) throws -> ChatSession {
        ChatSession(
            fixture.makeModelContext(), promptCache: promptCache,
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: GenerateParameters(
                maxTokens: fixtureTurnTokenCount, temperature: 0))
    }

    /// Writes the turn of ``fixtureFollowUpPrompt`` in `restored` and in `source`, and checks
    /// that the turn of `restored` speculated throughout and wrote the text of `source`.
    ///
    /// - Parameters:
    ///   - restored: The session restored from a prompt cache of `source`.
    ///   - source: The session that was never saved and restored, at the same cached position.
    ///   - fixture: The Qwen fixture.
    /// - Returns: The tokens the caches of `restored` held before the turn.
    private func expectFollowUpTurn(
        of restored: ChatSession, writesTheTextOf source: ChatSession,
        _ fixture: Qwen35MTPFixture
    ) async throws -> Int {
        let cachedPosition = try #require(try await restored.cacheStatus().processedTokenCount)
        #expect(cachedPosition == (try await source.cacheStatus().processedTokenCount))
        let followUp = fixture.text(for: fixtureFollowUpPrompt)

        let expected = try await collect(source.streamDetails(to: followUp))
        let result = try await collect(restored.streamDetails(to: followUp))

        #expect(result.text == expected.text)
        #expect(result.info.promptTokenCount == expected.info.promptTokenCount)
        #expect((result.info.proposedDraftTokens ?? 0) > 0)
        #expect(result.info.passthroughReason == nil)
        #expect(result.info.speculativeDecodingFallbackReason == nil)
        return cachedPosition
    }

    /// Checks that the drafter of the last turn of `session` cold-started at `position`.
    ///
    /// - Parameters:
    ///   - session: The session whose last turn speculated.
    ///   - position: The cached position the turn started at.
    private func expectColdStart(of session: ChatSession, at position: Int) async throws {
        let continuation = try #require(await session.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == position)
    }

    /// Tokens each turn of a Qwen fixture chat generates.
    private let fixtureTurnTokenCount = 6

    /// The prompt tokens of a one-turn Qwen fixture chat.
    private let fixturePrompt = [3, 9, 14]

    /// The prompt tokens of the second turn of a Qwen fixture chat.
    private let fixtureFollowUpPrompt = [11, 2]

    /// A memory budget no model fits in.
    private let rejectingMemoryLimit = 1

    /// The instructions of a Qwen fixture chat that takes checkpoints.
    private let checkpointInstructions = "answer briefly"

    /// The user turns of a Qwen fixture chat that takes checkpoints.
    private let checkpointPrompts = ["first", "second"]

    /// The checkpoints a first turn with instructions takes: at the end of the instructions, at
    /// the end of the user message, and at the end of the prompt.
    private let checkpointsOfAFirstTurn = 3

    /// The user turn that follows ``checkpointPrompts`` in a Qwen fixture chat.
    private let checkpointFollowUpPrompt = "third"

    /// The user turn an edit-and-regenerate writes in place of the second of ``checkpointPrompts``.
    private let editPrompt = "edited"

    /// The messages one turn adds to a history: the user message and the reply.
    private let messagesOfOneTurn = 2

    /// A log-probability request, which an MTP stream decodes without speculation, thus a turn
    /// with it runs target-only and leaves no drafter continuation.
    private let targetOnlyLogProbabilities = 0

    /// A greedy session over the Qwen fixture target with ``checkpointInstructions``, whose
    /// template writes a Qwen-like transcript: the generation prompt opens a think block that the
    /// next render closes, so the next turn goes back to a checkpoint.
    private func checkpointSession(
        _ fixture: Qwen35MTPFixture, speculativeDecoding: SpeculativeDecodingConfig?
    ) -> ChatSession {
        let tokenizer = ChatSessionRecurrentCheckpointTests.CheckpointTemplateTokenizer(
            historyShape: .closesTheThinkBlock)
        let context = ModelContext(
            configuration: fixture.modelConfiguration, model: fixture.target,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: fixture.modelConfiguration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        return ChatSession(
            context, instructions: checkpointInstructions,
            speculativeDecoding: speculativeDecoding,
            generateParameters: GenerateParameters(
                maxTokens: fixtureTurnTokenCount, temperature: 0))
    }

    /// A Qwen MTP drafter whose hidden state is narrower than the fixture target's, thus it
    /// cannot draft for that target.
    private func makeNarrowDrafter() throws -> Qwen35MTPDraftModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: Qwen35MTPFixture.mtpLayers,
                    hiddenSize: Qwen35FixtureJSON.defaultHiddenSize,
                    vocabularySize: Qwen35MTPFixture.vocabularySize
                ).utf8))
        return Qwen35MTPDraftModel(configuration)
    }

    /// Drafter cache rows of ``makeLiveDrafterContinuation(aheadBy:)``: far fewer than the
    /// 256-row buffer `KVCacheSimple` allocates, so a view of them holds a much larger buffer.
    private let liveDrafterRows = 3

    /// Heads of the live drafter cache. With ``liveDrafterHeadWidth`` the buffer outgrows the
    /// view by more than the 16 KiB `contiguous` leaves shared.
    private let liveDrafterHeads = 4

    /// The width of each head of the live drafter cache.
    private let liveDrafterHeadWidth = 64

    /// Target hidden rows of the prefill chunk whose last row is the live boundary row.
    private let liveChunkRows = 32

    /// The width of the live target hidden rows.
    private let liveHiddenWidth = 1024

    /// A live drafter continuation as a prefill chunk leaves it, beside a main cache that holds
    /// the tokens of a checkpoint.
    ///
    /// The drafter cache holds its rows in the front of a larger buffer, and the boundary row is
    /// a slice of the whole chunk, as in a real feed.
    ///
    /// - Parameter offset: How many positions past the checkpoint the continuation ends; zero
    ///   aligns it with the checkpoint.
    /// - Returns: The main cache, the tokens it holds, the continuation, and the bytes of the
    ///   rows the continuation holds: the drafter keys and values and the boundary row.
    private func makeLiveDrafterContinuation(aheadBy offset: Int) -> (
        mainCache: KVCache, tokens: [Int], continuation: MTPDrafterContinuation,
        heldByteCount: Int
    ) {
        let tokens = Array(0 ... liveDrafterRows)
        let mainCache = RotatingKVCache(maxSize: tokens.count)
        let mainRows = MLXRandom.normal([1, 1, tokens.count, liveDrafterHeadWidth])
        _ = mainCache.update(keys: mainRows, values: mainRows)
        let drafterCache = KVCacheSimple()
        let drafterRows = MLXRandom.normal(
            [1, liveDrafterHeads, liveDrafterRows, liveDrafterHeadWidth])
        _ = drafterCache.update(keys: drafterRows, values: drafterRows)
        let chunkHidden = MLXRandom.normal([1, liveChunkRows, liveHiddenWidth])
        eval(mainCache.state + drafterCache.state + [chunkHidden])
        let continuation = MTPDrafterContinuation(
            state: MTPDrafterState(cache: [drafterCache], nextPosition: liveDrafterRows + offset),
            targetBoundaryHidden: chunkHidden[0..., (liveChunkRows - 1)..., 0...],
            targetProcessedTokenCount: tokens.count + offset,
            drafterID: MTPDrafterID("checkpoint-compaction-test"))
        let keysAndValues = 2
        let heldByteCount =
            keysAndValues * drafterRows.nbytes + chunkHidden.nbytes / liveChunkRows
        return (mainCache, tokens, continuation, heldByteCount)
    }

    /// The address of the first byte of `array`, to tell whether two arrays share a buffer.
    private func bufferAddress(of array: MLXArray) -> Int? {
        array.asData(access: .noCopy).data.withUnsafeBytes { bytes in
            bytes.baseAddress.map { Int(bitPattern: $0) }
        }
    }

    private func makeModelContext(probe: MTPStagingProbe? = nil) -> ModelContext {
        let tokenizer = DeterministicMTPTokenizer()
        let processor = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        return ModelContext(
            configuration: processor.configuration,
            model: MTPStateEmittingTarget(probe: probe),
            processor: processor,
            tokenizer: tokenizer
        )
    }

    private func makeMTPConfiguration(
        requiresPromptPrefill: Bool = true
    ) throws -> SpeculativeDecodingConfig {
        let model: any MTPDrafterModel =
            requiresPromptPrefill
            ? QwenStyleMTPDrafter()
            : StatelessMTPDrafter()
        return try makeMTPConfiguration(model: model)
    }

    private func makeMTPConfiguration(
        model: any MTPDrafterModel
    ) throws -> SpeculativeDecodingConfig {
        let container = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(id: "qwen-mtp-test"),
                model: model
            )
        )
        return try SpeculativeDecodingConfig(mtpDrafter: container, blockSize: 2)
    }

    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let completion = event.info {
                info = completion
            }
        }
        return (text, try #require(info))
    }

    private func collectAll(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, infos: [GenerateCompletionInfo]) {
        var text = ""
        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let info = event.info {
                infos.append(info)
            }
        }
        return (text, infos)
    }

    /// Tokens a restored raw Qwen fixture cache already holds.
    private let qwenCachedPrefix = [3, 9, 14, 22]

    /// A Qwen fixture cache that already holds ``qwenCachedPrefix``, as a restored raw cache
    /// does: the session has no transcript and no drafter continuation for it.
    private func prefilledQwenCache(_ fixture: Qwen35MTPFixture) throws -> [KVCache] {
        let cache = try fixture.target.newCache(parameters: nil)
        _ = fixture.target(
            LMInput.Text(tokens: MLXArray(qwenCachedPrefix.map(Int32.init))[.newAxis, 0...]),
            cache: cache, state: nil)
        eval(cache)
        return cache
    }
}

private struct DeterministicMTPTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { $0 == "x" ? 4 : 7 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(repeating: "x", count: tokenIds.filter { $0 != 15 }.count)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            encode(
                text: message["content"] as? String ?? "",
                addSpecialTokens: false)
        }
    }
}

private struct DeterministicMTPInputProcessor: UserInputProcessor {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}

private struct PreparedMediaMTPInputProcessor: UserInputProcessor {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-media-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let base = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        let prepared = try await base.prepare(input: input)
        return LMInput(
            text: prepared.text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1]))
        )
    }
}

private final class SecondTurnMediaMTPInputProcessor: UserInputProcessor, @unchecked Sendable {
    let tokenizer: DeterministicMTPTokenizer
    let configuration = ModelConfiguration(id: "qwen-mtp-media-switch-test")

    private let lock = NSLock()
    private var prepareCount = 0

    init(tokenizer: DeterministicMTPTokenizer) {
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let base = DeterministicMTPInputProcessor(tokenizer: tokenizer)
        let prepared = try await base.prepare(input: input)

        let includesMedia = lock.withLock {
            prepareCount += 1
            return prepareCount > 1
        }

        guard includesMedia else { return prepared }
        return LMInput(
            text: prepared.text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1]))
        )
    }
}

private struct ToolRestartRenderedMessage: Sendable {
    let role: String
    let content: String
    let toolCallNames: [String]
    let toolResultName: String?
}

private final class ToolRestartProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var storedRenderPasses: [[ToolRestartRenderedMessage]] = []
    private var storedDispatchedToolCalls: [ToolCall] = []

    func recordRender(_ messages: [[String: any Sendable]]) {
        let snapshot = messages.map { message in
            let toolCalls = message["tool_calls"] as? [[String: any Sendable]] ?? []
            let names = toolCalls.compactMap { toolCall in
                let function = toolCall["function"] as? [String: any Sendable]
                return function?["name"] as? String
            }
            return ToolRestartRenderedMessage(
                role: message["role"] as? String ?? "",
                content: message["content"] as? String ?? "",
                toolCallNames: names,
                toolResultName: message["name"] as? String
            )
        }

        lock.lock()
        storedRenderPasses.append(snapshot)
        lock.unlock()
    }

    func recordDispatch(_ call: ToolCall) {
        lock.lock()
        storedDispatchedToolCalls.append(call)
        lock.unlock()
    }

    var renderPasses: [[ToolRestartRenderedMessage]] {
        lock.lock()
        defer { lock.unlock() }
        return storedRenderPasses
    }

    var dispatchedToolCalls: [ToolCall] {
        lock.lock()
        defer { lock.unlock() }
        return storedDispatchedToolCalls
    }
}

private final class ToolRestartMTPTokenizer: Tokenizer, @unchecked Sendable {
    private let probe: ToolRestartProbe
    private let lock = NSLock()
    private var pass = 0

    private let toolCallScript =
        #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
    private let finalScript = "Sunny in Paris."

    init(probe: ToolRestartProbe) {
        self.probe = probe
    }

    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        Array(repeating: 7, count: text.unicodeScalars.count)
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        lock.lock()
        let currentPass = pass
        lock.unlock()
        let script = currentPass == 1 ? toolCallScript : finalScript
        return String(script.prefix(min(tokenIds.count * 4, script.count)))
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        probe.recordRender(messages)
        lock.lock()
        pass += 1
        lock.unlock()

        var tokens: [Int] = []
        for message in messages {
            switch message["role"] as? String {
            case "user":
                tokens.append(1)
            case "assistant":
                // Match the generation marker below so a rendered assistant
                // tool call extends the exact live generation trajectory.
                tokens.append(9)
            case "tool":
                tokens.append(5)
            default:
                tokens.append(6)
            }
            let content = message["content"] as? String ?? ""
            tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
            if let toolCalls = message["tool_calls"] as? [[String: any Sendable]] {
                // The scripted first pass emits 24 token-4 values. Re-render
                // those exact tool-call tokens so the restart is append-only.
                tokens.append(contentsOf: Array(repeating: 4, count: toolCalls.count * 24))
            }
            tokens.append(8)
        }
        tokens.append(9)
        return tokens
    }
}

private struct ToolRestartMTPInputProcessor: UserInputProcessor {
    let tokenizer: ToolRestartMTPTokenizer
    let configuration = ModelConfiguration(
        id: "qwen-mtp-tool-restart-test",
        toolCallFormat: .json
    )

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}
