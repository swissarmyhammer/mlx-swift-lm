// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import Testing

@testable import MLXFoundationModels

/// Proves, with real weights, that a second turn of one session feeds fewer
/// prompt tokens than a cold turn, and that the answer does not change.
///
/// The numbers are read from the CHANNEL, the way
/// `Tests/MLXFoundationModelsTests/UsageChannelSendTests.swift` reads them. The
/// `generationObserver` task-local is a test-only mirror that `emitUsage`
/// notifies BEFORE the send, thus a test that reads the observer would pass even
/// with the send absent. A consumer reads the channel, so this suite does too.
///
/// A unit test cannot answer this question. Project memory records that
/// generation-priming tokens of some chat templates break the prefix between one
/// turn's render and the next, thus only a real render of a real tokenizer can
/// say whether reuse fires.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct PromptCacheReuseChannelTests {

    /// A small instruction-tuned model whose chat template renders an assistant
    /// turn of history with the same header it writes to prime a generation.
    private static let modelID = TestFixtures.llamaModelID

    /// Keeps each round short: this suite measures the PROMPT, not the answer.
    private static let maximumResponseTokens = 24

    /// Greedy sampling, thus the warm answer and the cold answer are comparable.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var greedyOptions: GenerationOptions {
        GenerationOptions(samplingMode: .greedy, maximumResponseTokens: maximumResponseTokens)
    }

    @Test("a second turn of one session feeds fewer prompt tokens than a cold turn")
    func aSecondTurnOfOneSessionFeedsFewerPromptTokensThanAColdTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()

        let model = makeTestModel(Self.modelID)
        let executor = try makeMLXExecutor(for: model)

        let firstPrompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Name one primary color. One word."))
            ])
        let secondPrompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Name one more. One word."))
            ])

        let firstTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(firstPrompt)]),
                generationOptions: Self.greedyOptions),
            model: model)
        #expect(
            firstTurn.cachedTokenCount == 0,
            "The first turn of a session has no earlier turn to reuse.")

        let history: [Transcript.Entry] = [
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: firstTurn.text))])),
            .prompt(secondPrompt),
        ]

        // The second turn keeps the FIRST entry of the first turn, thus it names
        // the same session and the executor finds that session's cache.
        let secondTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(firstPrompt)] + history),
                generationOptions: Self.greedyOptions),
            model: model)

        // A cold control: the same words in a transcript whose first entry is a
        // NEW entry, thus it names a session the store has never seen.
        let coldFirstPrompt = Transcript.Prompt(segments: firstPrompt.segments)
        let coldTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(coldFirstPrompt)] + history),
                generationOptions: Self.greedyOptions),
            model: model)

        let warmFedTokenCount = secondTurn.promptTokenCount - secondTurn.cachedTokenCount
        print(
            """
            round 1: prompt \(firstTurn.promptTokenCount), cached \(firstTurn.cachedTokenCount)
            round 2: prompt \(secondTurn.promptTokenCount), \
            cached \(secondTurn.cachedTokenCount), fed \(warmFedTokenCount)
            cold control: prompt \(coldTurn.promptTokenCount), \
            cached \(coldTurn.cachedTokenCount)
            """)

        #expect(
            coldTurn.cachedTokenCount == 0,
            "A session the store has never seen must reuse nothing.")
        #expect(
            secondTurn.promptTokenCount == coldTurn.promptTokenCount,
            "The warm turn and the cold control render the same prompt.")
        #expect(
            secondTurn.cachedTokenCount > 0,
            "The second turn of a session must reuse the prompt of its first turn.")
        #expect(
            warmFedTokenCount < coldTurn.promptTokenCount,
            "The second turn must feed fewer tokens than the cold control.")
        #expect(
            secondTurn.text == coldTurn.text,
            "Reusing a cache must not change the answer.")

        await releaseAllGPUMemory()
    }

    /// The smallest model whose cache ROTATES. `Gemma3TextModel.newCache` gives
    /// a sliding-window cache to five of each six layers, thus this model is a
    /// `RotatingKVCache` model in the sense card ^2nztex1 names.
    private static let slidingWindowModelID = TestFixtures.gemmaModelID

    /// The sliding window of the Gemma 3 configuration, which is the default of
    /// `sliding_window`. Past this position a rotating cache has overwritten the
    /// keys a rewind needs, thus the rewind lands nowhere.
    private static let slidingWindowTokenCount = 512

    /// How many filler words the first prompt of the sliding-window round
    /// carries. The render must stand PAST the window, or the cache still
    /// rewinds and the measurement says nothing about a rotating cache.
    private static let fillerWordCount = 900

    @Test("a second turn of a sliding-window model reuses the prompt of its first turn")
    func aSecondTurnOfASlidingWindowModelReusesThePromptOfItsFirstTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()

        let model = makeTestModel(Self.slidingWindowModelID)
        let executor = try makeMLXExecutor(for: model)

        let filler = (1 ... Self.fillerWordCount).map { "word\($0)" }.joined(separator: " ")
        let firstPrompt = Transcript.Prompt(
            segments: [
                .text(
                    Transcript.TextSegment(
                        content: "Read this list: \(filler). Now name one primary color."))
            ])
        let secondPrompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Name one more. One word."))
            ])

        let firstTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(firstPrompt)]),
                generationOptions: Self.greedyOptions),
            model: model)
        #expect(
            firstTurn.cachedTokenCount == 0,
            "The first turn of a session has no earlier turn to reuse.")
        #expect(
            firstTurn.promptTokenCount > Self.slidingWindowTokenCount,
            """
            the first turn rendered \(firstTurn.promptTokenCount) tokens, which stands inside \
            the \(Self.slidingWindowTokenCount)-token window. Lengthen the filler, or this \
            round measures a cache that still rewinds.
            """)

        let history: [Transcript.Entry] = [
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: firstTurn.text))])),
            .prompt(secondPrompt),
        ]

        let secondTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [.prompt(firstPrompt)] + history),
                generationOptions: Self.greedyOptions),
            model: model)

        print(
            """
            sliding-window round 1: prompt \(firstTurn.promptTokenCount), \
            cached \(firstTurn.cachedTokenCount)
            sliding-window round 2: prompt \(secondTurn.promptTokenCount), \
            cached \(secondTurn.cachedTokenCount), \
            fed \(secondTurn.promptTokenCount - secondTurn.cachedTokenCount)
            """)

        #expect(
            secondTurn.cachedTokenCount > 0,
            """
            the second turn of a rotating cache reused nothing. The ledger of the first turn \
            names its render plus the tokens it generated, thus ExtendCachedPrefixRule needs \
            no rewind here.
            """)
        #expect(
            secondTurn.cachedTokenCount > firstTurn.promptTokenCount,
            """
            the second turn reused \(secondTurn.cachedTokenCount) tokens of a first turn that \
            rendered \(firstTurn.promptTokenCount). A ledger that carries the generated tokens \
            reaches past the render.
            """)

        await releaseAllGPUMemory()
    }

    /// A model that loads through `VLMModelFactory`. Its processor batches a
    /// text-only prompt to one row and masks every token present, which is
    /// the input shape card ^7fy0d2z lets through the plan guard. The
    /// checkpoint must already stand in the local Hugging Face cache. The
    /// model always reasons, thus `.reasoning` must be declared on it.
    private static let vlmModelID = "mlx-community/Muse-Glimmer-30B-4bit"

    /// Runs two turns of one session on `model`, and reads what the channel
    /// carried for each.
    ///
    /// The second turn keeps the FIRST entry of the first turn, thus it names
    /// the same session and the executor finds that session's cache.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func twoTurnsOfOneSession(
        on model: MLXLanguageModel, firstPrompt: String, secondPrompt: String
    ) async throws -> (first: ChannelResponse, second: ChannelResponse) {
        let executor = try makeMLXExecutor(for: model)
        let firstEntry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: firstPrompt))]))

        let firstTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [firstEntry]),
                generationOptions: Self.greedyOptions),
            model: model)

        let secondTurn = try await respondReadingTheChannel(
            executor,
            request: makeExecutorRequest(
                transcript: Transcript(entries: [
                    firstEntry,
                    .response(
                        Transcript.Response(
                            assetIDs: [],
                            segments: [.text(Transcript.TextSegment(content: firstTurn.text))])),
                    .prompt(
                        Transcript.Prompt(
                            segments: [.text(Transcript.TextSegment(content: secondPrompt))])),
                ]),
                generationOptions: Self.greedyOptions),
            model: model)
        return (firstTurn, secondTurn)
    }

    @Test("a second turn of a VLM-processor model reuses the prompt of its first turn")
    func aSecondTurnOfAVLMProcessorModelReusesThePromptOfItsFirstTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()

        let turns = try await twoTurnsOfOneSession(
            on: makeReasoningTestModel(Self.vlmModelID),
            firstPrompt: "Name one primary color. One word.",
            secondPrompt: "Name one more. One word.")

        print(
            """
            VLM-processor round 1: prompt \(turns.first.promptTokenCount), \
            cached \(turns.first.cachedTokenCount)
            VLM-processor round 2: prompt \(turns.second.promptTokenCount), \
            cached \(turns.second.cachedTokenCount), \
            fed \(turns.second.promptTokenCount - turns.second.cachedTokenCount)
            """)

        #expect(
            turns.first.cachedTokenCount == 0,
            "The first turn of a session has no earlier turn to reuse.")
        #expect(
            turns.second.cachedTokenCount > 0,
            """
            the second turn of a VLM-processor session reused nothing. The processor batches \
            a text-only prompt to one row and masks every token present; the plan must read \
            the token ledger from that one row.
            """)

        await releaseAllGPUMemory()
    }
}

/// What one response sent into the generation channel.
private struct ChannelResponse {

    /// The concatenated `.response` text of the whole answer.
    var text = ""

    /// The `totalTokenCount` of the last usage event.
    var promptTokenCount = 0

    /// The `cachedTokenCount` of the last usage event.
    var cachedTokenCount = 0
}

/// Collects channel events across the consumer task and the test.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private actor ChannelEventCollector {
    private var events: [LanguageModelExecutorGenerationChannel.Event] = []

    /// Keeps one event.
    func append(_ event: LanguageModelExecutorGenerationChannel.Event) {
        events.append(event)
    }

    /// Every event the channel delivered, in order.
    func collected() -> [LanguageModelExecutorGenerationChannel.Event] {
        events
    }
}

/// Runs one response and reads what the CHANNEL carried.
///
/// The channel is a rendezvous: a send blocks until a consumer takes the event,
/// thus the consumer runs beside `respond`. The channel has no `finish()`, thus
/// the consumer is cancelled once `respond` returns and its task is awaited: an
/// event already handed to the loop is appended before the loop asks for the
/// next one.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func respondReadingTheChannel(
    _ executor: MLXLanguageModel.Executor,
    request: LanguageModelExecutorGenerationRequest,
    model: MLXLanguageModel
) async throws -> ChannelResponse {
    let channel = LanguageModelExecutorGenerationChannel()
    let collector = ChannelEventCollector()
    let consumer = Task {
        do {
            for try await event in channel {
                await collector.append(event)
            }
        } catch {
            // Including cancellation, which is how this consumer always ends.
        }
    }

    do {
        try await executor.respond(to: request, model: model, streamingInto: channel)
    } catch {
        consumer.cancel()
        await consumer.value
        throw error
    }
    consumer.cancel()
    await consumer.value

    var result = ChannelResponse()
    for event in await collector.collected() {
        guard
            let response = reflectedChannelPayload(
                of: event, caseLabel: "response",
                as: LanguageModelExecutorGenerationChannel.Response.self)
        else {
            continue
        }
        if let fragment = reflectedChannelPayload(
            of: response.action, caseLabel: "appendText",
            as: LanguageModelExecutorGenerationChannel.TextFragment.self)
        {
            result.text += fragment.content
        }
        if let usage = reflectedChannelPayload(
            of: response.action, caseLabel: "updateUsage",
            as: LanguageModelExecutorGenerationChannel.Usage.self)
        {
            result.promptTokenCount = usage.input.totalTokenCount
            result.cachedTokenCount = usage.input.cachedTokenCount
        }
    }
    return result
}

#endif  // FoundationModelsIntegration
