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
