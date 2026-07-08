// Copyright © 2026 Apple Inc.
//
// Regression test proving `PromptCache` actually reduces prefill work
// end-to-end, not just at the unit level: a second `respond()` round on the
// same model, with the transcript grown by the first round's own answer plus
// a new user turn, must report a *smaller* prompt token count than the first
// round -- proof the executor fed only the appended suffix into the model,
// not the whole (now-longer) transcript from scratch.
//
// Loads a real model, so it lives in the IntegrationTesting xcodeproj
// alongside the other `TextGeneration` tests that exercise `respond()`
// end-to-end (see `UpdateUsageEmissionTests`, whose `collectFinalUsage`
// helper this test's `respondCollectingTextAndUsage` mirrors).
//
// `PromptCache` is a single process-global actor keyed by `modelID` (mirrors
// `MLXLanguageModel`'s `ModelCache`), so this test's two rounds only observe
// each other's cache entry if nothing else touches the same model id
// concurrently -- run in isolation from other suites touching
// `TestFixtures.defaultModelID` for a reliable result (`.serialized` only
// orders this suite's own tests against each other).

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

@Suite(.serialized, .timeLimit(.minutes(5)))
struct PromptCacheReuseTests {

    @Test("Second respond() round prefills only the appended suffix, not the whole transcript")
    func secondRoundPrefillsOnlyAppendedSuffix() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        let firstUserText = "Say 'hi' in exactly one word."
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(
            transcript: firstTranscript,
            generationOptions: GenerationOptions(maximumResponseTokens: 8)
        )
        let first = try await respondCollectingTextAndUsage(
            executor, request: firstRequest, model: model)
        #expect(!first.text.isEmpty, "First round should produce some response text")
        #expect(first.promptTokenCount > 0, "First round's prompt token count should be positive")

        // Second round's transcript replays the first round's own answer as
        // an assistant entry (mirroring what a real `LanguageModelSession`
        // does) and appends one new user turn. Without cache reuse this
        // prefill is strictly longer than the first round's; with it,
        // `respond()` should feed only the assistant reply + new user turn.
        let secondTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil)),
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: first.text))])
            ),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Now say 'bye' in one word."))],
                    responseFormat: nil)),
        ])
        let secondRequest = makeExecutorRequest(
            transcript: secondTranscript,
            generationOptions: GenerationOptions(maximumResponseTokens: 8)
        )
        let second = try await respondCollectingTextAndUsage(
            executor, request: secondRequest, model: model)
        #expect(!second.text.isEmpty, "Second round should produce some response text")

        #expect(
            second.promptTokenCount < first.promptTokenCount,
            """
            Second round's prompt token count (\(second.promptTokenCount)) should be smaller \
            than the first round's (\(first.promptTokenCount)) -- proof the cache reused the \
            first round's KV state and prefilled only the appended suffix, not the whole \
            (now-longer) transcript from scratch.
            """
        )

        await releaseAllGPUMemory()
    }
}

/// Drains `executor.respond(...)`'s event stream, returning the concatenated
/// response text and the prompt token count from the last `.updateUsage`
/// event -- the same last-write-wins convention `UpdateUsageEmissionTests`'
/// `collectFinalUsage` documents (the framework's usage aggregator replaces
/// totals wholesale on each event).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func respondCollectingTextAndUsage(
    _ executor: MLXLanguageModel.Executor,
    request: LanguageModelExecutorGenerationRequest,
    model: MLXLanguageModel
) async throws -> (text: String, promptTokenCount: Int) {
    let stream = try await executeResponse(executor, request: request, model: model)
    var text = ""
    var promptTokenCount: Int?
    for try await event in stream {
        guard let response = event as? LanguageModelExecutorGenerationChannel.Response else {
            continue
        }
        if case .appendText(let delta) = response.action {
            text += delta.content
        }
        if case .updateUsage(let usage) = response.action {
            promptTokenCount = usage.input.totalTokenCount
        }
    }
    let count = try #require(
        promptTokenCount, "Expected at least one .updateUsage event before stream completion")
    return (text, count)
}

#endif  // FoundationModelsIntegration
