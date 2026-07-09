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
// end-to-end. Uses `Support/FMTestHelpers.swift`'s shared
// `respondCollectingTextAndUsage` (mirroring `UpdateUsageEmissionTests`'
// `collectFinalUsage`), also used by `PromptCacheEquivalenceTests` and
// `PromptCacheGuidedRoundTripTests`.
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

/// Integration test proving a second `respond()` round prefills only the appended suffix, not the whole (now-longer) transcript.
@Suite(
    .serialized, .timeLimit(.minutes(5)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheReuseTests {

    // Previously a `guard #available(...) else { return }` at the top of
    // the test body: on an OS below the 27 floor, that made the test
    // report PASSED with zero assertions ever run -- a silent no-op
    // indistinguishable from a real pass in CI output. The `@available`
    // attribute directly on the function (below) makes the compiler enforce
    // the same floor statically, and the suite's `.enabled(if:)` trait
    // (above) makes swift-testing report this test as explicitly SKIPPED
    // (not passed) when the floor isn't met, so a "green" run can no longer
    // hide an unmet OS requirement.
    @Test("Second respond() round prefills only the appended suffix, not the whole transcript")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func secondRoundPrefillsOnlyAppendedSuffix() async throws {
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let generationOptions = GenerationOptions(maximumResponseTokens: 8)

        let firstUserText = "Say 'hi' in exactly one word."
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(
            transcript: firstTranscript,
            generationOptions: generationOptions
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
            generationOptions: generationOptions
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

#endif  // FoundationModelsIntegration
