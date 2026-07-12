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

        // The FULL length of what round 1 actually stores (prompt tokens +
        // its own real generated reply tokens) -- the yardstick the
        // magnitude-bounded check below holds round 2's reuse to, mirroring
        // `PromptCacheForkReuseTests`' `sharedPrefixStoredLength`.
        let sharedPrefixTokens = first.promptTokenCount + first.outputTokenCount

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

        // Magnitude-bounded, not just `> 0`: a single stray cached token out
        // of a much longer prefix would satisfy `> 0` but prove nothing.
        // Round 2 replays round 1's OWN real reply verbatim before its new
        // question, so its rendered prefix is a byte-for-byte continuation
        // of round 1's stored tokens -- confirmed by a real run (this
        // exact scenario measured `cachedTokenCount == sharedPrefixTokens`,
        // i.e. slack 0). The `- 1` here is the one KNOWN, documented slop
        // source for this continuation shape: `PromptCache
        // .reconcileCacheAdvance`'s `.trimCacheByOne` case, where the
        // cache's real offset can legitimately land exactly one token
        // ahead of the observed generated-token count.
        #expect(
            second.cachedTokenCount >= sharedPrefixTokens - 1,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should be within \
            one token of the first round's full stored prefix (\(sharedPrefixTokens) = \
            \(first.promptTokenCount) prompt + \(first.outputTokenCount) output tokens) -- a \
            merely-positive bound (or the old promptTokenCount-derived growth check) could pass \
            with only a small fraction of the prefix actually reused.
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
