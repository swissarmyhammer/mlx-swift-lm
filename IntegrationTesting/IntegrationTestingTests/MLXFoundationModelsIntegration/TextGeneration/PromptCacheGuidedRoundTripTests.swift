// Copyright © 2026 Apple Inc.
//
// Guided-generation round-trip integration test for `PromptCache`: a
// schema-constrained (grammar-guided) turn followed by a follow-up turn,
// asserting the first round's cache entry was actually STORED (not
// silently dropped by reconciliation) and REUSED, and that its own output
// matches a cache-evicted baseline.
//
// This exercises the token-ID-threading fix from the multi-slot redesign
// (see kanban task `qawe2hb`'s comments): `GuidedGenerationLoop.run`'s
// `onTokenCommitted` callback threads REAL fed token IDs to
// `Executor.commitPromptCache`, replacing an earlier re-encode-by-count
// approach whose off-by-one (the grammar-accepted natural-termination
// token is sampled/detokenized/emitted but never fed back through the
// model) silently dropped the cache entry on essentially every successful
// guided-generation completion. A regression there would show up here as
// `cachedTokenCount == 0` on the follow-up round.
//
// Loads a real model, so this lives in the IntegrationTesting xcodeproj
// alongside `PromptCacheReuseTests`/`PromptCacheEquivalenceTests`, sharing
// their `respondCollectingTextAndUsage` helper
// (`Support/FMTestHelpers.swift`; its `cachedTokenCount` field is what this
// file's assertions need). Schema/request shape matches
// `UpdateUsageEmissionTests`' `usage_emittedOnGuidedPath`.
//
// SANDBOX CAVEAT: unverified by execution here, for the same pre-existing,
// unrelated SDK-compatibility reason documented in
// `PromptCacheEquivalenceTests.swift`'s header comment.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Generable type for the guided-generation round trip. Distinct name from
/// `UpdateUsageEmissionTests`' file-scope `YesOrNoAnswer` -- `@Generable`'s
/// macro-generated conformances collide across files on the type name, not
/// just within one file (see `EagerFallbackWeatherArgs`' doc comment for
/// the established precedent).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct PromptCacheRoundTripAnswer {
    @Guide(description: "Either 'yes' or 'no'.")
    var answer: String
}

@Suite(
    .serialized, .timeLimit(.minutes(10)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheGuidedRoundTripTests {

    @Test(
        """
        A guided-generation round's cache entry is actually stored (not silently \
        dropped) and reused by a follow-up turn, whose output matches a fresh, \
        cache-evicted rebuild
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func guidedRoundThenFollowUpReusesCacheAndMatchesBaseline() async throws {
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Round 1: schema-constrained (guided) generation -- exercises
        // `GuidedGenerationLoop.run`'s `onTokenCommitted`-based exact-
        // token-ID cache commit path.
        let firstUserText = "Is the sky blue? Reply with yes or no."
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(
            transcript: firstTranscript, schema: PromptCacheRoundTripAnswer.generationSchema)
        let first = try await respondCollectingTextAndUsage(
            executor, request: firstRequest, model: model)
        #expect(!first.text.isEmpty, "First guided round should produce some response text")

        // Round 2: a follow-up guided turn, replaying round 1's own
        // answer. If reconciliation had silently dropped round 1's cache
        // entry (the original off-by-one bug), `resolve()` would find no
        // slot for this model id and `cachedTokenCount` would be 0.
        let secondUserText = "Is grass green? Reply with yes or no."
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
                    segments: [.text(Transcript.TextSegment(content: secondUserText))],
                    responseFormat: nil)),
        ])
        let secondRequest = makeExecutorRequest(
            transcript: secondTranscript, schema: PromptCacheRoundTripAnswer.generationSchema)
        let second = try await respondCollectingTextAndUsage(
            executor, request: secondRequest, model: model)
        #expect(!second.text.isEmpty, "Second guided round should produce some response text")
        #expect(
            second.cachedTokenCount > 0,
            """
            round 1's guided-generation cache entry should have been stored and reused by \
            round 2 -- cachedTokenCount == 0 means reconciliation silently dropped it (the \
            original off-by-one bug this redesign fixed)
            """
        )

        // Baseline: force a fresh rebuild of round 2's SAME transcript and
        // compare output.
        await MLXLanguageModel.removePromptCache(modelID: model.modelID)
        let freshRequest = makeExecutorRequest(
            transcript: secondTranscript, schema: PromptCacheRoundTripAnswer.generationSchema)
        let fresh = try await respondCollectingTextAndUsage(
            executor, request: freshRequest, model: model)

        #expect(
            second.text == fresh.text,
            """
            guided round-2 output (\(second.text)) diverged between the cache-reused run and \
            a fresh, cache-evicted rebuild (\(fresh.text))
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
