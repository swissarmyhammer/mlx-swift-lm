// Copyright © 2026 Apple Inc.
//
// Integration test for kanban `1h5jhne`: proves the chunk-based `PromptCache`
// (kanban `cthbfmw`'s chunk cutover) holds the `FoundationModelsRouter` fork
// scenario's real end-to-end guarantee, not just at the unit level (see
// `Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift`'s
// `forkFanOutSharesParentPrefixFromChunksForEveryForkFoundationModelsRouterScenario`,
// which proves the identical shape against a synthetic `PromptCacheProbeModel`).
// A `FoundationModelsRouter`-style fork spins up a new, transcript-seeded
// `LanguageModelSession` per branch: each fork's FIRST `respond()` call
// receives a transcript that is the shared parent's entries verbatim, plus
// its own new, diverging tail -- exactly `forkATranscript`/`forkBTranscript`
// below (mirrors `PromptCachePrewarmTests`' `forkTranscript(question:)`).
//
// Under the OLD slot-pool design (`PromptCache.defaultMaxSlotsPerModel`-bounded
// checkout), a fork's own round could check out the shared parent's slot for
// itself, leaving a SIBLING fork -- or the parent's own next turn -- to find
// it already checked out (a miss forcing a full rebuild) or stolen outright.
// Chunks are read-only and dedup'd (see `PromptCache`'s own doc comment: "no
// checkout, no steal, and no double-checkout hazard by construction"), so
// this is now a STRICTLY STRONGER guarantee: every fork, AND the parent's own
// continuation issued after both forks have run, must all still reuse the
// shared prefix from chunks.
//
// Loads a real model, so this lives in the IntegrationTesting xcodeproj
// alongside the other `TextGeneration` tests that exercise `respond()`
// end-to-end, sharing `Support/FMTestHelpers.swift`'s
// `respondCollectingTextAndUsage` (also used by `PromptCacheReuseTests`,
// `PromptCacheEquivalenceTests`, `PromptCacheGuidedRoundTripTests`).
//
// `PromptCache` is a single process-global actor keyed by `modelID`, so this
// test's rounds only observe each other's cache entries if nothing else
// touches the same model id concurrently -- run in isolation from other
// suites touching `TestFixtures.defaultModelID` for a reliable result
// (`.serialized` only orders this suite's own tests against each other).
//
// Verified by real execution in this environment:
//   xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/PromptCacheForkReuseTests

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration test proving a fork fan-out's shared parent prefix is reused by every fork AND survives back to the parent's own next turn, end-to-end against a real model.
@Suite(
    .serialized, .timeLimit(.minutes(5)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheForkReuseTests {

    @Test(
        """
        Two forks extending the SAME parent transcript differently both reuse the shared \
        prefix, AND the parent's own continuation issued after both forks have run is still \
        reused -- the no-steal guarantee that was false under the old slot-pool design
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func forksAndParentContinuationAllReuseSharedPrefixWithoutStealing() async throws {
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Parent turn: establishes the shared prefix every fork below extends.
        let parentUserText = "Say 'hi' in exactly one word."
        let parentPromptEntries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: parentUserText))],
                    responseFormat: nil))
        ]
        let parentRequest = makeExecutorRequest(
            transcript: Transcript(entries: parentPromptEntries),
            generationOptions: greedyOptions)
        let parentFirst = try await respondCollectingTextAndUsage(
            executor, request: parentRequest, model: model)
        #expect(!parentFirst.text.isEmpty, "Parent round should produce some response text")

        // The FULL length of what the parent round actually stores (prompt
        // tokens + its own real generated reply tokens) -- a tighter
        // yardstick than `parentFirst.promptTokenCount` alone for the
        // "only the new tail was fed" bound checks below (mirrors
        // `PromptCacheEquivalenceTests.editedEarlierTurnForcesTrimAndMatchesFreshRebuild`'s
        // `firstStoredSlotLength`).
        let sharedPrefixStoredLength = parentFirst.promptTokenCount + parentFirst.outputTokenCount

        // The shared parent prefix every fork's transcript extends verbatim --
        // the parent's own real reply, threaded through exactly like a real
        // `LanguageModelSession` would replay it.
        let sharedParentEntries: [Transcript.Entry] =
            parentPromptEntries + [
                .response(
                    Transcript.Response(
                        assetIDs: [],
                        segments: [.text(Transcript.TextSegment(content: parentFirst.text))]))
            ]

        // Fork A: the shared prefix plus fork A's own distinct new user turn --
        // exactly the shape a forked, transcript-seeded `LanguageModelSession`'s
        // first `respond()` call would receive.
        let forkAQuestion = "Now say 'bye' in exactly one word."
        let forkATranscript = Transcript(
            entries: sharedParentEntries + [
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: forkAQuestion))],
                        responseFormat: nil))
            ])
        let forkARequest = makeExecutorRequest(
            transcript: forkATranscript, generationOptions: greedyOptions)
        let forkA = try await respondCollectingTextAndUsage(
            executor, request: forkARequest, model: model)
        #expect(!forkA.text.isEmpty, "Fork A should produce some response text")
        #expect(
            forkA.cachedTokenCount > 0,
            """
            Fork A's first respond() round should reuse the shared parent prefix from chunks -- \
            cachedTokenCount == 0 means the fork's own tokenization diverged from the parent's \
            stored chunks, or nothing was stored for the parent round
            """
        )
        #expect(
            forkA.promptTokenCount - forkA.cachedTokenCount < sharedPrefixStoredLength,
            """
            Fork A's uncached suffix should be smaller than the parent round's full stored \
            length (\(sharedPrefixStoredLength)) -- proof only fork A's own appended tail was \
            fed, not the whole shared prefix rebuilt from scratch
            """
        )

        // Fork B: the SAME shared prefix, a DIFFERENT distinct new user turn --
        // must ALSO reuse the shared prefix, proving fork A's own respond()
        // round did not check out, evict, or otherwise steal the parent's
        // chunks out from under a sibling fork (the old slot-pool design's
        // failure mode this chunk-based redesign eliminates).
        let forkBQuestion = "Now say 'ok' in exactly one word."
        let forkBTranscript = Transcript(
            entries: sharedParentEntries + [
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: forkBQuestion))],
                        responseFormat: nil))
            ])
        let forkBRequest = makeExecutorRequest(
            transcript: forkBTranscript, generationOptions: greedyOptions)
        let forkB = try await respondCollectingTextAndUsage(
            executor, request: forkBRequest, model: model)
        #expect(!forkB.text.isEmpty, "Fork B should produce some response text")
        #expect(
            forkB.cachedTokenCount > 0,
            """
            Fork B's first respond() round should ALSO reuse the shared parent prefix from \
            chunks, even though fork A already resolved (and respond()ed) against it -- \
            cachedTokenCount == 0 here would mean fork A's round stole or evicted the shared \
            prefix out from under fork B
            """
        )
        #expect(
            forkB.promptTokenCount - forkB.cachedTokenCount < sharedPrefixStoredLength,
            """
            Fork B's uncached suffix should be smaller than the parent round's full stored \
            length (\(sharedPrefixStoredLength)) -- proof only fork B's own appended tail was \
            fed, not the whole shared prefix rebuilt from scratch
            """
        )

        // Parent's OWN continuation, issued AFTER both forks have already
        // resolved and respond()ed -- proves the shared prefix survived both
        // fork rounds intact (the no-steal guarantee the old slot-pool design
        // could violate: a fork's round checking out the parent's only slot
        // would leave the parent's own next turn to miss and rebuild from
        // scratch).
        let parentContinuationQuestion = "Now say 'thanks' in exactly one word."
        let parentContinuationTranscript = Transcript(
            entries: sharedParentEntries + [
                .prompt(
                    Transcript.Prompt(
                        segments: [
                            .text(Transcript.TextSegment(content: parentContinuationQuestion))
                        ],
                        responseFormat: nil))
            ])
        let parentContinuationRequest = makeExecutorRequest(
            transcript: parentContinuationTranscript, generationOptions: greedyOptions)
        let parentContinuation = try await respondCollectingTextAndUsage(
            executor, request: parentContinuationRequest, model: model)
        #expect(
            !parentContinuation.text.isEmpty, "Parent's continuation should produce response text"
        )
        #expect(
            parentContinuation.cachedTokenCount > 0,
            """
            The parent's OWN continuation, issued after BOTH forks already ran, should still \
            reuse the shared prefix from chunks -- cachedTokenCount == 0 here would mean one of \
            the forks' rounds stole or evicted the parent's shared prefix, the exact hazard the \
            old slot-pool design could not prevent
            """
        )
        #expect(
            parentContinuation.promptTokenCount - parentContinuation.cachedTokenCount
                < sharedPrefixStoredLength,
            """
            The parent continuation's uncached suffix should be smaller than the parent round's \
            full stored length (\(sharedPrefixStoredLength)) -- proof only the continuation's \
            own appended tail was fed, not the whole shared prefix rebuilt from scratch
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
