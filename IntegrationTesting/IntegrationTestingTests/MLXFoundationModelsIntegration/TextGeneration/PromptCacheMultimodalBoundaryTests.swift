// Copyright © 2026 Apple Inc.
//
// Multimodal-boundary integration test for `PromptCache`: an image-bearing
// turn sandwiched between two text-only turns in the same session, proving
// (a) a text-only round FOLLOWING a multimodal round still correctly
// reuses the pre-image cache entry (an intervening multimodal round must
// not evict or otherwise disturb it), and (b) the multimodal round's own
// state is never stored under a reusable key -- see
// `MLXLanguageModel.Executor.isTextOnly`/`makePromptCacheSlot`'s doc
// comments: multimodal rounds are excluded from the cache path entirely
// (`cache: nil`, never resolved, never committed), specifically because
// reducing `LMInput` to a raw token suffix would silently drop attached
// media with no way to re-attach it.
//
// `isTextOnly`/`makePromptCacheSlot` are `private` inside
// `MLXLanguageModel.Executor` -- not reachable even via `@testable import`
// (Swift's `private` is scoped to the enclosing declaration, not just the
// module) -- so this can't be a unit test against that logic directly.
// This exercises it end-to-end instead, through a real vision-capable
// model, the only level at which the exclusion is externally observable.
//
// Requires a real several-GB VLM download, so -- like `VisionIntegrationTests`,
// whose `Qwen3-VL-4B-Instruct-4bit` model and opt-in gating this file
// mirrors -- this is SKIPPED BY DEFAULT and only runs with
// `MLX_RUN_VLM_INTEGRATION=1` set, on Apple silicon, on demand.
//
// EXPLICITLY UNTESTED (not silently skipped): attempted for real in this
// environment with `MLX_RUN_VLM_INTEGRATION=1` set (2026-07-12) -- the
// `IntegrationTesting` target itself builds and runs fine now (the
// pre-existing SDK-incompatibility this caveat used to describe no longer
// reproduces; see `PromptCacheEquivalenceTests.swift`'s header), and this
// sandbox DOES have network access and disk space for the multi-GB
// `Qwen3-VL-4B-Instruct-4bit` download -- confirmed by watching the real
// download progress past 2.7 GB. What did NOT succeed: the download alone
// did not finish downloading + loading + running all three rounds within
// this suite's own `.timeLimit(.minutes(10))` before the run was cut off
// (`Time limit was exceeded: 600.000 seconds`, `PromptCacheMultimodalBoundaryTests
// .swift:50`), given this environment's download throughput for a model
// this size. Raising the suite's own time limit was judged out of scope for
// this pass (a real behavior change to the test, not just a caveat
// update) -- left as a follow-up rather than silently declared passing or
// silently left as a stale, no-longer-accurate "unverified" note.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import CoreGraphics
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration test proving a sandwiched multimodal round neither disturbs a surrounding text-only cache entry nor is itself ever cached.
@Suite(
    .serialized, .timeLimit(.minutes(10)),
    .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_VLM_INTEGRATION"] == "1"))
struct PromptCacheMultimodalBoundaryTests {

    @Test(
        """
        A multimodal round sandwiched between two text-only turns never disturbs the \
        pre-image cache entry, and is itself never stored under a reusable key
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func multimodalRoundDoesNotDisturbPreImageCacheEntry() async throws {
        let model = makeTestModel(
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            capabilities: [.vision, .guidedGeneration, .toolCalling])
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Round 1 (text-only): establishes a cache entry for this model id.
        let firstUserText = "Say 'hi' in exactly one word."
        let firstPrompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: firstUserText))],
            responseFormat: nil)
        let firstRequest = makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(firstPrompt)]),
            generationOptions: greedyOptions)
        let first = try await respondCollectingTextAndUsage(
            executor, request: firstRequest, model: model)
        #expect(!first.text.isEmpty)

        // Round 2 (multimodal): an image attachment alongside text, on top
        // of round 1's transcript. Per `isTextOnly`, this round must never
        // participate in prompt-cache reuse in either direction -- it
        // neither reuses round 1's entry nor stores its own.
        let image = makeSolidCGImageForCacheBoundaryTest()
        let imageAttachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(image)), label: "photo")
        let imagePrompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "What color is this image?")),
                .attachment(imageAttachment),
            ],
            responseFormat: nil)
        let imageEntries: [Transcript.Entry] = [
            .prompt(firstPrompt),
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: first.text))])
            ),
            .prompt(imagePrompt),
        ]
        let imageRequest = makeExecutorRequest(
            transcript: Transcript(entries: imageEntries), generationOptions: greedyOptions)
        let imageRound = try await respondCollectingTextAndUsage(
            executor, request: imageRequest, model: model)
        #expect(!imageRound.text.isEmpty)
        #expect(
            imageRound.cachedTokenCount == 0,
            "a multimodal round must never itself participate in prompt-cache reuse"
        )

        // Round 3 (text-only): continues from ROUND 1 (not round 2's
        // multimodal branch) -- the pre-image cache entry must still be
        // present and reusable despite the intervening multimodal round.
        let thirdUserText = "Now say 'bye' in exactly one word."
        let thirdEntries: [Transcript.Entry] = [
            .prompt(firstPrompt),
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: first.text))])
            ),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: thirdUserText))],
                    responseFormat: nil)),
        ]
        let thirdRequest = makeExecutorRequest(
            transcript: Transcript(entries: thirdEntries), generationOptions: greedyOptions)
        let third = try await respondCollectingTextAndUsage(
            executor, request: thirdRequest, model: model)
        #expect(!third.text.isEmpty)
        #expect(
            third.cachedTokenCount > 0,
            """
            round 1's pre-image cache entry should still be reusable by round 3 despite the \
            intervening multimodal round 2 -- cachedTokenCount == 0 means the entry was lost \
            or the multimodal round incorrectly disturbed it
            """
        )

        await releaseAllGPUMemory()
    }
}

/// A tiny solid-color `CGImage`, built in-memory, for a valid labeled
/// image attachment without any binary fixture. Self-contained here
/// (rather than reusing `Tests/MLXFoundationModelsTests/TestHelpers.swift`'s
/// `makeSolidCGImage`) since that helper lives in a separate SwiftPM test
/// target this xcodeproj-based target doesn't share sources with.
private func makeSolidCGImageForCacheBoundaryTest(width: Int = 2, height: Int = 2) -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1.0))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
}

#endif  // FoundationModelsIntegration
