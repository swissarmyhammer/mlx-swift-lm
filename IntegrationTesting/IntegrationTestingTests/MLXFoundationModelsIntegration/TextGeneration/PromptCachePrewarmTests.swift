// Copyright © 2026 Apple Inc.
//
// Integration tests for `prewarm(model:transcript:)`'s new chunk-population
// behavior (kanban `kr3zkap`): before this change, `prewarm` ran a fixed
// dummy warmup prompt and ignored its `transcript` argument entirely --
// useless for the prompt cache. With `PromptCache`'s chunk store (kanban
// `cthbfmw`), prewarming a PARENT transcript genuinely matters for a
// `FoundationModelsRouter`-style fork scenario: populating the shared
// prefix's chunks BEFORE any fork's first `respond()` means every fork's
// first turn hits, instead of only the first fork paying to build it.
//
// `Executor.populatePromptCacheChunks(model:transcript:)` is the internal
// (not `private`) async core `prewarm(model:transcript:)` wraps in a
// fire-and-forget detached `Task` -- these tests call it DIRECTLY (awaited),
// exactly like `PrewarmGrammarTests` calls `model.warmUp()` directly rather
// than going through the fire-and-forget `prewarm(model:transcript:)`
// wrapper, since a detached `Task` gives a test no handle to await
// completion before asserting on the result.
//
// SANDBOX CAVEAT: unverified by execution here, exactly like
// `PromptCacheReuseTests`/`PromptCacheEquivalenceTests`/
// `PromptCacheMultimodalBoundaryTests` already are -- the whole
// `IntegrationTesting` xcodeproj target fails to compile against this
// environment's Xcode-beta/FoundationModels SDK for the pre-existing,
// unrelated reason documented in `PromptCacheEquivalenceTests.swift`'s
// header comment (`Response.Action`/`.appendText`/`.updateUsage` are an
// opaque struct with static factory methods in this SDK build, not the
// plain enum a dozen pre-existing files in this directory assume). Written
// to compile correctly against the SDK types and follow this directory's
// established idioms, but not run to a passing result in this sandbox.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import CoreGraphics
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration tests proving `prewarm`'s new transcript chunk-population
/// step makes a later `respond()` round reuse the prewarmed prefix, starting
/// on its FIRST round -- the `FoundationModelsRouter` fork scenario's whole
/// point.
@Suite(
    .serialized, .timeLimit(.minutes(5)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCachePrewarmTests {

    /// A shared "parent" instructions entry -- stands in for a
    /// `FoundationModelsRouter`-style shared system prompt every fork
    /// session inherits before diverging on its own question.
    private static let sharedInstructions =
        "You are a helpful assistant that answers trivia questions concisely."

    private static func parentTranscript() -> Transcript {
        Transcript(entries: [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: sharedInstructions))],
                    toolDefinitions: []))
        ])
    }

    /// A "fork" transcript: the shared parent prefix, plus one new
    /// fork-specific user turn -- exactly what a forked session's first
    /// `respond()` call would receive.
    private static func forkTranscript(question: String) -> Transcript {
        Transcript(entries: [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: sharedInstructions))],
                    toolDefinitions: [])),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: question))],
                    responseFormat: nil)),
        ])
    }

    @Test(
        "prewarming a parent transcript makes a forked session's FIRST respond() round report cachedTokenCount > 0"
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func forkedSessionFirstRoundReusesPrewarmedParentPrefix() async throws {
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Prewarm the shared parent prefix BEFORE any fork's first respond()
        // -- mirroring the router prewarming the parent transcript once,
        // ahead of spinning up per-fork sessions.
        try await executor.populatePromptCacheChunks(
            model: model, transcript: Self.parentTranscript())

        // The fork's FIRST respond() call ever, for this model -- extends
        // the prewarmed parent prefix with its own new question.
        let forkRequest = makeExecutorRequest(
            transcript: Self.forkTranscript(question: "What is the capital of France?"),
            generationOptions: greedyOptions)
        let forkResult = try await respondCollectingTextAndUsage(
            executor, request: forkRequest, model: model)

        #expect(!forkResult.text.isEmpty)
        #expect(
            forkResult.cachedTokenCount > 0,
            """
            the fork's FIRST respond() round should reuse the prewarmed shared parent \
            prefix -- cachedTokenCount == 0 means prewarm never populated the chunk store, \
            or the fork's own tokenization diverged from the prewarmed prefix
            """
        )

        await releaseAllGPUMemory()
    }

    @Test(
        "prewarming the identical transcript twice does not error, and a fork afterward still reuses the prefix"
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func prewarmingSameTranscriptTwiceStaysHealthy() async throws {
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Prewarm the identical parent transcript twice in a row -- the
        // second call should dedup (near-zero additional work), and neither
        // call should throw.
        try await executor.populatePromptCacheChunks(
            model: model, transcript: Self.parentTranscript())
        try await executor.populatePromptCacheChunks(
            model: model, transcript: Self.parentTranscript())

        let forkRequest = makeExecutorRequest(
            transcript: Self.forkTranscript(question: "What is the capital of Spain?"),
            generationOptions: greedyOptions)
        let forkResult = try await respondCollectingTextAndUsage(
            executor, request: forkRequest, model: model)

        #expect(!forkResult.text.isEmpty)
        #expect(
            forkResult.cachedTokenCount > 0,
            "a double-prewarmed prefix must still be reusable by a fork's first respond() round"
        )

        await releaseAllGPUMemory()
    }

    @Test("prewarming a multimodal transcript skips chunk population without throwing")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func prewarmingMultimodalTranscriptSkipsWithoutError() async throws {
        let model = makeTestModel(
            TestFixtures.defaultModelID,
            capabilities: [.vision, .guidedGeneration, .toolCalling])

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        let image = makeSolidCGImageForPrewarmTest()
        let imageAttachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(image)), label: "photo")
        let multimodalTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "What color is this image?")),
                        .attachment(imageAttachment),
                    ],
                    responseFormat: nil))
        ])

        let executor = try makeMLXExecutor(for: model)
        // Must complete without throwing -- the `isTextOnly` gate skips
        // chunk population for multimodal input; it must never surface as
        // an error to a caller that only wanted a best-effort prewarm.
        // Reaching the end of this test IS the proof: a Swift `#expect`
        // has nothing meaningful left to check once the only failure mode
        // under test (an escaping throw) has already not happened.
        try await executor.populatePromptCacheChunks(model: model, transcript: multimodalTranscript)

        await releaseAllGPUMemory()
    }
}

/// A tiny solid-color `CGImage`, built in-memory, for a valid labeled image
/// attachment without any binary fixture. Self-contained here (rather than
/// reusing a shared helper) since this xcodeproj-based target doesn't share
/// sources with the SwiftPM `Tests/` target -- mirrors
/// `PromptCacheMultimodalBoundaryTests.swift`'s own
/// `makeSolidCGImageForCacheBoundaryTest`.
private func makeSolidCGImageForPrewarmTest(width: Int = 2, height: Int = 2) -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(CGColor(red: 0.3, green: 0.5, blue: 0.7, alpha: 1.0))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
}

#endif  // FoundationModelsIntegration
