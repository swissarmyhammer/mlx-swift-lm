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
// Verified by real execution in this environment (2026-07-11 20:19 log;
// re-run at c624207, and again while tightening this file's assertions to a
// magnitude bound): all three tests pass repeatedly against a real MLX
// model --
//   xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting -destination 'platform=macOS' \
//     -only-testing:IntegrationTestingTests/PromptCachePrewarmTests

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import CoreGraphics
import MLX
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

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
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
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
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

        // The exact tokenized length of the prewarmed parent transcript --
        // measured through the SAME `TranscriptConverter` + `UserInput`
        // rendering pipeline `populatePromptCacheChunks` itself uses, so
        // this is a precise yardstick, not an estimate. Includes the
        // rendered prefix's trailing "generation prompt" (assistant-header)
        // tokens, which a fork's own render does NOT reproduce at that
        // position (it continues straight to a new user turn instead) --
        // see `sharedPrefixTokenSlack`'s doc for why that gap is the slack
        // this test's bound tolerates.
        let sharedPrefixTokens = try await measuredPromptTokenCount(
            for: Self.parentTranscript(), model: model)

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
        // Magnitude-bounded, not just `> 0`: one cached token out of a much
        // longer prefix would satisfy `> 0` but prove nothing.
        #expect(
            forkResult.cachedTokenCount >= sharedPrefixTokens - sharedPrefixTokenSlack,
            """
            the fork's FIRST respond() round cached \(forkResult.cachedTokenCount) tokens, \
            expected within \(sharedPrefixTokenSlack) of the prewarmed parent prefix's \
            \(sharedPrefixTokens) tokens -- cachedTokenCount == 0 means prewarm never \
            populated the chunk store, or the fork's own tokenization diverged from the \
            prewarmed prefix; a merely-positive bound could also pass with only a small \
            fraction of the prefix actually reused
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

        let sharedPrefixTokens = try await measuredPromptTokenCount(
            for: Self.parentTranscript(), model: model)

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
        // Magnitude-bounded (see `forkedSessionFirstRoundReusesPrewarmedParentPrefix`'s
        // identical check above for the slack's justification).
        #expect(
            forkResult.cachedTokenCount >= sharedPrefixTokens - sharedPrefixTokenSlack,
            """
            a double-prewarmed prefix must still be reusable by a fork's first respond() \
            round: cached \(forkResult.cachedTokenCount) tokens, expected within \
            \(sharedPrefixTokenSlack) of the prewarmed prefix's \(sharedPrefixTokens) tokens
            """
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

/// The exact tokenized length of `transcript` as rendered by the same
/// `TranscriptConverter.mlxMessages` + `UserInput(chat:)` pipeline
/// `Executor.populatePromptCacheChunks` itself uses -- a precise
/// measurement, not an estimate, of how many tokens prewarming `transcript`
/// actually stores.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func measuredPromptTokenCount(
    for transcript: Transcript, model: MLXLanguageModel
) async throws -> Int {
    let messages = TranscriptConverter.mlxMessages(for: transcript)
    let container = try await model.loadContainer()
    let input = try await container.prepare(input: UserInput(chat: messages))
    return input.text.tokens.asArray(Int.self).count
}

/// Slack tolerated between a prewarmed parent transcript's measured token
/// length and a fork's reported `cachedTokenCount` on its first round.
///
/// Not zero: `measuredPromptTokenCount` renders the parent transcript
/// ALONE, which ends in the chat template's trailing "generation prompt" /
/// assistant-header tokens (priming the model to reply next). A fork's own
/// transcript never has that header at the same position -- it continues
/// straight into a NEW USER turn instead (see `forkTranscript(question:)`)
/// -- so those trailing tokens are genuine, expected divergence, not a
/// caching defect. `PromptCache.resolve`'s own doc comment calls this out
/// by name for exactly this scenario: "the prewarm-stored sequence ends
/// with generation-prompt/assistant-header tokens that diverge from a
/// fork's next-turn render at that position -- an exact-whole-tail match
/// would still yield zero reuse for the prewarm scenario," which is why
/// longest-common-prefix (not exact-whole-tail) matching is what makes any
/// reuse possible here at all. Confirmed by a real run against
/// `TestFixtures.defaultModelID` (measured with `sharedPrefixTokenSlack`
/// temporarily set to `0`, to see the real gap and prove this bound can
/// actually fail): prewarming a 21-token instructions-only transcript
/// yields `cachedTokenCount == 19` on the fork's first round -- a
/// reproducible 2-token gap, matching the trailing generation-prompt
/// divergence described above. `4` (2x that measured gap) leaves headroom
/// for minor template/tokenizer variation while staying far tighter than
/// `PromptCache.defaultChunkSize` (64), which this file deliberately does
/// NOT use as its slack: the shared prefix here is itself well under one
/// chunk, so a 64-token slack would make the bound vacuous
/// (`sharedPrefixTokens - 64` going negative, satisfied by any
/// `cachedTokenCount >= 0`).
private let sharedPrefixTokenSlack = 4

#endif  // FoundationModelsIntegration
