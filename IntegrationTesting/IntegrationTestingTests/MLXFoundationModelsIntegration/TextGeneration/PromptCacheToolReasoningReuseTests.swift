// Copyright © 2026 Apple Inc.
//
// Integration tests closing a real gap flagged by a third adversarial
// verification round (2026-07-12): every other `PromptCache*Tests` file in
// this directory exercises only the plain-text `respond()` path -- zero
// integration coverage exercised a tool-calling round or a reasoning
// (`<think>`) round, even though both commit to the cache through the SAME
// `Executor.commitPromptCache(...generatedTokenIDs:)` overload the plain
// path uses (`runToolCallReasoningPhase`/`runGuidedGenerationLoop` for
// tools, `runReasoning`/the reasoning phase of the two-phase think-then-call
// flow for reasoning), just via a different generation loop upstream of it.
// A regression specific to either of those code paths committing a bad or
// empty cache entry would have gone undetected by every existing
// `PromptCache*Tests` suite.
//
// Round 2 in each test below replays round 1's OWN real, model-generated
// output verbatim (the tool call + a synthetic tool result for the
// tool-calling test; just the final answer for the reasoning test, since
// `TranscriptConverter` intentionally does NOT replay `.reasoning` entries
// into chat history -- see its own doc comment) before a new question, so
// round 2's rendered prefix is a byte-for-byte continuation of round 1's
// stored tokens, mirroring `PromptCacheReuseTests`/`PromptCacheForkReuseTests`.
//
// Loads real models, so this lives in the IntegrationTesting xcodeproj
// alongside the other `TextGeneration` tests that exercise `respond()`
// end-to-end, sharing `Support/FMTestHelpers.swift`'s `executeResponse`/
// `reflectedChannelPayload` (also used by `ToolCalling/ToolCallingReasoningTests`,
// whose `Models.qwen3` reasoning-capable model id this file's reasoning test
// mirrors -- duplicated locally rather than referenced across files since
// `private` members are file-scoped and each `PromptCache*Tests` file is
// already self-contained by this directory's own convention (e.g.
// `PromptCachePrewarmTests`' and `PromptCacheMultimodalBoundaryTests`' each
// carry their own near-identical `CGImage` fixture helper).
//
// `PromptCache` is a single process-global actor keyed by `modelID`, so each
// test's rounds only observe each other's cache entry if nothing else
// touches the same model id concurrently -- run in isolation from other
// suites touching the same model ids for a reliable result (`.serialized`
// only orders this suite's own tests against each other).

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// `@Generable`'s macro-expanded conformance extension is emitted at file
/// scope, so this must be a top-level (not suite-nested) type -- mirrors
/// `ToolCalling/ToolCallingReasoningTests.swift`'s own top-level
/// `WeatherArgs`, duplicated locally since it is `private` there
/// (file-scoped).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct PromptCacheToolReasoningWeatherArgs {
    // Constrained to a small closed set (`.anyOf`), not a free-form
    // `@Guide(description:)` string: empirically, this suite's usual
    // `defaultModelID` (Qwen2.5-3B-Instruct-4bit) AND `Qwen3-1.7B-4bit`
    // both drove an UNCONSTRAINED string field's grammar-constrained decode
    // into a degenerate single-character repetition loop that never
    // terminated (confirmed by real runs; a pre-existing, already-failing
    // test elsewhere in this suite --
    // `FoundationModelsToolCallingTests.toolAwarePromptRoutesWeatherQueryToGetWeather`
    // -- shows this is a real, pre-existing tool-calling reliability gap,
    // not specific to this file). A closed enum of short literal strings
    // keeps the grammar's choice set small enough to sidestep that failure
    // mode entirely, which is all this test needs: SOME real tool call, to
    // exercise the prompt-cache commit path a tool round takes.
    @Guide(.anyOf(["Tokyo", "Paris", "New York"]))
    var location: String
}

/// Integration tests proving a tool-calling round and a reasoning round each commit a genuinely reusable prompt-cache entry, magnitude-bounded like every other `PromptCache*Tests` file.
@Suite(
    .serialized, .timeLimit(.minutes(15)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheToolReasoningReuseTests {

    /// The reasoning-capable model id `ToolCalling/ToolCallingReasoningTests`
    /// already exercises (Qwen3's template renders tools AND honors
    /// `enable_thinking`) -- duplicated here rather than referenced across
    /// files since `Models.qwen3` is `private` there (file-scoped).
    private static let reasoningModelID = "mlx-community/Qwen3-1.7B-4bit"

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func weatherTool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather in a given location. "
                + "Use this whenever the user asks about weather, temperature, or conditions.",
            parameters: PromptCacheToolReasoningWeatherArgs.generationSchema)
    }

    /// Streams a response, capturing the first tool call's id/name/arguments
    /// alongside the same `(promptTokenCount, cachedTokenCount,
    /// outputTokenCount)` usage triple `respondCollectingTextAndUsage`
    /// reports -- that shared helper only looks at the "response" event's
    /// `appendText`/`updateUsage` actions, so it never observes a
    /// `toolCalls` event; this local variant adds that without duplicating
    /// the usage-extraction logic's shape.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ToolCallAndUsage {
        var responseText = ""
        var toolCallID: String?
        var toolCallName: String?
        var toolArgsJSON = ""
        var promptTokenCount = 0
        var cachedTokenCount = 0
        var outputTokenCount = 0
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func collectToolCallAndUsage(_ stream: TestResponseStream) async throws
        -> ToolCallAndUsage
    {
        var result = ToolCallAndUsage()
        for try await event in stream {
            if let t = reflectedChannelPayload(
                of: event, caseLabel: "toolCalls",
                as: LanguageModelExecutorGenerationChannel.ToolCalls.self),
                let toolCall = reflectedChannelPayload(
                    of: t.action, caseLabel: "toolCall",
                    as: LanguageModelExecutorGenerationChannel.ToolCalls.ToolCall.self)
            {
                if result.toolCallName == nil {
                    result.toolCallName = toolCall.name
                    result.toolCallID = toolCall.id
                }
                if let argsDelta = reflectedChannelPayload(
                    of: toolCall.action, caseLabel: "appendArguments",
                    as: LanguageModelExecutorGenerationChannel.ToolCalls.ToolCall.ArgumentsFragment
                        .self)
                {
                    result.toolArgsJSON += argsDelta.content
                }
            } else if let r = reflectedChannelPayload(
                of: event, caseLabel: "response",
                as: LanguageModelExecutorGenerationChannel.Response.self)
            {
                if let fragment = reflectedChannelPayload(
                    of: r.action, caseLabel: "appendText",
                    as: LanguageModelExecutorGenerationChannel.TextFragment.self)
                {
                    result.responseText += fragment.content
                }
                if let usage = reflectedChannelPayload(
                    of: r.action, caseLabel: "updateUsage",
                    as: LanguageModelExecutorGenerationChannel.Usage.self)
                {
                    result.promptTokenCount = usage.input.totalTokenCount
                    result.cachedTokenCount = usage.input.cachedTokenCount
                    result.outputTokenCount = usage.output.totalTokenCount
                }
            }
        }
        return result
    }

    @Test(
        "A tool-calling round's real tool call + synthetic tool result, replayed before a new question, reuses round 1's prompt cache on round 2"
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func toolCallingRoundReusesPromptCacheOnRoundTwo() async throws {
        // `TestFixtures.defaultModelID` (Qwen2.5-3B-Instruct-4bit), this
        // suite's usual model, empirically drove this exact tool call's
        // grammar-constrained string-argument decode into a degenerate
        // repetition loop (confirmed by two real runs, greedy AND default
        // sampling: 256 tokens of a single repeating digit inside the
        // `location` value, never terminating). `Models.qwen3` with
        // thinking explicitly disabled is `ToolCalling
        // /ToolCallingReasoningTests`' own known-working single-phase
        // tool-calling combination (`qwen3ThinkingDisabledStaysSinglePhase`);
        // reused here rather than `defaultModelID` to get a reliable real
        // tool call instead of chasing an unrelated model-specific decode bug.
        let model = makeTestModel("mlx-community/Qwen3-1.7B-4bit")
        let executor = try makeMLXExecutor(for: model)
        var toolContextOptions = ContextOptions()
        toolContextOptions.reasoningLevel = .custom("no_think")
        let toolOptions = GenerationOptions(maximumResponseTokens: 256)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        let weatherQuestion = "What's the weather in Tokyo?"
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: weatherQuestion))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(
            transcript: firstTranscript,
            enabledTools: [Self.weatherTool()],
            generationOptions: toolOptions,
            contextOptions: toolContextOptions)
        let first = try await collectToolCallAndUsage(
            try await executeResponse(executor, request: firstRequest, model: model))

        let toolCallName = try #require(
            first.toolCallName, "expected round 1 to produce a real tool call")
        #expect(!first.toolArgsJSON.isEmpty, "expected non-empty tool call arguments")
        #expect(first.promptTokenCount > 0, "First round's prompt token count should be positive")

        // The FULL length of what round 1 actually stores (prompt tokens +
        // its own real generated reply tokens, i.e. the tool-call
        // envelope) -- same yardstick `PromptCacheReuseTests`/
        // `PromptCacheForkReuseTests` use.
        let sharedPrefixTokens = first.promptTokenCount + first.outputTokenCount

        // Round 2 replays round 1's REAL tool call + a synthetic tool
        // result before a brand-new question -- exactly what a real
        // `LanguageModelSession` continuation looks like after executing
        // the tool and feeding its result back.
        let toolCall = Transcript.ToolCall(
            id: first.toolCallID ?? UUID().uuidString,
            toolName: toolCallName,
            arguments: try GeneratedContent(json: first.toolArgsJSON))
        let toolOutput = Transcript.ToolOutput(
            id: toolCall.id, toolName: toolCallName,
            segments: [.text(Transcript.TextSegment(content: "68F and sunny"))])
        let secondTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: weatherQuestion))],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls([toolCall])),
            .toolOutput(toolOutput),
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(Transcript.TextSegment(content: "Now say 'done' in one word."))
                    ],
                    responseFormat: nil)),
        ])
        let secondRequest = makeExecutorRequest(
            transcript: secondTranscript,
            enabledTools: [Self.weatherTool()],
            generationOptions: toolOptions,
            contextOptions: toolContextOptions)
        let second = try await collectToolCallAndUsage(
            try await executeResponse(executor, request: secondRequest, model: model))

        // Magnitude-bounded, not just `> 0`: one cached token out of a much
        // longer prefix would satisfy `> 0` but prove nothing. Unlike the
        // plain-text/reasoning continuations elsewhere in this file (which
        // measured essentially zero slack, since they replay round 1's own
        // decoded text verbatim), a tool-calling round 2 replays a
        // RECONSTRUCTED envelope: `TranscriptConverter` re-renders the
        // `.toolCalls` entry as `{"name": ..., "arguments": ...}` from
        // `Transcript.ToolCall.arguments`'s canonical `GeneratedContent
        // .jsonString`, not the raw streamed text round 1 actually
        // generated -- confirmed by a real run to differ enough in
        // formatting to shift token boundaries for part of the tool-call
        // span. `sharedPrefixTokens / 10` (measured real gap: 27 of 314,
        // i.e. ~9%) is a generous, data-derived slack for that
        // reconstruction mismatch, while still requiring the bulk of the
        // prefix -- prompt AND tool-call span -- to be genuinely reused.
        let toolEnvelopeReconstructionSlack = sharedPrefixTokens / 10
        #expect(
            second.cachedTokenCount >= sharedPrefixTokens - toolEnvelopeReconstructionSlack,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should be within \
            \(toolEnvelopeReconstructionSlack) tokens of round 1's full stored prefix \
            (\(sharedPrefixTokens) = \(first.promptTokenCount) prompt + \
            \(first.outputTokenCount) output tokens) -- round 1 committed via the tool-calling \
            generation path's `commitPromptCache(...generatedTokenIDs:)` call, and this proves \
            that entry is both present AND substantially reused, not merely nonzero
            """
        )

        await releaseAllGPUMemory()
    }

    @Test(
        "A reasoning round's final answer, replayed before a new question, reuses round 1's prompt cache on round 2"
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func reasoningRoundReusesPromptCacheOnRoundTwo() async throws {
        let model = makeReasoningTestModel(Self.reasoningModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 512)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Padded well past `PromptCache.defaultChunkSize` (64): unlike a
        // plain-text or tool-calling round, a reasoning round's `<think>`
        // content is real, substantial generated output that
        // `TranscriptConverter` intentionally never replays into chat
        // history (only the final answer survives to round 2 -- see its own
        // doc comment). Confirmed by a real run: with a short (~17-token)
        // prompt, round 1's full stored sequence (prompt + `<think>` chain +
        // answer, well over one chunk) has NO chunk boundary that falls
        // entirely within the shared (replayed) prompt region, so round 2 --
        // which replaces the `<think>` chain with a short final-answer
        // replay -- diverges from round 1's stored tokens WITHIN the very
        // first chunk and gets `cachedTokenCount == 0`, not just a small
        // one. Padding the prompt out past one full chunk guarantees at
        // least that leading chunk is pure shared prompt content in BOTH
        // rounds, independent of how long or short the model's own
        // `<think>` chain runs -- the magnitude bound below is tied to
        // exactly that guaranteed chunk, not to the full (mostly
        // unreplayed) round 1 stored length.
        let paddingSentence =
            "This is filler context padding the prompt length past one prompt-cache chunk boundary. "
        let firstUserText =
            String(repeating: paddingSentence, count: 8) + "Say 'hi' in exactly one word."
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(
            transcript: firstTranscript, generationOptions: greedyOptions)
        let first = try await respondCollectingTextAndUsage(
            executor, request: firstRequest, model: model)

        #expect(!first.text.isEmpty, "First round should produce some response text")
        #expect(first.promptTokenCount > 0, "First round's prompt token count should be positive")

        // The number of tokens GUARANTEED reusable on round 2: every full
        // `chunkSize`-token span that falls entirely within the padded
        // prompt (shared verbatim by both rounds), floored down from the
        // measured prompt length. Anything past this floor is round 1's
        // OWN `<think>` chain + answer, which round 2 does not replay, so
        // it is deliberately excluded from the yardstick (unlike
        // `PromptCacheReuseTests`/`PromptCacheForkReuseTests`, where the
        // FULL prior round is replayed and is therefore the yardstick).
        let sharedPrefixTokens =
            (first.promptTokenCount / PromptCache.defaultChunkSize) * PromptCache.defaultChunkSize
        #expect(
            sharedPrefixTokens >= PromptCache.defaultChunkSize,
            """
            expected the padded prompt (\(first.promptTokenCount) tokens) to span at least one \
            full \(PromptCache.defaultChunkSize)-token chunk -- otherwise this test cannot prove \
            anything beyond the already-covered short-prompt case
            """
        )

        // Round 2 replays round 1's own real reply verbatim before a new
        // question. `TranscriptConverter` intentionally does NOT replay
        // `.reasoning` entries into chat history (see its own doc comment),
        // so only the final answer -- never the `<think>` content -- needs
        // replaying here, exactly like a real `LanguageModelSession`
        // continuation would look.
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
            transcript: secondTranscript, generationOptions: greedyOptions)
        let second = try await respondCollectingTextAndUsage(
            executor, request: secondRequest, model: model)
        #expect(!second.text.isEmpty, "Second round should produce some response text")

        // Magnitude-bounded, not just `> 0`: one cached token out of a much
        // longer prompt would satisfy `> 0` but prove nothing. `- 1` is a
        // small buffer for boundary/tokenizer variance around the floor
        // computed above.
        #expect(
            second.cachedTokenCount >= sharedPrefixTokens - 1,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should be within one \
            token of the padded prompt's guaranteed-shared chunk floor (\(sharedPrefixTokens), \
            from a \(first.promptTokenCount)-token prompt) -- round 1 committed via the \
            reasoning generation path's own `commitPromptCache(...generatedTokenIDs:)` call, and \
            this proves that entry is both present AND substantially reused within the region \
            round 2 actually replays, not merely nonzero
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
