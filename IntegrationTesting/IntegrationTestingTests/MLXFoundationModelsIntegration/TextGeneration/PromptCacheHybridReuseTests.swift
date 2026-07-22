// Copyright © 2026 Apple Inc.
//
// End-to-end proof that a real HYBRID Mamba/attention model (Qwen3.6, a
// genuine `Qwen35Model` per `Libraries/MLXLLM/LLMModelFactory.swift`) reports
// prompt-cache reuse through the full `respond()` -> `commitPromptCache` ->
// `store()` -> `resolve()` pipeline -- the same pipeline
// `PromptCacheReuseTests.swift` already proves for the pure-attention
// `mlx-community/Qwen2.5-3B-Instruct-4bit`. The hybrid Mamba/Qwen3.5/Qwen3Next
// architecture family is proven at the UNIT level with real model classes but
// only synthetic random weights (`PromptCacheHybridArchitectureTests.swift`);
// this test closes the gap by running an actual hybrid checkpoint end-to-end.
//
// Loads a real model, so it lives in the IntegrationTesting xcodeproj
// alongside the other `TextGeneration` tests that exercise `respond()`
// end-to-end. Uses `Support/FMTestHelpers.swift`'s shared collectors
// (`respondCollectingTextAndUsage` / `respondCollectingReasoningTextAndUsage`),
// also used by `PromptCacheReuseTests`, `PromptCacheEquivalenceTests`, and
// `PromptCacheGuidedRoundTripTests`.
//
// `PromptCache` is a single process-global actor keyed by `modelID` (mirrors
// `MLXLanguageModel`'s `ModelCache`), so these rounds only observe each
// other's cache entry if nothing else touches the same model id concurrently
// -- run in isolation from other suites touching
// `TestFixtures.qwen36HybridModelID` for a reliable result (`.serialized`
// only orders this suite's own tests against each other).
//
// IMPORTANT -- the assertions below intentionally document a real,
// structural bound for these two-round scenarios, not an aspirational
// ceiling; do not casually loosen or tighten them without re-deriving why
// from the chat-template mechanism explained at the assertion sites.
//
// HISTORY of this bound (each stage verified against the real model):
//
// 1. Originally asserted the pure-attention bound
//    (`cachedTokenCount >= prompt + output - 1`) and failed: Qwen3.6's chat
//    template renders a turn's generation region differently live vs. as
//    history, so round 1's FED token sequence was never a prefix of round
//    2's re-render (token-level trace on kanban er33v06).
// 2. kanban er33v06 added the transcript-stable-boundary split prefill
//    (`MLXLanguageModel.Executor.makePromptCacheSlot`): each hybrid round
//    also snapshots a checkpoint at the exact prefix future rounds
//    re-render verbatim, BEFORE any generation-region tokens touch the
//    cache. That made the PROMPT prefix reusable
//    (`cachedTokenCount >= round 1 promptTokenCount - 8`), but round 1's
//    generated reply and priming tokens stayed behind the divergence.
// 3. kanban 05zt40g closed the remaining gap using the template's own
//    `preserve_thinking` kwarg: history renders now keep each assistant
//    turn's `<think>…</think>` block (`ReasoningConfig
//    .historyPreservationKey`), and `TranscriptConverter` replays prior
//    reasoning as `reasoning_content`. With that, a past turn re-renders as
//    `<|im_start|>assistant\n<think>\n{reasoning|trim}\n</think>\n\n{content}`
//    -- for a thinking-suppressed round (empty reasoning) that is exactly
//    the `<think>\n\n</think>\n\n` priming block round 1 actually fed, and
//    for a thinking round it is the reasoning the model actually generated.
//    Round 1's full fed sequence (prompt + generated reply) therefore
//    reappears verbatim at the start of round 2's re-render, and the strong
//    bound (`cachedTokenCount >= prompt + output - slack`) is achievable
//    for BOTH modes. The assertions below hold it there.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration test proving a second `respond()` round on a real hybrid
/// Mamba/attention model prefills only the appended suffix -- through round
/// 1's own generated response, not just its prompt -- in both
/// thinking-suppressed and thinking modes.
@Suite(
    .serialized, .timeLimit(.minutes(20)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheHybridReuseTests {

    /// The slack the strong bound allows below `prompt + output`: covers the
    /// re-rendered turn seams (the closing `<|im_end|>\n` after the replayed
    /// reply, the template's `|trim` of a reply's trailing whitespace, and a
    /// stop-token accounting difference between `outputTokenCount` and the
    /// stored checkpoint) without ever excusing a fallback to the
    /// prompt-only stable-boundary checkpoint, which would fall short by
    /// the whole generated reply.
    private static let strongBoundSlack = 8

    // Previously a `guard #available(...) else { return }` at the top of
    // the test body: on an OS below the 27 floor, that made the test
    // report PASSED with zero assertions ever run -- a silent no-op
    // indistinguishable from a real pass in CI output. The `@available`
    // attribute directly on the function (below) makes the compiler enforce
    // the same floor statically, and the suite's `.enabled(if:)` trait
    // (above) makes swift-testing report this test as explicitly SKIPPED
    // (not passed) when the floor isn't met, so a "green" run can no longer
    // hide an unmet OS requirement.
    @Test("Second respond() round reuses round 1's prompt AND reply (thinking suppressed)")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func secondRoundReusesPromptAndReplyOnHybridModel() async throws {
        // No `.reasoning` capability declared: the executor renders
        // reasoning-SUPPRESSED prompts for this family (`enable_thinking:
        // false`), whose priming block `<think>\n\n</think>\n\n` is exactly
        // what a preserved-thinking history turn with empty
        // `reasoning_content` re-renders.
        let model = makeTestModel(TestFixtures.qwen36HybridModelID)
        let executor = try makeMLXExecutor(for: model)

        // Capability signal: the loaded container's model must report
        // support for prompt-cache reuse via the hybrid-checkpoint path
        // (`PromptCache.isHybridMambaAttention`), even though its cache is
        // not chunkable (`PromptCache.isChunkable == false` for hybrid
        // stacks). Computed inside `perform` because `context.model` (`any
        // MLXLMCommon.LanguageModel`) is not `Sendable` and must not cross
        // the actor boundary itself.
        let container = try await model.loadContainer()
        let supportsReuse = await container.perform { context in
            MLXLanguageModel.supportsPromptCacheReuse(model: context.model)
        }
        #expect(
            supportsReuse,
            "A real hybrid Qwen3.6 checkpoint should report prompt-cache reuse support")

        // Enough budget for the model to CLOSE the reply with its own EOS
        // (the er33v06-era budget of 8 cut generation mid-stream): a
        // finished reply retokenizes cleanly as history and carries no
        // trailing whitespace for the template's `|trim` to eat, which is
        // what makes the full fed sequence re-appear verbatim.
        let generationOptions = GenerationOptions(maximumResponseTokens: 24)

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
        print("[hybrid-reuse suppressed] round 1 text=<<<\(first.text)>>>")

        // Round 1's full stored slot length (prompt tokens + its own real
        // generated reply tokens) -- with history preservation this IS the
        // reuse yardstick, exactly like `PromptCacheReuseTests`'
        // pure-attention counterpart.
        let sharedPrefixTokens = first.promptTokenCount + first.outputTokenCount

        // Second round's transcript replays the first round's own answer as
        // an assistant entry (mirroring what a real `LanguageModelSession`
        // does) and appends one new user turn. Without cache reuse this
        // prefill is strictly longer than the first round's; with it,
        // `respond()` should feed only the closing turn seam + new user turn.
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
        print("[hybrid-reuse suppressed] round 2 text=<<<\(second.text)>>>")

        // THE STRONG BOUND (see the file header's bound history): with
        // `preserve_thinking` + `reasoning_content` replay (kanban 05zt40g),
        // round 2's re-render of round 1's turn is byte-continuous with what
        // round 1 actually fed -- suppressed rounds re-render the exact
        // `<think>\n\n</think>\n\n` priming block -- so the post-round
        // hybrid checkpoint covering ALL of round 1's fed tokens (prompt +
        // reply) must match, not just the er33v06 prompt-only
        // stable-boundary checkpoint. A fallback to the stable-boundary
        // checkpoint would miss the whole reply and fail this bound.
        #expect(
            second.cachedTokenCount >= sharedPrefixTokens - Self.strongBoundSlack
                && second.cachedTokenCount > 0,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should cover round \
            1's FULL fed sequence: its prompt (\(first.promptTokenCount)) plus its generated \
            reply (\(first.outputTokenCount)), minus a \(Self.strongBoundSlack)-token seam \
            allowance -- the strong bound history preservation makes reachable. A value near \
            \(first.promptTokenCount) alone means the post-round checkpoint failed to match \
            and resolve fell back to the prompt-only stable-boundary checkpoint.
            """
        )

        await releaseAllGPUMemory()
    }

    @Test("Second respond() round reuses round 1's prompt AND reply (thinking mode)")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func secondRoundReusesPromptAndReplyWithThinking() async throws {
        // `.reasoning` declared: the executor renders a reasoning-PRIMED
        // prompt (`<think>\n` tail) and streams the chain-of-thought into a
        // `.reasoning` transcript entry. Round 2 replays that entry, so the
        // preserved-thinking history turn re-renders the SAME reasoning
        // tokens round 1 actually generated.
        let model = makeReasoningTestModel(TestFixtures.qwen36HybridModelID)
        let executor = try makeMLXExecutor(for: model)

        // Enough budget for the model to finish thinking (`</think>`) AND
        // close its reply with EOS -- a cut-off-inside-<think> round has no
        // response text to replay and would invalidate the scenario (the
        // `#expect`s below catch that explicitly rather than masking it).
        let generationOptions = GenerationOptions(maximumResponseTokens: 640)

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
        let first = try await respondCollectingReasoningTextAndUsage(
            executor, request: firstRequest, model: model)
        #expect(
            !first.reasoning.isEmpty,
            "First round should think (reasoning-primed prompt, .reasoning declared)")
        #expect(
            !first.text.isEmpty,
            "First round should close </think> and produce response text within the budget")
        print("[hybrid-reuse thinking] round 1 reasoning=<<<\(first.reasoning)>>>")
        print("[hybrid-reuse thinking] round 1 text=<<<\(first.text)>>>")

        let sharedPrefixTokens = first.promptTokenCount + first.outputTokenCount

        // Round 2 replays round 1's reasoning AND answer -- exactly the
        // entries the framework's own transcript accrues (the executor
        // streams `.reasoning` immediately before its `.response`) -- then
        // appends one new user turn.
        let secondTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil)),
            .reasoning(
                Transcript.Reasoning(
                    segments: [.text(Transcript.TextSegment(content: first.reasoning))])),
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
        let second = try await respondCollectingReasoningTextAndUsage(
            executor, request: secondRequest, model: model)
        #expect(!second.text.isEmpty, "Second round should produce some response text")
        print("[hybrid-reuse thinking] round 2 reasoning=<<<\(second.reasoning)>>>")
        print("[hybrid-reuse thinking] round 2 text=<<<\(second.text)>>>")

        // Same strong bound as the suppressed test above: the preserved
        // history turn re-renders `<think>\n{reasoning|trim}\n</think>\n\n
        // {content}` -- byte-continuous with round 1's primed `<think>\n`
        // tail plus everything it generated -- so round 2 must reuse round
        // 1's prompt AND its whole generated output (reasoning + reply),
        // within the seam allowance.
        #expect(
            second.cachedTokenCount >= sharedPrefixTokens - Self.strongBoundSlack
                && second.cachedTokenCount > 0,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should cover round \
            1's FULL fed sequence: its prompt (\(first.promptTokenCount)) plus its generated \
            reasoning + reply (\(first.outputTokenCount)), minus a \(Self.strongBoundSlack)-token \
            seam allowance. A value near \(first.promptTokenCount) alone means the replayed \
            `reasoning_content` did not re-render byte-identically to what round 1 generated \
            (retokenization seam), so the post-round checkpoint failed to match.
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
