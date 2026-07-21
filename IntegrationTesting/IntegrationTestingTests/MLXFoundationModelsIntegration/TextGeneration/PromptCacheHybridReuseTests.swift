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
// end-to-end. Uses `Support/FMTestHelpers.swift`'s shared
// `respondCollectingTextAndUsage` (mirroring `UpdateUsageEmissionTests`'
// `collectFinalUsage`), also used by `PromptCacheReuseTests`,
// `PromptCacheEquivalenceTests`, and `PromptCacheGuidedRoundTripTests`.
//
// `PromptCache` is a single process-global actor keyed by `modelID` (mirrors
// `MLXLanguageModel`'s `ModelCache`), so this test's two rounds only observe
// each other's cache entry if nothing else touches the same model id
// concurrently -- run in isolation from other suites touching
// `TestFixtures.qwen36HybridModelID` for a reliable result (`.serialized`
// only orders this suite's own tests against each other).
//
// IMPORTANT -- the assertion below intentionally documents a real,
// structural bound for THIS two-round scenario, not an aspirational ceiling;
// do not casually loosen or tighten it without re-deriving why from the
// chat-template mechanism explained at the assertion site.
//
// This test originally asserted `cachedTokenCount >= sharedPrefixTokens - 1`
// (mirroring `PromptCacheReuseTests`' pure-attention bound), on the
// assumption that a hybrid round's only reuse hazard was the EOS-trim
// degradation: a round whose cache lands in
// `PromptCache.reconcileCacheAdvance`'s `.trimCacheByOne` case used to be
// DROPPED entirely, because hybrid checkpoints cannot be trimmed
// (`MambaCache.isTrimmable == false`), unlike pure attention. That bug is
// FIXED (`PromptCache.planCacheStore`'s `.storeExtended` case), but
// real-model verification against `mlx-community/Qwen3.6-27B-mxfp4` proved
// it was never this test's failure: round 1 here terminates by hitting
// `maximumResponseTokens` (a budget cutoff, not EOS), reconciles as
// `.matches`, and stores cleanly on its own. The real cause is the
// chat-template mechanism documented at the assertion below: Qwen3.6
// renders a turn's generation region differently live vs. as history, so
// round 1's FED token sequence is never a prefix of round 2's re-render.
//
// The fix for THAT (kanban er33v06) is the transcript-stable-boundary
// split prefill in `MLXLanguageModel.Executor.makePromptCacheSlot`: each
// hybrid round now also snapshots a checkpoint at the exact prefix future
// rounds re-render verbatim, BEFORE any generation-region tokens touch the
// cache. Full-prefix reuse (`>= prompt + output - 1`) remains structurally
// UNACHIEVABLE for this template family -- the correct maximum is the
// stable prefix, which is what the bound below asserts.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration test proving a second `respond()` round on a real hybrid
/// Mamba/attention model prefills only the appended suffix, not the whole
/// (now-longer) transcript.
@Suite(
    .serialized, .timeLimit(.minutes(20)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheHybridReuseTests {

    // Previously a `guard #available(...) else { return }` at the top of
    // the test body: on an OS below the 27 floor, that made the test
    // report PASSED with zero assertions ever run -- a silent no-op
    // indistinguishable from a real pass in CI output. The `@available`
    // attribute directly on the function (below) makes the compiler enforce
    // the same floor statically, and the suite's `.enabled(if:)` trait
    // (above) makes swift-testing report this test as explicitly SKIPPED
    // (not passed) when the floor isn't met, so a "green" run can no longer
    // hide an unmet OS requirement.
    @Test("Second respond() round on a real hybrid model prefills only the appended suffix")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func secondRoundPrefillsOnlyAppendedSuffixOnHybridModel() async throws {
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
        // its own real generated reply tokens) -- reported in the assertion
        // message below purely for context (how much of round 1 is stored
        // vs. how much of it round 2 can actually reuse), unlike
        // `PromptCacheReuseTests.secondRoundPrefillsOnlyAppendedSuffix`'s
        // pure-attention counterpart, where this quantity IS the reuse
        // yardstick.
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

        // Round 2 replays round 1's OWN real reply verbatim before its new
        // question, but its rendered prefix is NOT a byte-for-byte
        // continuation of what round 1 FED the model: Qwen3.6's chat
        // template renders the SAME assistant turn differently depending on
        // whether it is being generated live or replayed as history:
        //
        //   - Round 1's generation prompt ends in a generation-priming
        //     region after the final assistant header (`<|im_start|>
        //     assistant\n` + the thinking-suppression injection `[248068,
        //     198]` -- the executor renders reasoning-suppressed prompts
        //     for this family, see `ReasoningConfig`'s qwen3 row).
        //   - Round 2 renders that SAME assistant turn as HISTORY, with no
        //     generation-region tokens at all, and appends its OWN priming
        //     suffix after its new final user turn instead.
        //
        // Verified empirically against the real model (token-level trace on
        // kanban er33v06): round 1's fed prompt and round 2's re-render
        // share everything through `...assistant\n` (17 of round 1's 19
        // prompt tokens), then diverge at the injected priming tokens. So a
        // checkpoint keyed on round 1's FULL fed sequence can never match a
        // later round -- Mamba state cannot be trimmed backward past the
        // baked-in priming tokens, and `PromptCache.resolveHybridCheckpoint`
        // only reuses a checkpoint on an exact, WHOLE-checkpoint prefix
        // match.
        //
        // The fix (kanban er33v06): `MLXLanguageModel.Executor
        // .makePromptCacheSlot` computes the transcript-stable boundary --
        // the same messages re-rendered as past turns via
        // `applyChatTemplate(addGenerationPrompt: false)` -- and snapshots a
        // hybrid checkpoint exactly THERE, before the priming tokens touch
        // the cache. Round 2 must therefore reuse round 1's stable prefix:
        // its prompt tokens minus the assistant header + priming region
        // (empirically 2 tokens for this template; 8 is a safe allowance for
        // template evolution). Full-prefix reuse (`>= sharedPrefixTokens -
        // 1`, the pure-attention bound) remains structurally unreachable --
        // round 1's generated reply and priming tokens sit behind the
        // divergence, and there is no partial-prefix credit for a hybrid
        // checkpoint -- so the STABLE PREFIX is the correct maximum, and
        // this bound asserts it is actually achieved rather than settling
        // for a meaningless `> 0` alone.
        #expect(
            second.cachedTokenCount >= first.promptTokenCount - 8
                && second.cachedTokenCount > 0,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should cover round \
            1's transcript-stable prefix: at least its prompt token count \
            (\(first.promptTokenCount)) minus an 8-token allowance for the assistant header + \
            generation-priming region, and strictly positive. Round 1 stores TWO hybrid \
            checkpoints -- one at the transcript-stable boundary (pre-generation, see \
            MLXLanguageModel.Executor.makePromptCacheSlot) and one post-round covering all \
            \(sharedPrefixTokens) fed tokens (\(first.promptTokenCount) prompt + \
            \(first.outputTokenCount) output). The post-round one can never match round 2's \
            re-render (Qwen3.6 strips the generation-priming region from history turns), but \
            the stable-boundary one must.
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
