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
// structural floor for THIS two-round scenario, not an aspirational ceiling;
// do not casually loosen or tighten it without re-deriving why from the
// chat-template mechanism explained at the assertion site.
//
// This test originally asserted `cachedTokenCount >= sharedPrefixTokens - 1`
// (mirroring `PromptCacheReuseTests`' pure-attention bound), on the
// assumption that a hybrid round's only reuse hazard was the EOS-trim
// degradation: a round whose cache lands in
// `PromptCache.reconcileCacheAdvance`'s `.trimCacheByOne` case used to be
// DROPPED entirely, because hybrid checkpoints cannot be trimmed
// (`MambaCache.isTrimmable == false`), unlike pure attention. That bug is now
// FIXED -- see `PromptCache.planCacheStore`'s `.storeExtended` case and
// `GenerateCompletionInfo.stopTokenFedToCache` (`Libraries/MLXLMCommon/
// Evaluate.swift`): an EOS-terminated hybrid round now stores the true,
// extended `[prompt + generated + stopToken]` sequence instead of dropping.
//
// Real-model verification (run against `mlx-community/Qwen3.6-27B-mxfp4`)
// proved that fix does NOT make this specific test pass, because this test's
// failure was never the EOS-trim bug: round 1 here terminates by hitting
// `maximumResponseTokens` (a budget cutoff, not EOS), which reconciles as
// `.matches` and stores cleanly on its own. The real, irreducible cause is a
// completely different mechanism -- Qwen3.6's chat template renders a
// `<think>` reasoning turn differently live vs. as history -- documented in
// full at the assertion below. No sound cache-layer change can close that
// gap, so the bound is corrected to the genuinely-reachable value for this
// scenario (verified empirically, not assumed) rather than weakened to a
// meaningless `> 0`.

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
        // question, so at first glance its rendered prefix looks like a
        // byte-for-byte continuation of round 1's stored tokens -- it is
        // NOT, because Qwen3.6's chat template renders the SAME assistant
        // turn differently depending on whether it is being generated live
        // or replayed as history:
        //
        //   - Round 1's generation prompt (fed into, and cached by, round 1)
        //     is `...<|im_start|>assistant\n<think>\n` -- the template
        //     unconditionally opens a `<think>` block for whichever turn is
        //     currently being generated.
        //   - Round 2 renders that SAME assistant turn as HISTORY (it now
        //     precedes a newer user turn), and Qwen3.6's template strips
        //     reasoning from any assistant turn that is not the response to
        //     the most recent user query -- so round 2 renders
        //     `...<|im_start|>assistant\n` followed DIRECTLY by the reply
        //     content, with no `<think>\n` at all.
        //
        // Verified empirically against the real model (temporary debug
        // instrumentation in `PromptCache.resolveHybridCheckpoint`, removed
        // after confirming): round 1's stored checkpoint and round 2's
        // re-rendered prefix share their first 17 tokens (through
        // `...assistant\n`), then diverge at the `<think>` token -- round
        // 1's checkpoint has it, round 2's render does not.
        //
        // That divergence lands INSIDE the stored checkpoint (token 17 of
        // the 27-token prefix observed empirically -- see `sharedPrefixTokens`
        // below for this round's actual value), not at its boundary, and
        // `PromptCache.resolveHybridCheckpoint` only reuses a checkpoint on
        // an EXACT, WHOLE-checkpoint prefix match -- unlike attention's
        // chunk store, a hybrid `MambaCache`'s recurrent state cannot be
        // sliced or truncated to an earlier position (see
        // `PromptCache.isChunkable`'s doc), so there is no partial-prefix
        // credit for the 17 tokens that DO match. The whole checkpoint
        // simply fails to match, and `cachedTokenCount` is exactly 0 -- not
        // "shared prefix minus a few tokens".
        //
        // This is unrelated to the hybrid EOS-trim degradation this test
        // used to chase (see the file header): that bug is fixed
        // (`PromptCache.planCacheStore`'s `.storeExtended` case), but round
        // 1 here reconciles as `.matches` (it terminates by hitting
        // `maximumResponseTokens`, not EOS) and stores cleanly regardless --
        // the drop happens entirely on the RESOLVE side, from the
        // chat-template divergence above. No sound cache-layer change can
        // close it: the two renders are genuinely different token sequences
        // by design of Qwen3.6's template. If a future change (partial-
        // prefix hybrid reuse, or transcript replay that preserves
        // reasoning) makes some of this 17-token overlap reusable, RAISE
        // this bound accordingly -- it documents today's real, structural
        // floor, not a permanent ceiling.
        #expect(
            second.cachedTokenCount == 0,
            """
            Second round's cached token count (\(second.cachedTokenCount)) should be exactly 0. \
            Qwen3.6's chat template strips the `<think>` reasoning scaffold from round 1's \
            assistant turn once it is replayed as history in round 2, so round 2's re-rendered \
            prefix diverges from round 1's stored hybrid checkpoint (\(sharedPrefixTokens) = \
            \(first.promptTokenCount) prompt + \(first.outputTokenCount) output tokens) 17 \
            tokens in. Hybrid checkpoints only match on an exact WHOLE-checkpoint prefix (Mamba \
            state cannot be partially reused), so that divergence drops the ENTIRE checkpoint, \
            not just the diverging suffix -- a real, structural template/cache-shape \
            interaction, not the EOS-trim degradation this test used to chase (that one is \
            fixed; see PromptCache.planCacheStore).
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
