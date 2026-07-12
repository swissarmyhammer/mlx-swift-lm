// Copyright © 2026 Apple Inc.
//
// Correctness-equivalence integration tests for `PromptCache`: the review
// that drove the multi-slot redesign (see kanban task `qawe2hb`) called
// this class of test "non-negotiable" -- unit tests on `PromptCache`'s pure
// chunk-matching functions (`lookupLongestPrefix`/`assemble`, since the
// chunk-path cutover in kanban `cthbfmw` replaced the slot-pool's
// `decide`/`selectSlot`) can prove the MATCHING logic is correct,
// but only a real model run can catch "wrong tokens attended to": a cache
// reuse that silently feeds the model a KV state that doesn't correspond
// to what it's being told the prompt is, producing plausible-looking but
// WRONG output with no crash and no error. Greedy sampling makes the
// comparison exact (no distributional slack): if the cache is being reused
// correctly, a cached round's output must be byte-identical to the same
// prompt run from a forced-fresh (cache-evicted) rebuild.
//
// `MLXLanguageModel.removePromptCache(modelID:)` is an internal (not
// public) entry point -- reachable here via `@testable import
// MLXFoundationModels` -- that drops a model's remembered `PromptCache`
// slots without evicting the loaded model weights (unlike `evict()`/
// `evictAll()`, which would also force a slow reload). Calling it
// immediately before a round forces `PromptCache.resolve` to find nothing
// and rebuild from scratch: a real, unassisted full prefill, matching
// exactly what pre-cache behavior looked like.
//
// Loads a real model, so this lives in the IntegrationTesting xcodeproj
// alongside `PromptCacheReuseTests`, sharing that file's
// `respondCollectingTextAndUsage` helper (now hoisted to
// `Support/FMTestHelpers.swift`) -- run in isolation from other suites
// touching the same model id for a reliable result.
//
// UPDATE (kanban `1h5jhne`): the SDK-incompatibility this file's original
// caveat described (`Response.Action`/etc. as an opaque struct) no longer
// reproduces in this environment -- the whole `IntegrationTesting`
// xcodeproj target builds and this file's suite, including the
// fork/evictAll addition below, has been run for real against real MLX
// models (`xcodebuild test -project IntegrationTesting/IntegrationTesting
// .xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'
// -only-testing:IntegrationTestingTests/PromptCacheEquivalenceTests`) and
// passes. One real-execution finding folded in above:
// `editedEarlierTurnForcesTrimAndMatchesFreshRebuild` originally used
// `TestFixtures.gemmaModelID`, but Gemma 3's sliding-window `RotatingKVCache`
// layers (see `Gemma3Text.newCache`) make it unconditionally non-chunkable
// under `PromptCache.sliceChunks`'s `KVCacheSimple`-only requirement, so
// `cachedTokenCount` was always `0` for it -- switched to `defaultModelID`,
// which this test's assertions require.

#if FoundationModelsIntegration

import Testing
import Foundation
import FoundationModels
import MLXLMCommon
@testable import MLXFoundationModels

/// Integration tests proving cached-run output stays byte-identical to a fresh, cache-evicted rebuild, across a plain multi-turn conversation and an edited-earlier-turn trim.
@Suite(
    .serialized, .timeLimit(.minutes(10)),
    .enabled(
        if: ProcessInfo.processInfo.isOperatingSystemAtLeast(
            OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0)),
        "Requires the iOS/macOS/visionOS 27 FoundationModels APIs"))
struct PromptCacheEquivalenceTests {

    @Test(
        """
        A multi-turn cached conversation produces IDENTICAL output, turn by turn, to \
        the same growing transcript replayed with the cache forcibly evicted before \
        every round
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func cachedRunMatchesFreshPrefillEveryRound() async throws {
        let model = makeTestModel(TestFixtures.gemmaModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)
        let userTurns = [
            "Say 'hi' in exactly one word.",
            "Now say 'bye' in exactly one word.",
            "Now say 'thanks' in exactly one word.",
            "Now say 'ok' in exactly one word.",
        ]

        // Clean starting state: no leftover slot from an earlier test/run
        // touching the same model id.
        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Phase A: run the whole conversation once, letting `PromptCache`
        // reuse freely across turns -- the normal production code path.
        // Record every turn's growing transcript and its output text.
        var entries: [Transcript.Entry] = []
        var transcriptsByTurn: [Transcript] = []
        var cachedOutputsByTurn: [String] = []
        for userText in userTurns {
            entries.append(
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: userText))],
                        responseFormat: nil)))
            let transcript = Transcript(entries: entries)
            transcriptsByTurn.append(transcript)

            let request = makeExecutorRequest(transcript: transcript, generationOptions: greedyOptions)
            let output = try await respondCollectingTextAndUsage(
                executor, request: request, model: model
            ).text
            cachedOutputsByTurn.append(output)

            entries.append(
                .response(
                    Transcript.Response(
                        assetIDs: [], segments: [.text(Transcript.TextSegment(content: output))])))
        }

        // Phase B: for each turn's IDENTICAL growing transcript, force a
        // fresh (cache-evicted) rebuild and compare against Phase A's
        // recorded output for that same turn.
        for (turnIndex, transcript) in transcriptsByTurn.enumerated() {
            await MLXLanguageModel.removePromptCache(modelID: model.modelID)
            let request = makeExecutorRequest(transcript: transcript, generationOptions: greedyOptions)
            let freshOutput = try await respondCollectingTextAndUsage(
                executor, request: request, model: model
            ).text

            #expect(
                freshOutput == cachedOutputsByTurn[turnIndex],
                """
                turn \(turnIndex): cached-run output (\(cachedOutputsByTurn[turnIndex])) diverged \
                from a fresh, cache-evicted rebuild of the identical prompt (\(freshOutput)) -- \
                the KV cache reuse produced different generated tokens
                """
            )
        }

        await releaseAllGPUMemory()
    }

    @Test(
        """
        Editing an earlier turn (forcing the chunk-store walk to stop at the last shared \
        chunk boundary and feed the diverging suffix) still produces output identical to a \
        fresh, cache-evicted rebuild of the same edited transcript
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func editedEarlierTurnForcesTrimAndMatchesFreshRebuild() async throws {
        // Deliberately NOT `TestFixtures.gemmaModelID` (unlike this file's
        // other test): confirmed by real execution (kanban `1h5jhne`) that
        // Gemma 3's `newCache()` mixes `RotatingKVCache` sliding-window
        // layers in with its global-attention layers (see
        // `Gemma3Text.newCache`) -- `PromptCache.sliceChunks`'s
        // `verifiedSimpleLayers` requires EVERY layer to be a plain
        // `KVCacheSimple`, so a Gemma 3 cache is never chunkable at all and
        // `cachedTokenCount` is unconditionally `0` for it, making this
        // test's own partial-chunk-match assertion below unsatisfiable.
        // `defaultModelID` has no sliding-window layers, matching the model
        // choice `PromptCacheReuseTests`/`PromptCacheGuidedRoundTripTests`
        // already use for the same reason.
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Round 1: establishes a cached prefix (prompt tokens + the model's
        // real generated reply tokens) for this model id.
        let firstUserText = "Say 'hi' in exactly one word."
        let firstTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil))
        ])
        let firstRequest = makeExecutorRequest(transcript: firstTranscript, generationOptions: greedyOptions)
        let first = try await respondCollectingTextAndUsage(
            executor, request: firstRequest, model: model)
        let firstOutput = first.text
        // Round 1 is a full rebuild (first turn, nothing cached yet), so
        // promptTokenCount is exactly round 1's fed prompt length; adding
        // its real generated-token count gives the FULL length round 1's
        // stored slot actually holds -- the yardstick used below to prove
        // round 2 partially (not fully) reused it.
        let firstStoredSlotLength = first.promptTokenCount + first.outputTokenCount

        // Round 2's transcript EDITS round 1's own reply to something
        // different (guaranteed to differ from whatever the model actually
        // said) before appending a new user turn. The rendered prompt
        // shares a genuine, nonzero common prefix with round 1's stored
        // chunks (the preamble + first user turn) but diverges at the
        // edited assistant reply -- exactly the case where
        // `PromptCache.lookupLongestPrefix`'s chunk walk matches some
        // leading chunks (`0 < matched < stored chunk count`) then stops at
        // the first diverging chunk, never exercised by any other test with
        // a REAL model run (only `PromptCacheChunkStoreTests`'/
        // `PromptCacheChunkCutoverTests`' pure-function tests and
        // `PromptCacheReuseTests`' clean-prefix-extension case, which
        // matches every stored chunk instead).
        let editedFirstReply = firstOutput == "Hello!" ? "Hi there!" : "Hello!"
        let secondUserText = "Now say 'bye' in exactly one word."
        let editedTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: firstUserText))],
                    responseFormat: nil)),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: editedFirstReply))])),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: secondUserText))],
                    responseFormat: nil)),
        ])

        let cachedRequest = makeExecutorRequest(
            transcript: editedTranscript, generationOptions: greedyOptions)
        let cached = try await respondCollectingTextAndUsage(
            executor, request: cachedRequest, model: model)
        let cachedOutput = cached.text

        // Confirm this round genuinely matched a partial chunk prefix, not
        // zero chunks (full rebuild) or every stored chunk (clean
        // extension) -- otherwise the output-equality check below could
        // pass for the wrong reason (e.g. a tokenizer quirk collapsing the
        // edit into a full prefix match) without ever exercising the
        // diverging-suffix path this test claims to cover.
        // `cachedTokenCount` reports how many of the edited transcript's
        // tokens were served from matched chunks: `0` would mean no chunks
        // matched (full rebuild), and `== firstStoredSlotLength` would mean
        // every one of round 1's stored chunks still matched (the edit
        // didn't actually change the tokenized prefix). Strictly between
        // the two proves a genuine partial chunk match followed by a
        // diverging suffix.
        #expect(
            cached.cachedTokenCount > 0 && cached.cachedTokenCount < firstStoredSlotLength,
            """
            expected a genuine partial chunk match with \
            0 < cachedTokenCount < firstStoredSlotLength(\(firstStoredSlotLength)), got \
            cachedTokenCount = \(cached.cachedTokenCount) -- this round did not take the \
            partial-chunk-match path the test claims to exercise (0 means no chunks matched; \
            == \(firstStoredSlotLength) would mean every stored chunk still matched)
            """
        )

        // Baseline: force a fresh rebuild of the SAME edited transcript.
        await MLXLanguageModel.removePromptCache(modelID: model.modelID)
        let freshRequest = makeExecutorRequest(
            transcript: editedTranscript, generationOptions: greedyOptions)
        let freshOutput = try await respondCollectingTextAndUsage(
            executor, request: freshRequest, model: model
        ).text

        #expect(
            cachedOutput == freshOutput,
            """
            edited-earlier-turn cached output (\(cachedOutput)) diverged from a fresh, \
            cache-evicted rebuild of the identical (edited) transcript (\(freshOutput)) -- \
            the partial-chunk-match / diverging-suffix path produced different generated tokens
            """
        )

        await releaseAllGPUMemory()
    }

    @Test(
        """
        A forked-transcript response -- the shared parent prefix extended with a fork-specific \
        tail, exactly the shape a transcript-seeded LanguageModelSession fork produces -- is \
        byte-identical to the SAME fork transcript's response after MLXLanguageModel.evictAll() \
        forces a genuinely fresh model reload and cache rebuild
        """
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func forkedTranscriptResponseMatchesFreshRebuildAfterEvictAll() async throws {
        // `defaultModelID`, not `TestFixtures.gemmaModelID` -- see
        // `editedEarlierTurnForcesTrimAndMatchesFreshRebuild`'s doc comment:
        // Gemma 3's sliding-window `RotatingKVCache` layers make it
        // unconditionally non-chunkable, so `cachedTokenCount` would never
        // be positive for it.
        let model = makeTestModel(TestFixtures.defaultModelID)
        let executor = try makeMLXExecutor(for: model)
        let greedyOptions = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 8)

        await MLXLanguageModel.removePromptCache(modelID: model.modelID)

        // Parent turn: establishes the shared prefix a fork extends.
        let parentUserText = "Say 'hi' in exactly one word."
        let parentTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: parentUserText))],
                    responseFormat: nil))
        ])
        let parentRequest = makeExecutorRequest(
            transcript: parentTranscript, generationOptions: greedyOptions)
        let parent = try await respondCollectingTextAndUsage(
            executor, request: parentRequest, model: model)
        #expect(!parent.text.isEmpty, "Parent round should produce some response text")

        // Fork transcript: the parent's own entries verbatim (its real reply
        // included), plus a NEW fork-specific user turn -- exactly what a
        // forked, transcript-seeded `LanguageModelSession`'s first
        // `respond()` call would receive (mirrors `PromptCachePrewarmTests`'
        // `forkTranscript(question:)` and this task's own
        // `PromptCacheForkReuseTests`).
        let forkQuestion = "Now say 'bye' in exactly one word."
        let forkTranscript = Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: parentUserText))],
                    responseFormat: nil)),
            .response(
                Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: parent.text))])
            ),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: forkQuestion))],
                    responseFormat: nil)),
        ])
        let forkRequest = makeExecutorRequest(
            transcript: forkTranscript, generationOptions: greedyOptions)
        let forked = try await respondCollectingTextAndUsage(
            executor, request: forkRequest, model: model)
        #expect(!forked.text.isEmpty, "Fork round should produce some response text")
        #expect(
            forked.cachedTokenCount > 0,
            """
            the fork's own respond() round should reuse the parent's cached prefix -- \
            cachedTokenCount == 0 means nothing was reused, so this test would not actually be \
            comparing a CACHED fork response against the fresh rebuild below
            """
        )

        // Baseline: `MLXLanguageModel.evictAll()` drops BOTH the loaded model
        // weights AND every model's `PromptCache` chunk store -- a strictly
        // stronger reset than `removePromptCache(modelID:)` alone (used
        // elsewhere in this file), which drops only the cache and leaves the
        // weights resident. Forces a genuinely fresh rebuild from scratch:
        // reloading the model container from disk, re-tokenizing, and
        // re-prefilling the identical fork transcript with no prior state at
        // all.
        await MLXLanguageModel.evictAll()
        let freshRequest = makeExecutorRequest(
            transcript: forkTranscript, generationOptions: greedyOptions)
        let fresh = try await respondCollectingTextAndUsage(
            executor, request: freshRequest, model: model)

        #expect(
            forked.text == fresh.text,
            """
            forked-transcript cached output (\(forked.text)) diverged from the same fork \
            transcript's output after MLXLanguageModel.evictAll() forced a fresh model reload \
            and cache rebuild (\(fresh.text)) -- the forked prefix reuse produced different \
            generated tokens than a genuinely fresh run
            """
        )

        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
