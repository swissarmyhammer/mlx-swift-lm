---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m3d5tshypxdrr24rf5sv9gpj
  text: |-
    ### Research — root cause

    The failure is a library bug in `SpeculativeTokenIterator.finalizeGeneration()` (`Libraries/MLXLMCommon/Evaluate.swift`). It is not only a test problem.

    Facts:
    - `Gemma3TextModel.prepare` returns `.tokens` with the last prompt token. Thus after prefill, main and draft both hold P-1 tokens, and `y == draftY == last prompt token`.
    - In a round, the draft feeds `y` and the first `numDraft - 1` drafts. It does not feed the last draft. The main verifies `y` plus all `numDraft` drafts.
    - When all drafts are accepted (`accepted == numDraft`), `mainCommittedPendingTokenCount = accepted` but `draftCommittedPendingTokenCount = accepted - 1`. The next round feeds the missing draft through `draftY` (two tokens). This is correct while the generation continues.
    - When the generation stops after the consumer took `accepted` or more tokens of that round, `finalizeGeneration()` keeps `min(pendingIndex, accepted)` tokens in the main cache and `min(pendingIndex, accepted - 1)` tokens in the draft cache. The draft cache stays one token behind the main cache.
    - `ChatSession` then sees `draftKVCache.processedTokenCount != cachedTokens.count`, sets `draftCacheIsAligned = false`, and every reuse rule refuses. The next turn rebuilds both caches. The test name says the storage is reused across turns, so the design wants aligned caches.

    Which random weights give main 11, draft 10: any pair where the last round of the first turn (maxTokens 3, numDraft 2) accepts all its drafts and the generation stops in or after them. Example: round 1 rejects (1 token), round 2 rejects (1 token), round 3 has numDraft 1, the draft is accepted, the consumer takes the draft and stops at maxTokens. Main: P-1 + 1 + 1 + 2 = P+3, draft: P+2. Another: round 1 accepts both drafts and the consumer takes all 3 tokens. Two random models seldom agree, thus the failure is intermittent. When the draft and the main model are the same model, all drafts are accepted at temperature 0 and the failure is deterministic.

    The existing unit test `finalizeGeneration trims unreturned speculative lookahead` in `Tests/MLXLMTests/SpeculativeDecodingTests.swift` records the misaligned state as expected (`expectedDraft = 1 + min(consumed, 2)`, `expectedMain = 1 + min(consumed, 3)`). Its values must change with the fix.

    Plan: in `finalizeGeneration()`, keep the same count of tokens in both caches: `min(pendingIndex, draftCommittedPendingTokenCount)`. The main cache gives back at most one verified token, and the next turn prefills it again. Add deterministic tests (unit iterator test and a ChatSession test with a full-accept draft), and seed the weights of the flaky test.

    Other discovery (not this task): in `SpeculativeTokenIterator.prepare`, the `.logits` branch sets `draftY` to the draft's own sample and `y` to the main sample, and never emits the first main sample. Filed as a separate task.
  timestamp: 2026-09-25T20:59:09.246625+00:00
- actor: claude-code
  id: 01m3d60q6bwaz3f08km8q57qa8
  text: |-
    ### implement — changed
    - Fix: `SpeculativeTokenIterator.finalizeGeneration()` (`Libraries/MLXLMCommon/Evaluate.swift`) now keeps `min(pendingIndex, mainCommittedPendingTokenCount, draftCommittedPendingTokenCount)` tokens of the last round in BOTH caches (new private helper `trimLookahead(of:beyond:committed:)`). After a fully accepted round that the loop consumed, the main cache gives back the one draft that the draft cache does not hold.
    - Deterministic reproduction before the fix: new test `ChatSessionTests.testFullyAcceptedSpeculativeRoundLeavesAlignedStorageAcrossTurns` (draft model = main model, seeded weights, maxTokens 3, 2 drafts; telemetry proves 1 round with all drafts accepted) failed with `("Optional(10)") is not equal to ("Optional(9)")` (main 10, draft 9) and `("20") is not equal to ("10")` (the second turn prefilled the full render because `draftCacheIsAligned` was false). After the fix it passes.
    - `testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns` now builds both random models under `withRandomState(MLXRandom.RandomState(seed: 20_260_925))`, and shares the two-turn assertions with the new test through `assertSpeculativeStorageIsReusedAcrossTurns`.
    - `SpeculativeDecodingTests.finalizeGeneration trims unreturned speculative lookahead`: expected values changed from `main = 1 + min(consumed, 3), draft = 1 + min(consumed, 2)` (the misaligned state) to `1 + min(consumed, 2)` for both; argument 3 added.
    - evidence: 3 files — Libraries/MLXLMCommon/Evaluate.swift, Tests/MLXLMTests/ChatSessionTests.swift, Tests/MLXLMTests/SpeculativeDecodingTests.swift
    - next: /test (loop run + five bundles)
  timestamp: 2026-09-25T21:02:23.435365+00:00
- actor: claude-code
  id: 01m3d9zdz1r6x4dnw31hepjera
  text: |-
    ### implement — changed (second approach)
    - What did not work: the first fix trimmed the MAIN cache down to the draft position. The full MLXLMTests run then hung for 45 minutes in `HarmonyChatSessionRoundTripTests.testSpeculativeFinalCallTokenResumesFromTheLiveMainCache` (95% CPU, killed). That test needs the main cache to keep all returned tokens (the Harmony restart rule splices the tool result right after the live `<|call|>` boundary). Do not trim the main cache.
    - New fix: `finalizeGeneration()` keeps the old trims and then calls a new private `feedDraft(_:)`, which runs the draft model on `pendingTokens[draftKept ..< mainKept]` (at most one token: the last accepted draft of a fully accepted round), commits it on `draftCacheStorage`, applies the KV-cache plan, and evaluates the draft cache. Both caches now hold the same returned tokens.
    - `SpeculativeDecodingTests.finalizeGeneration trims unreturned speculative lookahead`: now expects `1 + min(consumed, 3)` for both caches (argument 3 added).
    - `HarmonyChatSessionRoundTripTests.testSpeculativeFinalCallTokenResumesFromTheLiveMainCache`: its first turn is exactly the failing pattern (a fully accepted last round, main 12, draft 11). The old assertion "A lagging draft cache must not be reused" described the bug. With aligned caches the Harmony rule continues with both caches: the main verify pass is at offset 12 with 5 suffix tokens + 3 drafts, and the draft pass is at offset 12 with the 5 suffix tokens. The assertions now state that.
    - evidence: 4 files — Libraries/MLXLMCommon/Evaluate.swift, Tests/MLXLMTests/ChatSessionTests.swift, Tests/MLXLMTests/SpeculativeDecodingTests.swift, Tests/MLXLMTests/HarmonyChatSessionRoundTripTests.swift

    ### test — green
    - evidence: `swift build --build-tests` — only the known warning `missing creator for mutated node`. `xcrun xctest` on all five bundles: MLXLMTests 696 XCTest (695 + 1 new) 0 failures + 1217 Swift Testing passed; MLXGuidedGenerationTests 70 passed; MLXFoundationModelsTests 312 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 5 passed. 0 skipped. `TextToolCallRecoveryBenchmark` passed.
    - Loop: 40 runs of `xcrun xctest -XCTest MLXLMTests.ChatSessionTests/testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns,MLXLMTests.ChatSessionTests/testFullyAcceptedSpeculativeRoundLeavesAlignedStorageAcrossTurns,MLXLMTests.HarmonyChatSessionRoundTripTests/testSpeculativeFinalCallTokenResumesFromTheLiveMainCache` — pass=40 fail=0.
    - next: /commit
  timestamp: 2026-09-25T22:11:35.521016+00:00
position_column: doing
position_ordinal: '80'
title: 'ChatSessionTests.testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns fails sometimes: main 11, draft 10'
---
## What

`Tests/MLXLMTests/ChatSessionTests.swift:1667` failed 1 time in 6 full runs of MLXLMTests on 2026-09-25 (branch `stable`, during task ^y3kxbe5):

```
XCTAssertEqual failed: ("Optional(11)") is not equal to ("Optional(10)")
```

`firstProgress.main` (the processed token count of the main cache) is 11, and `firstProgress.draft` is 10.

Facts:
- `ChatSessionTests.makeModel` (line 395) makes a `Gemma3TextModel` with random weights and no fixed seed. The test calls `model()` two times, thus the main model and the draft model are two different random models, and the weights change at each run.
- Thus the accept and reject pattern of `SpeculativeTokenIterator` changes at each run. One pattern leaves the draft cache one token behind the main cache after the first turn.
- The code path does not call `loadPromptCacheSnapshot(url:into:)`, thus the change of ^y3kxbe5 does not touch it.

Find the cause: either the speculative iterator does not align the main and the draft caches after some accept patterns (a library bug), or the test expects alignment that the design does not give. Make the test deterministic (fixed weight seed) and add a case for the pattern that fails.

## Acceptance Criteria

- [x] The cause is on this task.
- [x] The test passes 20 runs of MLXLMTests in sequence, or the library bug has a fix and a deterministic test.

#flaky-test