---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m627e1evx95eyedgsnbxp2j0
  text: 'Research: production Qwen35MTPDraftModel has maximumBlockSize 2, so the bug is reachable only with a drafter of a larger block (scripted drafters in tests; ^hcfzyem later). At blockSize 3 a partial round accepts exactly 1 draft, so no stop leaves lookahead; the bug needs blockSize >= 4. In ChatSession the bug shows as Conversation.record invalidating the ledger (processed - cached > generated) -> next turn full rebuild. Plan: MambaCache keeps the slots of the accepted tokens after a partial-accept rollback (instead of releasing them), so finalizeGeneration''s rewindSpeculative(lookahead) finds them; next round / passthrough / finalize release them as for an all-accepted round. Tests: new Tests/MLXLMTests/Qwen35PartialRoundFinalizeTests.swift (main cache vs one prefill, drafter continuation vs promptBoundaryContinuation of a prefill, second call == cold == plain, ChatSession next turn reuses cache) + Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift (ScriptedDraftPlan shared by ScriptedRejectionDrafter (moved out of Qwen35VerifyRewindTests) and a new ScriptedResumableDrafter delegating state to Qwen35MTPDraftModel). RED confirmed: 27 issues at blockSize 4..6 (held includes lookahead, continuation nil, chat turn 2 promptTokenCount 16 = full rebuild); blockSize 3 passes as predicted.'
  timestamp: 2026-10-28T21:42:35.227218+00:00
- actor: claude-code
  id: 01m6286mvk07qdbrsweexvtxzb
  text: 'Implementation landed. Decision: keep the slots, no replay. MambaCache.SpeculativeSlots gains keptTokenCount; selectSpeculativeSlot(rollingBack:) selects the slot and keeps the slots of the accepted tokens (keptTokenCount = accepted), so finalizeGeneration''s existing rewindSpeculative(lookahead) rewinds the accepted drafts exactly. Slots are released by the next round''s discardSpeculativePromptCacheCheckpoints, passthrough, finalize, or at once when a rollback keeps only the bonus token (unrejectableHeadTokenCount = 1) — so blockSize 2 is exactly as before (a rejected round releases at once; an accepted round keeps, as before). No change to finalizeGeneration logic was needed; only comments. Memory cost: none at peak — a partially accepted round holds the same one-verify snapshot buffer an all-accepted round already holds, until the next round starts; never in checkpoints, copies, saved caches or byte counts. tok/s cost: no extra compute (no replay). Measured Release qwen38FourBitMTPGeneration: plain 58.9/58.8/58.8, MTP 91.0/80.8/78.8 tok/s, output matches plain on all three (baseline ~91/81/80, all within 3%). qwen38FourBitMTPSystemPromptCache: passed, no known issues, edit turn prefilled 123 tok with MTP on. Test contract changes: KVCacheContinuationTests / Qwen35GDNDecodeKernelTests now expect a rollback to keep the slots of the kept tokens (0 when only the bonus is kept); new aRollbackKeepsTheSlotsOfTheKeptTokensForAFurtherRollback and aRewoundHybridCacheRewindsItsKeptTokensAgain. ChatSession test trick: ScalarTokenizer renders no end-of-turn token, so the next user message starts with the stop token (as a real template renders <|im_end|> after the reply); turn 2 then reuses prompt+reply+eos (cachedPromptTokenCount = prompt + stopIndex + 1, promptTokenCount = 3). Note: MTPGreedyVerificationTests.benchmarkVerification is env-gated (MTP_GREEDY_BENCHMARK) and shows as skipped in MLXLMTests — preexisting, not from this card.'
  timestamp: 2026-10-28T21:56:01.523113+00:00
- actor: claude-code
  id: 01m6286v3wy8687hpakwdng908
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/ArraysCache.swift, Libraries/MLXLMCommon/KVCache.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Tests/MLXLMTests/Qwen35PartialRoundFinalizeTests.swift (new, 16 KB), Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift (new, 8 KB), Tests/MLXLMTests/Qwen35VerifyRewindTests.swift, Tests/MLXLMTests/KVCacheContinuationTests.swift, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift. `swift build --build-tests`: 0 compiler warnings (only the preexisting swiftbuild "missing creator" build-system note). xctest: MLXLMTests 1945 Swift Testing + 920 XCTest passed, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures. Release qwen38FourBitMTPGeneration passed (MTP 91.0/80.8/78.8 tok/s, matches plain); qwen38FourBitMTPSystemPromptCache passed, no known issues, edit turn 123 tok with MTP.
    - next: /review
  timestamp: 2026-10-28T21:56:07.932208+00:00
- actor: claude-code
  id: 01m628dgzrb1p97kgf204hwwyy
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (9fdbae15) — 3 findings (3 confirmed, 2 refuted): Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:319, Tests/MLXLMTests/Qwen35PartialRoundFinalizeTests.swift:209, Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift:187
    - next: implement the unchecked items in `## Review Findings (2026-10-28 16:56)`; task stays in review
  timestamp: 2026-10-28T21:59:46.936079+00:00
- actor: claude-code
  id: 01m628dy2bts59j4qtexjymwn6
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 8 files
    - test: green (implementer run) — 5 bundles; Release MTP 91.0/80.8/78.8 = plain; cache guard pass, edit turn 123 tok
    - commit: 9fdbae15
    - review: findings — Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:319, Qwen35PartialRoundFinalizeTests.swift:209, Qwen35ScriptedMTPSupport.swift:187
  timestamp: 2026-10-28T22:00:00.331210+00:00
- actor: claude-code
  id: 01m628t71yy542eafmz2tc8659
  text: |-
    Review findings (2026-10-28 16:56) addressed, test code only, cause removed across each file:
    1. duplication: SnapshotSlotFixture (KVCacheContinuationTests.swift) changed from `private enum` to internal so the GDN kernel tests can call it. Qwen35GDNDecodeKernelTests.swift now uses `SnapshotSlotFixture.slotCount(keeping:)` in both places that inlined the rule: the finding's `kept > 1 ? kept : 0`, and a second copy in aVerifyThroughTheLayerKeepsOneSlotForEachToken, which hard-coded `speculativeSlotCount == 0` for acceptedTokens = 1.
    2. no-test-cheating: Qwen35PartialRoundFinalizeTests.swift binds `let stops = try partialRoundStops(...)` and adds `#expect(!stops.isEmpty)` before the loop in both unguarded tests (drafter-continuation and second-call). The drafter-continuation state-pair loop now also expects its cache-state pairs to be non-empty, so the loop cannot pass on the boundary-hidden pair alone. Not changed: `Qwen35MTPFixture.expectCache` (Qwen35ScriptedMTPSupport.swift) loops over zip(states) without a non-empty guard. It is outside the file the finding names; a later review may flag it.
    3. reuse: in ScriptedResumableDrafter, the stateful draftBlock(..., state:, ...) overload now delegates to the stateless overload, so the drafting rule lives in one place.
    Note: the dump validators file for .swift is 754 KB / 12k lines, too large to read in one pass. Edits followed the named rules (duplication, reuse, test-integrity).
  timestamp: 2026-10-28T22:06:42.750152+00:00
- actor: claude-code
  id: 01m628tbkwg402tyvwawfkt3wq
  text: |-
    ### implement — changed
    - evidence: 4 files — Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift, Tests/MLXLMTests/KVCacheContinuationTests.swift, Tests/MLXLMTests/Qwen35PartialRoundFinalizeTests.swift, Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift. `swift build --build-tests`: 0 compiler warnings (only the preexisting swiftbuild "missing creator" note). xcrun xctest: MLXLMTests 1945 Swift Testing + 920 XCTest passed, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 376, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures, 0 issues. Findings 3/3 checked.
    - next: /review
  timestamp: 2026-10-28T22:06:47.420673+00:00
- actor: claude-code
  id: 01m628w3frj7afex92krcatx0d
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (ddd3d106) — 0 findings (0 confirmed, 0 refuted; 7 validator runs attempted, 0 failed); 4 files reviewed, 2 .kanban files not matched by any validator. All 3 findings from 2026-10-28 16:56 are checked.
    - next: none — task moved to done
  timestamp: 2026-10-28T22:07:44.632031+00:00
- actor: claude-code
  id: 01m628wg92b0bpxm2s44447xj4
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 test files
    - test: green (implementer run) — 5 bundles (MLXLMTests 920 + 1945)
    - commit: ddd3d106
    - review: clean — task in done
  timestamp: 2026-10-28T22:07:57.730220+00:00
depends_on:
- 01M5SKBV27GED43EB3T8MCSVWE
position_column: done
position_ordinal: fff480
title: 'MTP: finalizeGeneration cannot rewind the committed lookahead of a partially accepted hybrid round'
---
## What
Found while doing ^8mcsvwe. With blockSize > 2 (^hcfzyem), a round can accept some drafts and reject the rest. The iterator then rewinds the main cache by `numDraft - accepted` with `rewindSpeculativePromptCache`, and `MambaCache.selectSpeculativeSlot(rollingBack:)` releases all snapshot slots (decision of ^cst7wan: a round with a rejection releases its slots). The accepted drafts stay in the cache as committed lookahead (`committedPendingTokenCount = accepted`).

If the stream stops before the caller consumes those accepted drafts (stop token, maxTokens, cancel), `finalizeGeneration` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift) calls `mainCacheStorage.trim(lookahead)` (refused: MambaCache is not trimmable) and then `mainCacheStorage.rewindSpeculative(lookahead)`, which returns 0 because no slots remain. `rewound != lookahead`, so `targetBoundaryHidden = nil`, the drafter continuation is dropped, and the main cache keeps tokens that were never emitted. At blockSize 2 this cannot happen (accepted = 1 implies no rejection, and the slots stay).

- [x] Keep enough recurrent state after a partial-accept rewind to rewind the accepted drafts again (for example: keep the slots of tokens 1..accepted after `selectSpeculativeSlot`, and release them at the next round / passthrough / finalize, as an all-accepted round does), or replay — choose and document; measure the memory and tok/s cost.
- [x] `finalizeGeneration` rewinds the committed lookahead of a partially accepted round exactly.

## Acceptance Criteria
- [x] With a scripted drafter at blockSize 3..6 (see Tests/MLXLMTests/Qwen35VerifyRewindTests.swift `ScriptedRejectionDrafter`), a stream that stops inside the accepted drafts of a partially accepted round leaves the main cache at the emitted ledger count, and a continued second call equals a cold call and plain decode.

## Tests
- [x] Tests/MLXLMTests: stop-inside-accepted-drafts test at blockSize 3..6.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache

## Review Findings (2026-10-28 16:56)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 8 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M61T8ZW58TVX5JPXREENN005.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61T8ZW58TVX5JPXREENN005.md` — no validator matches this file

- [x] `Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift:319` `duplication/duplication` — The test re-inlines the slot-count predicate `kept > 1 ? kept : 0` instead of calling the helper that defines it. If the rule for which rollbacks keep slots changes in `SnapshotSlotFixture.slotCount(keeping:)`, this assertion keeps the old rule and the two tests disagree. Call `SnapshotSlotFixture.slotCount(keeping: kept)` at line 319, matching the other tests, and delete the inline ternary.
- [x] `Tests/MLXLMTests/Qwen35PartialRoundFinalizeTests.swift:209` `test-integrity/no-test-cheating` — `stopLeavesTheDrafterContinuationOfTheReturnedTokens` loops over `partialRoundStops(...)` with no non-empty guard. If that list is empty, the test runs zero iterations and passes without checking the drafter continuation. The sibling test `stopLeavesTheMainCacheOfTheReturnedTokens` guards this with `#expect(!stops.isEmpty)`, so this test can pass vacuously. Read the stops into a `let stops = try partialRoundStops(...)` and add `#expect(!stops.isEmpty)` before the loop, as the main-cache test does.
- [x] `Tests/MLXLMTests/Qwen35ScriptedMTPSupport.swift:187` `reuse/reuse` — The stateful `draftBlock(..., state:, ...)` overload repeats the body of the stateless `draftBlock` overload directly above it (`plan.drafts(queryOffset:blockSize:batchSize:)`). The same behavior is written twice in one class, so a change to the drafting rule has to be made in two places. Have the stateful overload delegate to the stateless one (or both to a single private helper), so the drafting rule lives in one place.
