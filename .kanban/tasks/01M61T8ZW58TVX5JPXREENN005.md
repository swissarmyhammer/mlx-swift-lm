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
depends_on:
- 01M5SKBV27GED43EB3T8MCSVWE
position_column: doing
position_ordinal: '8180'
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