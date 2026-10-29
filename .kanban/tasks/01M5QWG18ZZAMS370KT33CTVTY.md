---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5zk9z7vvrj468dcjhgeezgz
  text: |-
    Research (implement pickup):
    - Drafter RoPE comes only from the explicit `positionOffset` (text) / `qwen35MTPPositionIds(offset:)` (VLM); the drafter cache mask is built from cache.offset, so a cold start at P needs only a cache whose entries begin at P. Today every invariant is `cache.offset == state.nextPosition` (finalizeShiftedMTPDrafterState, resumeDrafterState in both Qwen35MTP files), which assumes entries start at 0. Plan: add `MTPDrafterState.cacheStartPosition` (absolute position of entry 0, default 0) and check `cache.offset == nextPosition - cacheStartPosition`.
    - Iterator `prepare` (MTPSpeculativeTokenIterator) has three target-only fallbacks for a warm cache: continuation not aligned, drafter refused resume, and no continuation on a warm cache (`requiresPromptPrefill && cachePositionBeforePrefill > 0`). All three become a cold start at `cachePositionBeforePrefill` over the prefilled suffix (pairs suffix[1...]+bonus with the suffix hidden rows; the entry for suffix[0] needs h_{P-1}, which no cache keeps, so it is skipped).
    - ChatSession never hands a stateful drafter a warm cache without an aligned continuation: `mtpSpeculativeStateIsReady` / `mtpStateIsRecoverable` route those turns to `defaultGeneration()` with `.unavailableContinuation` (tests samplingChangeDropsMTPContinuation…, rawPromptCacheFallsBack…). The card's acceptance (warm cache, no continuation, runs MTP, next turn continues) needs those two predicates to accept resumable drafters. Overlap: ^ QWGWNFD (01M5QWGWNFDVP77YAW9ACHVXY1) lists "`unavailableContinuation`: use the cold start" for ChatSession; this card's acceptance requires it, so the predicate change lands here and that bullet should be re-checked when ^QWGWNFD is picked up. The `.appendOnly` capability (trim → main only) is left alone: Qwen's hybrid cache cannot trim anyway (rebuild).
    - Fakes conforming to ResumableMTPDrafterModel (QwenStyleMTPDrafter, EarlyTerminationMTPDrafter) use `cache: []`; new protocol requirements must work for them.
  timestamp: 2026-10-27T21:12:21.499412+00:00
- actor: claude-code
  id: 01m5zmqqe3emx2k8pst9r51ksd
  text: |-
    Implementation landed (TDD: 7 new tests watched fail for the expected reasons, then pass).

    API (Libraries/MLXLMCommon/MTPDrafterModel.swift):
    - `MTPDrafterState.cacheStartPosition` (absolute position of cache entry 0; init default 0, source compatible), `entryCount`, `isAligned` (every layer holds `nextPosition - cacheStartPosition` entries; a state with no cache layers tracks positions only), and `removeLastEntries(_:) -> Bool` — the general trim: tentative entries first, then committed; drops the seed; refuses (state unchanged) when misaligned or short.
    - `ResumableMTPDrafterModel` gains `appendDrafterState(target:tokens:targetHidden:positionDeltas:state:sampler:)` (the one model-specific requirement: run the predictor over (token, hidden-row) pairs at `nextPosition`, seed the next proposal) and `coldStartDrafterState(...position:...)`.
    - Protocol-extension defaults describe a shifted drafter and are built on append + removeLastEntries: `resumeDrafterState`, `coldStartDrafterState` (removes all entries, sets cacheStartPosition = nextPosition = P, appends suffix[1...]+bonus with the suffix hidden rows), `prepareDrafterState` (= cold start at 0), `commitDrafterState`, `finalizeDrafterState` (the old `finalizeShiftedMTPDrafterState`, now also drops tentative entries: committed position = nextPosition - proposalAppended). Both Qwen35MTP.swift files (LLM + VLM) now implement only `appendDrafterState`; their copies of prepare/resume/commit/finalize are gone.
    - Iterator `prepare` → new `primeDrafterState(promptTokens:cachePosition:continuation:)`: position 0 prepares; warm cache resumes an aligned continuation, else (no continuation, misaligned, or refused) a resumable drafter cold-starts at the cached position; only a non-resumable stateful drafter still goes target-only.
    - ChatSession: `mtpSpeculativeStateIsReady = !stateful || resumable` and `mtpStateIsRecoverable = ready || processedTokenCount == 0`, so warm turns without a continuation (sampling turn dropped it, raw prompt cache) reach the iterator and cold-start. Trim decisions are unchanged (`.appendOnly` → trimToCommonPrefixMainOnly → `.unavailableContinuation`): Qwen's hybrid cache cannot rewind anyway; left for ^QWGWNFD (01M5QWGWNFDVP77YAW9ACHVXY1), whose "`unavailableContinuation`: use the cold start" bullet is now covered for the append paths.

    Tests: Qwen35MTPTests `Qwen35MTPDrafterStateTests` (trim+append == cold prefill on the seeded fixture, finalize with a tentative entry, cold start at P then finalize+resume, VLM drafter cold start with position deltas, iterator on a warm fixture cache without continuation == plain greedy tokens with proposals/accepts > 0 and no passthrough). ChatSessionQwenTextMTPTests: refused continuation cold-starts (probe records [0, P] — the first-turn prefill is itself a cold start at 0), raw Qwen fixture cache cold-starts on turn 1 and resumes on turn 2 with text == plain session; `samplingChangeDrops…` and `rawPromptCache…` updated: those turns now speculate instead of `.unavailableContinuation`.
    VLM: no VLM fixture exists; the VLM drafter shares the protocol defaults and has a unit test for cold-start positions/deltas; the existing VLM ChatSession resume test still passes. No VLM end-to-end cold-start ChatSession test.

    Notes: the fakes QwenStyleMTPDrafter/EarlyTerminationMTPDrafter gained `appendDrafterState`; QwenStyle also records cold starts and can refuse resumption via `MTPStagingProbe`. Pre-existing warning unrelated to this change: Tests/MLXLMTests/ChatSessionTests.swift `let first = await iterator.next()` inferred `Void?`.

    Benchmark qwen38FourBitMTPGeneration (Release): run 1 (right after build) MTP 81.9 / 80.0 / 79.2 tok/s; run 2 MTP 90.5 / 81.4 / 79.2 tok/s (plain 59.3/59.4/59.4), acceptance 93.9/73.6/69.0%, matches plain: true. Baseline 89.4 / 80.5 / 78.5 — no regression.
  timestamp: 2026-10-27T21:37:20.835406+00:00
- actor: claude-code
  id: 01m5zmqwxnmb4wz2qtensjt967
  text: |-
    ### implement — changed
    - evidence: 9 files — Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift, Tests/MLXLMTestSupport/MTPFakes/ChatSessionMTPFakes.swift, Tests/MLXLMTestSupport/MTPFakes/EarlyTerminationMTPFakes.swift. `swift build --build-tests` ok; `xcrun xctest MLXLMTests.xctest`: 895 XCTest 0 failures, 1849 Swift Testing passed (1842 + 7 new; env-gated benchmarkVerification skip as before); MLXFoundationModelsTests 376 passed. Release qwen38FourBitMTPGeneration MTP 90.5 / 81.4 / 79.2 tok/s (second run; first run 81.9 / 80.0 / 79.2), matches plain: true.
    - next: /review (task left in doing; not committed)
  timestamp: 2026-10-27T21:37:26.453137+00:00
- actor: claude-code
  id: 01m5zn8bnb79ar3pd0cxvewy1y
  text: |-
    ### test — green
    - evidence: swift build --build-tests (forced full recompile of MLXLMCommon and test sources; only warning is the known "missing creator ... mlx-swift_Cmlx.bundle"); xcrun xctest MLXLMTests — 895 XCTest, 0 failures, and 1849 Swift Testing passed (benchmarkVerification() env-gated skip, known); MLXFoundationModelsTests — 376 passed. The warning at Tests/MLXLMTests/ChatSessionTests.swift (`constant 'first' inferred to have type 'Void?'`) was pre-existing and not caused by this change: it is the `AsyncStream<Void>` iterator in testTokenObservationStopsAfterCancellationAndReleasesTheCache, and no changed signature is involved. Fixed with an explicit `let first: Void?` annotation; the build now has zero warnings besides the known bundle one.
    - next: review; nothing committed.
  timestamp: 2026-10-27T21:46:25.835151+00:00
- actor: claude-code
  id: 01m5znj1cgg5afe793yepkbc6d
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (3d70f39e): 1 finding (confirmed 2, refuted 1, attempted 21, failed 0) — Tests/MLXLMTests/Qwen35MTPTests.swift:716 code-hygiene/magic-numbers-swift. Requested checks passed: (1) the drafter only proposes and the target verifies every token; a failed resume or cold start falls to passthrough. (2) removeLastEntries leaves cacheStartPosition unchanged, lowers nextPosition and proposalAppended, and re-checks per-layer offset == entryCount. RoPE uses absolute state.nextPosition. finalizeDrafterState removes tentative entries before the boundary trim. (3) For a resumable drafter, ChatSession readiness passes either the continuation state or a fresh state. A continuation is resumed only when targetProcessedTokenCount == cachePosition. Otherwise coldStartDrafterState removes every existing entry before it re-primes, so stale entries never reach a turn.
    - next: replace the magic numbers on the drafterCacheTolerance line (1e-5, 2e-3) and the same cause across Qwen35MTPTests.swift with named constants, then re-review.
  timestamp: 2026-10-27T21:51:42.992610+00:00
- actor: claude-code
  id: 01m5znjk6e7ygmnt5t4nmvdyc8
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 9 files (drafter append/trim API, cold start in iterator and ChatSession)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1849 Swift Testing; FoundationModels 376; fixed pre-existing Void? warning
    - commit: 3d70f39e
    - review: findings — Tests/MLXLMTests/Qwen35MTPTests.swift:716
  timestamp: 2026-10-27T21:52:01.230980+00:00
- actor: claude-code
  id: 01m5zny87jtbc3t3y89bqa50pm
  text: |-
    Finding fix: `Tests/MLXLMTests/Qwen35MTPTests.swift:716` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
    - Cause: the tolerance literals 1e-5 / 2e-3 were written inline. The same pair also sat in `gdnStateTolerance`, which the review did not cite, and plain `1e-5` sat in the two GDN checkpoint-output `allClose` calls. All of them now use named, documented file-level constants: `ieeeFloat32Tolerance` (1e-5), `tf32ReshapedMatmulTolerance` (2e-3), and `gdnCheckpointOutputTolerance` (= ieeeFloat32Tolerance; both runs use the same matmul shapes, so it keeps full float32 precision under TF32 as well). `gdnStateTolerance` and `drafterCacheTolerance` are now built from the first two.
    - Fixture literals in Qwen35MTPDrafterStateTests (the suite this task added) are now named stored properties: `bonusToken` (17; `bonus` is computed from it), `nextTurnPromptToken` (8), `nextTurnBonusToken` (11), `vlmSuffixTokens` ([5, 6]), `vlmBonusToken` (7), `vlmPositionDelta` (2).
    - Checked by running swiftlint `no_magic_numbers` (allowed [0, 1, -1, 100], the validator's config) on the file: before the fix it reported 225:68, 225:83, 716:72, 716:87; after the fix it reports 0.
    - Left unchanged on purpose: inline token and shape literals in suites added by earlier tasks on this branch (for example Qwen35MTPMetalTests `MLXArray([Int32(1), 2, 3])`). swiftlint does not flag them, and renaming them would refactor existing tests outside this card.
  timestamp: 2026-10-27T21:58:23.218339+00:00
- actor: claude-code
  id: 01m5znyayrdb3s2kamwnzb6v4n
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/Qwen35MTPTests.swift. swiftlint no_magic_numbers: 0 findings (4 before). `swift build --build-tests`: only warning is the known "missing creator ... mlx-swift_Cmlx.bundle". `xcrun xctest MLXLMTests.xctest`: 895 XCTest, 0 failures, and 1849 Swift Testing passed (benchmarkVerification env-gated skip, known). Finding flipped to [x].
    - next: /review (task left in doing; not committed)
  timestamp: 2026-10-27T21:58:26.008788+00:00
- actor: claude-code
  id: 01m5zp076evpxggega39rep1c3
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (2bfa5601) — 0 findings (0 confirmed, 0 refuted; 7 validator passes attempted, 0 failed). The prior item Tests/MLXLMTests/Qwen35MTPTests.swift:716 code-hygiene/magic-numbers-swift is checked off, and nothing on the changed lines was flagged again. Inline numbers in older suites (e.g. Qwen35MTPMetalTests) are on lines this commit did not change, so the diff op does not review them, and the review skill drops findings that ask for refactors of tests that already existed.
    - next: none — moved to done.
  timestamp: 2026-10-27T21:59:27.694225+00:00
- actor: claude-code
  id: 01m5zp0hefqr7bhb45ddts6wa7
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 file (named tolerances and token values in Qwen35MTPTests.swift)
    - test: green — implementer's run after the fix: MLXLMTests 895 XCTest 0 failures + 1849 Swift Testing; no separate tester run (test-only change, identical commands)
    - commit: 2bfa5601
    - review: clean — 0 findings, prior item verified; task done
  timestamp: 2026-10-27T21:59:38.191304+00:00
depends_on:
- 01M5SK8FDMF5XH4FDH12H0FPX1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffe780
title: 'MTP: drafter cold start at a cached position, and a general trim/append API (resume is done by PR 581)'
---
## What
Upstream PR 581 (commit 843854ec) added `ResumableMTPDrafterModel` with `resumeDrafterState(...)` and `finalizeDrafterState(...)` in both Qwen35MTP.swift files (shared code `finalizeShiftedMTPDrafterState`), and `MTPDrafterContinuation { state, targetBoundaryHidden, targetProcessedTokenCount }` (Libraries/MLXLMCommon/MTPDrafterModel.swift). Thus a drafter continues from a cached prefix when a continuation is present.

What remains:
- [x] Cold start: today a warm main cache with no aligned continuation makes the iterator run target-only for the turn (and for later turns on that cache). Add a cold start: an empty drafter state at position P (`positionOffset` sets RoPE before `cache.ropeOffset`, Libraries/MLXLLM/Models/Qwen35.swift ~597), fed from the first new token with the boundary hidden row of the current prefill. Correctness does not depend on the drafter. Use it in the iterator in place of the target-only fallback for `unavailableContinuation`.
- [x] A general drafter-state API to trim the last n tentative entries and append entries with given hidden rows. `finalizeShiftedMTPDrafterState` requires `proposalAppended == 0`; make it work when tentative entries remain (needed for chained drafts).

## Acceptance Criteria
- [x] A turn on a warm cache with no continuation runs MTP (no passthrough) and gives the same greedy tokens as plain decode; the next turn continues the drafter.
- [x] Trim and append give the same drafter cache contents as a cold run over the same tokens (within float tolerance).

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPTests.swift and ChatSessionQwenTextMTPTests.swift: cold start, then continuation; trim/append.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38

## Review Findings (2026-10-27 16:46)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWG18ZZAMS370KT33CTVTY.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWG18ZZAMS370KT33CTVTY.md` — no validator matches this file
> - `.kanban/tasks/01M5SK8FDMF5XH4FDH12H0FPX1.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SK8FDMF5XH4FDH12H0FPX1.md` — no validator matches this file

- [x] `Tests/MLXLMTests/Qwen35MTPTests.swift:716` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
