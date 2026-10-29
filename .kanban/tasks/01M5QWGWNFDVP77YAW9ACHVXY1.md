---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5zxy5h66kbwhkp785nwcwr4
  text: 'Note from the ^vgma4mq review (2026-10-27), for checkpoint byte accounting: `MTPDrafterContinuation.byteCount` counts like the main caches, and inherits two inaccuracies of `KVCacheSimple.copy()` / `residentByteCount`. (1) Undercount: a copied drafter cache counts only its filled rows (offset); once the live cache grows into a new buffer, the old one is kept alive by the copy at full size (up to 255 extra rows per layer; the Qwen35 drafter has 1 layer). (2) Double count: while a copy and its source both live, each counts the shared boundary hidden row and seed. Decide whether the checkpoint policy needs exact numbers (e.g. compact the copy with a contiguous slice when it is stored) and test it.'
  timestamp: 2026-10-28T00:18:09.062099+00:00
- actor: claude-code
  id: 01m6012qb96tj0290mnq13w6f2
  text: |-
    Research (implement pickup):
    - Checkpoints: `keepsCheckpoints` requires `speculativeDecoding == nil`. On Qwen3.8 an edit or a re-rendered turn is `.rebuild` (GDN maxTrimCount 0, so RewindToCommonPrefixRule never trims), and `.rebuild` already restores a checkpoint and sets `mtpDrafterContinuation = nil`, so with checkpoints on the iterator cold-starts the drafter at the restored position (^33ctvty). That alone gives "cached ~123, drafter fed only the suffix" on the edit turn; restoring the stored continuation is left to ^ahnb0j9.
    - Must stay off for a stateful non-resumable drafter: a checkpoint prefill warms the main cache before the iterator, and that drafter cannot cold-start (passthrough). Gate: `speculativeDecoding == nil || (MTP && mtpSpeculativeStateIsReady)`.
    - `PromptCacheCheckpoint.prefill` calls `model(text, cache:, state:)` per chunk, not `prepare`, so the drafter feed has to set `mtpEmitFlagKey` + the chunk handler per forward and call `feedMTPPrefillChunk` itself (same as LLMModel.prepare), then strip the MTP exchange keys (iterator's `persistentTargetState`) before the state is stored in a checkpoint.
    - Feed result at boundary b: `MTPDrafterPrefillFeed` state has nextPosition b-1 and boundaryHidden = row b-1, i.e. an aligned continuation at b; the iterator then resumes it over promptTokens[b...].
    - End-of-prompt checkpoint: plain takes it in `defaultGeneration`; the MTP path takes none today. After MTP prepare the drafter holds the first-bonus entry (nextPosition == P); a copy with `removeLastEntries(1)` and the iterator's `targetBoundaryHidden` is the aligned continuation at P.
    - MTP path today reports promptTokenCount = input only and does not add the checkpoint feed time; must add both like defaultGeneration does once checkpoints run with MTP.
    - Byte accounting (^vgma4mq note): `KVCacheSimple.copy()` stores `keys[..<offset]` views (keeps the 256-row buffer alive, counts offset rows), and the feed's boundary row is a slice of the whole chunk's hidden rows. `contiguous()` copies when the buffer exceeds the view by >16 KiB (mlx Contiguous::eval_gpu), so a checkpoint stores a compacted copy and its byteCount is then honest within that slack.
    - Incompatible drafter: ChatSession calls `try mtpDrafter.validateCompatibility(with:)` (throws out of respond). Plan: new `SpeculativeDecodingFallbackReason.incompatibleDrafter`, plain decode; throwing API `MTPDrafterContainer.validateCompatibility(with: ModelContainer)`.
    - Memory policy: MTP init sets `memoryPolicy = nil`; plan `init(mtpDrafter:blockSize:memoryPolicy:)` and evaluate drafter weight bytes + carried continuation bytes before any cache work (`.fail` throws SpeculativeDecodingMemoryError, fallback -> `.memoryBudgetExceeded`).
    - unavailableContinuation: for a resumable drafter the only remaining producer is `.trimToCommonPrefixMainOnly` (a rewind on a trimmable target), which the user assigned to ^ahnb0j9 (reuse policy change); a non-resumable stateful drafter cannot cold-start, so target-only stays correct there.
  timestamp: 2026-10-28T01:13:04.105513+00:00
- actor: claude-code
  id: 01m602kycw3s8mv1hva96g1j7c
  text: |-
    Implementation landed (TDD: 9 new tests in ChatSessionQwenTextMTPTests; 8 failed first for the expected reasons; the misaligned-continuation guard test passed against the stub, then held once the implementation landed).

    - Checkpoints with MTP: `keepsCheckpoints` is now `(speculativeDecoding == nil || (usesMTP && mtpSpeculativeStateIsReady))`. A stateful drafter that is not resumable keeps them off, because it cannot start at a warm position. A new `PromptCacheCheckpoint.applies(to:)` guard runs before `applies(to:adding:)`, so caches that can never lose a rewind are not asked about the positions of this pass, and no probe render is made for them. Without that guard, the staging-refusal fakes recorded an extra width.
    - Drafter feed at checkpoints: `MTPDrafterPrefillFeed` is now `package`. It has `starting(drafter:target:parameters:cachePosition:continuation:)` (extends an aligned continuation, else starts a new state) and `continuation` (the aligned continuation after the fed tokens). `PromptCacheCheckpoint.prefill(..., drafterFeed:)` sets the emit flag and the chunk handler on each forward, calls `feedMTPPrefillChunk`, and removes the MTP exchange keys (`LMOutput.State.withoutMTPExchange`, which the iterator's `persistentTargetState` now uses) from the state it carries and stores. `prefillToCheckpoint(..., drafterFeed:)` returns `drafterContinuation`, and the checkpoint keeps it. ChatSession makes one feed per pass, used for both the leading and the last-message checkpoint, and sets `mtpDrafterContinuation` to the feed result after each checkpoint prefill. The iterator then resumes it at the boundary.
    - Prompt-end checkpoint in the MTP path: new `MTPSpeculativeTokenIterator.promptBoundaryContinuation`. It copies the continuation after prepare and drops the first-bonus entry, which gives the aligned continuation at the end of the prompt. It is nil once a token was read, in passthrough, or when the cache is past the prompt. Like defaultGeneration, the MTP path now counts the checkpoint-fed tokens in promptTokenCount and adds their time to promptPrefillTime (before this, MTP turns reported only the iterator input).
    - Byte accounting (^vgma4mq note): a checkpoint stores `MTPDrafterContinuation.compacted()`, and only when the continuation is aligned with its token count. That copy runs `contiguous` on each KVCacheSimple layer array, on the boundary row (a slice of the whole chunk's hidden rows) and on the seed. `PromptCacheCheckpoint.byteCount` adds its byteCount. The count is exact except where MLX keeps a view shared, which needs a buffer no more than 16 KiB larger than the view (Contiguous::eval_gpu). A unit test checks the stored arrays share no buffer with the live ones, and that byteCount equals the bytes of the kept rows. The live buffer is 256 rows; the copy keeps 3 rows.
    - Restore is unchanged (^ahnb0j9): `.rebuild` + checkpoint restore still sets `mtpDrafterContinuation = nil`, so the iterator cold-starts at the restored position. The stored continuations are not handed out yet. When ^ahnb0j9 restores one, it must hand out a `copy()`: the checkpoint value is shared by forks.
    - Incompatible drafter: new `SpeculativeDecodingFallbackReason.incompatibleDrafter`. `ChatSession` no longer calls the throwing validateCompatibility. The throwing API is `MTPDrafterContainer.validateCompatibility(with: ModelContainer) async throws`.
    - Memory policy: `SpeculativeDecodingConfig(mtpDrafter:blockSize:memoryPolicy:)`. A private static `mtpFallbackReason` checks, in order, incompatibility, then sampling, then the policy (main weights against drafter weights + the carried continuation byteCount). It runs before any cache work. The fail action throws `SpeculativeDecodingMemoryError`, and fallback reports `.memoryBudgetExceeded`. `modelWeightBytes` now takes `Module`.
    - unavailableContinuation: for a resumable drafter, the only remaining producer is `.trimToCommonPrefixMainOnly`, a rewind on a trimmable target. That is the reuse-policy change of ^ahnb0j9, as this pass was scoped.
    - Benchmark rows removed from `mtpChatKnownGaps` (both edit-turn rows of ^achvxy1; the dictionary is now empty, and the unused `mtpCheckpointGap` text is deleted).

    Release qwen38FourBitMTPSystemPromptCache (run 2, after the row removal, passed with 0 known issues):
    | scenario/turn | MTP cached/prefilled, prefill ms | MTP spec, tok/s | plain cached/prefilled, prefill ms | plain tok/s | match |
    | follow-ups t0 | 0/2212, 1834.9 | MTP, 85.9 | 0/2212, 1827.7 | 56.5 | exact |
    | follow-ups t1 | 2403/38, 111.1 | MTP, 81.3 | 2404/37, 124.5 | 56.1 | exact |
    | follow-ups t2 | 2544/29, 107.1 | MTP, 88.9 | 2544/29, 122.8 | 56.1 | near-tie tok 1, gap 0.125 |
    | thinking t0 | 0/2191, 1804.2 | MTP, 85.5 | 0/2191, 1793.8 | 54.3 | exact |
    | thinking t1 | 2302/25, 94.4 | MTP, 86.1 | 2303/24, 123.1 | 55.3 | near-tie tok 52, gap 0.125 |
    | thinking t2 | 2423/25, 94.5 | MTP, 83.7 | 2425/24, 112.6 | 45.4 | near-tie tok 0, gap 0.375 |
    | edit t0 | 0/2197, 1824.5 | MTP, 86.8 | 0/2197, 1800.1 | 56.1 | exact |
    | edit t1 | 2264/26, 105.8 | MTP, 88.9 | 2265/25, 121.0 | 56.1 | exact |
    | edit t2 (edit) | 2169/123, 170.9 | MTP, 87.3 | 2169/123, 185.2 | 55.7 | exact |
    In every turn, drafter prefilled == prefilled and there is no fallback. Before this change the edit turn recomputed 0/2292 in 1.6–1.9 s. Run 1, still with the rows, recorded "Known issue was not recorded" twice and showed the same numbers (edit t2 2169/123 in 171.4 ms). One outlier in run 1: edit t0 decoded at 58.7 tok/s; it was 86.8 in run 2.
    Costs found: warm MTP prefill is 94–111 ms, against 67–84 ms before this change (it now pays the checkpoint feed and copies plain pays; still below plain's 112–124 ms). Cold MTP prefill is about plain's (1804–1835 vs 1794–1828 ms), where it was about 4% faster before.
    Release qwen38FourBitMTPGeneration: run 1 MTP 89.7 / 79.9 / 78.1 tok/s, run 2 89.8 / 81.4 / 78.1 (plain about 59.4), acceptance 93.9 / 73.6 / 69.0%, matches plain: true. The baseline was about 89 / 80 / 78.
    After the benchmarks, takePromptEndCheckpoint got an @autoclosure and swift-format ran; the result does not change when checkpoints are kept, so the benchmarks were not run again.
  timestamp: 2026-10-28T01:39:56.956382+00:00
- actor: claude-code
  id: 01m602mawrt8wsmmqstgqsch8a
  text: |-
    ### implement — changed
    - evidence: 10 files — Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift, Libraries/MLXLMCommon/MTPDrafterContinuation.swift, Libraries/MLXLMCommon/MTPDrafterPrefillFeed.swift, Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/SpeculativeDecoding.swift, Tests/MLXLMTestSupport/Qwen35MTPFixture.swift, Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift, IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift. `swift build --build-tests`: only the known Cmlx.bundle warning. `xcrun xctest MLXLMTests.xctest`: 895 XCTest, 0 failures; 1883 Swift Testing passed (1874 + 9 new; benchmarkVerification env-gated skip, known). MLXFoundationModelsTests: 376 passed. swift-format lint is clean. Release qwen38FourBitMTPSystemPromptCache passed with no known issues; the edit turn is MTP 2169 cached / 123 prefilled in 170.9 ms at 87.3 tok/s, exact match. Release qwen38FourBitMTPGeneration MTP 89.7/79.9/78.1 and 89.8/81.4/78.1 tok/s, matches plain.
    - next: /review (task left in doing; not committed)
  timestamp: 2026-10-28T01:40:09.752735+00:00
- actor: claude-code
  id: 01m6031aecfhppjxsqm0n6tdd1
  text: |-
    ### test — green
    - evidence: touched changed .swift files; swift build --build-tests (no compiler warnings; only known "missing creator ... mlx-swift_Cmlx.bundle"); xcrun xctest MLXLMTests — 895 XCTest + 1883 Swift Testing, 0 failures, only skip is env-gated benchmarkVerification(); MLXFoundationModelsTests — 376 passed, 0 failures; Release xcodebuild qwen38FourBitMTPSystemPromptCache() — TEST SUCCEEDED, 0 issues/known issues; edit turn 2: prefilled 123 tok with MTP on (cached 2169), text equal to plain, exact token match, decode 1.57x plain, acceptance 94.3%.
    - note: Release build shows 6 pre-existing #UnnecessaryEffectMarker warnings in untouched files TokenizerVocabExtractorTests.swift (1) and HardReserveStressTests.swift (5); not from this task.
    - next: ready for review/commit
  timestamp: 2026-10-28T01:47:15.276346+00:00
- actor: claude-code
  id: 01m603cptvzr0j1a73kxzjgdb9
  text: |-
    ### review — findings
    - evidence: `review sha HEAD~1..HEAD` (a33c6877): 2 findings (2 confirmed, 2 refuted, 14 attempted, 0 failed) — Libraries/MLXLMCommon/ChatSession.swift:1874 (duplication/duplication), Libraries/MLXLMCommon/ChatSession.swift:2431 (reuse/reuse).
    - requested checks (no defect found):
      1. Alignment: `PromptCacheCheckpoint.init` stores the continuation only when `isAligned(withMainTokenCount: tokens.count)`, and only when the caches stand at `tokens.count`. The feed starts at `kvCache.processedTokenCount` (= prefillStart), restarts the drafter at that absolute position when the carried continuation is not aligned, and reports `startPosition + fedTokenCount`. A mid-prefill checkpoint therefore stores the drafter state at that boundary. The prompt-end checkpoint drops the first-bonus entry and checks alignment again.
      2. Sharing: a checkpoint stores `compacted()` (`copy()` + `contiguous`), never the live value. The feed extends only the session-local carried continuation in place, and a fork deep-copies that (`RealizedCache.copy`). No production code reads `checkpoint.drafterContinuation` yet (`.rebuild` + restore sets `mtpDrafterContinuation = nil`), so no turn can mutate a stored one. ^ahnb0j9 must hand out `copy()` when it adds the restore, as its hand-off note says.
      3. Memory policy: `mtpFallbackReason` runs before any cache decision or trim. It counts drafter parameters plus the carried continuation's byteCount once. The drafter reaches the target embedding/head through the call, so they are not double counted. On fallback, `drafterFeed` is nil and `defaultGeneration` runs; the carried continuation is then replaced by the plain generation's nil, so the session stays usable and the next turn can cold-start.
      4. Incompatible drafter: `isCompatible` is checked first and makes `speculationIsEligible` false. That leaves `drafterFeed` nil and goes to `defaultGeneration`, so no MTP state is created.
      5. `keepsCheckpoints`: needs `usesMTP && mtpSpeculativeStateIsReady` (not stateful, or resumable), so a stateful non-resumable drafter keeps checkpoints off.
    - next: fix the 2 findings, then re-run `/review 01M5QWGWNFDVP77YAW9ACHVXY1 <sha>`.
  timestamp: 2026-10-28T01:53:28.411079+00:00
- actor: claude-code
  id: 01m603dafqs4cn0fxy7grq80kt
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 10 files (checkpoints with MTP on, compacted drafter continuation per checkpoint, incompatible-drafter fallback, MTP memory policy)
    - test: green — MLXLMTests 895 XCTest 0 failures + 1883 Swift Testing; FoundationModels 376; Release SystemPromptCache passed, edit turn 2169/123 with MTP on
    - commit: a33c6877
    - review: findings — Libraries/MLXLMCommon/ChatSession.swift:1874, Libraries/MLXLMCommon/ChatSession.swift:2431
  timestamp: 2026-10-28T01:53:48.535867+00:00
- actor: claude-code
  id: 01m6040p7tzcydmgbvs6ch0xax
  text: |-
    Review fixes (both pure refactors):
    - duplication (ChatSession.swift:1874): `takePromptEndCheckpoint` is replaced by a local `finishPrompt(_:drafterContinuation:) -> GenerationRun`. It adds `checkpointPrefillTime`, takes the prompt-end checkpoint, and calls `startGeneration`. The plain path passes `{ _ in nil }`; the MTP path passes `\.promptBoundaryContinuation`. The continuation is still read only when the pass keeps checkpoints. The suggested `inout I` + `@autoclosure` shape does not compile: the autoclosure `iterator.promptBoundaryContinuation` overlaps the inout access to `iterator`. So the helper takes the iterator by value plus an `(Iterator) -> MTPDrafterContinuation?` reader. `TokenIteratorProtocol.promptPrefillTime` is get-only, so a file-private `PromptPrefillTimedIterator` (get set) refines it; `TokenIterator` and `MTPSpeculativeTokenIterator` conform to it in ChatSession.swift.
    - reuse (ChatSession.swift:2431): new `SpeculativeDecodingMemoryPolicy.admitsSpeculation(mainModelBytes:draftModelBytes:) throws -> Bool`, plus a `(mainModel:draftModel:)` overload. It returns true when admitted and false on fallback, and throws `SpeculativeDecodingMemoryError(evaluation:)` on a refused `.fail`. `mtpFallbackReason`, the pre-load deferred-draft check and the post-load draft check all call it. The package `evaluate(mainModel:draftModel:)` lost its only caller and is deleted. The pre-load verdict is kept as a `Result<Bool, any Error>?`, so the `.fail` throw still happens only when the draft-model branch is taken, as before. A turn that falls back for media or a missing continuation still does not throw, and `willFallBackBeforeLoadingDraft` keeps its old meaning (`(try? result.get()) == false`).
    - Tests: 3 new Swift Testing tests in SpeculativeDecodingConfigTests (within budget admits; over budget with allow/fallback follows the action; over budget with fail throws the policy's evaluation). Watched 2 of them fail against a `true` stub.
  timestamp: 2026-10-28T02:04:23.162216+00:00
- actor: claude-code
  id: 01m6040t98c32traddrhbhs82m
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLMCommon/SpeculativeDecoding.swift, Tests/MLXLMTests/SpeculativeDecodingConfigTests.swift. `swift build --build-tests`: only the known Cmlx.bundle "missing creator" warning. `xcrun xctest MLXLMTests.xctest`: 895 XCTest, 0 failures; 1886 Swift Testing passed (1883 + 3 new; benchmarkVerification env-gated skip, known). MLXFoundationModelsTests: 376 passed. swift-format lint is clean. Release qwen38FourBitMTPSystemPromptCache(): TEST SUCCEEDED, no known issues. Edit turn 2 with MTP: 2169 cached / 123 prefilled in 170.4 ms, drafter prefilled 123, 88.4 tok/s, exact match with plain. The only Release compile warnings are the 6 pre-existing ones in HardReserveStressTests.swift and TokenizerVocabExtractorTests.swift. Both review findings are checked.
    - next: /review (task left in doing; not committed)
  timestamp: 2026-10-28T02:04:27.304441+00:00
- actor: claude-code
  id: 01m6043vak9pwc8x5krdeyhk9y
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (e5a96cab) — 0 findings (attempted 7, confirmed 0, refuted 0, failed 0); 3 files reviewed, 2 .kanban files skipped (no validator). Both prior items from 2026-10-27 20:47 are checked and fixed in the diff: ChatSession.swift duplication → `finishPrompt(_:drafterContinuation:)` used by both the plain and MTP paths; ChatSession.swift reuse → `SpeculativeDecodingMemoryPolicy.admitsSpeculation` used by `mtpFallbackReason` and both draft-model checks. Behavior kept: the before-loading verdict is captured as `Result` and only `.get()`-rethrown in the draft-model branch (`try?` in `willFallBackBeforeLoadingDraft` maps a fail error to "no fallback", as before); the drafter continuation closure is invoked only inside `if keepsCheckpoints`.
    - next: none — task moved to done.
  timestamp: 2026-10-28T02:06:06.675664+00:00
- actor: claude-code
  id: 01m6044ap0kzc6tfe8yehdrpsv
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 3 files (finishPrompt helper; SpeculativeDecodingMemoryPolicy.admitsSpeculation)
    - test: green — implementer's run on the final code: MLXLMTests 895 XCTest 0 failures + 1886 Swift Testing; FoundationModels 376; Release SystemPromptCache passed, edit turn 2169/123 with MTP on
    - commit: e5a96cab
    - review: clean — 0 findings, prior 2/2 verified; task done
  timestamp: 2026-10-28T02:06:22.400767+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: ffeb80
title: 'MTP: ChatSession MTP gaps after PR 581 (checkpoints, incompatible drafter, cold start, memory policy)'
---
## What
Upstream PR 581 (commit 843854ec) added `SpeculativeDecodingConfig(mtpDrafter: MTPDrafterContainer, blockSize:) throws` with a `Strategy` enum, `MTPDrafterContinuation` in `RealizedCache`, the append-suffix reuse, fork copy, and typed `SpeculativeDecodingFallbackReason`. Tests: Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift.

What remains in Libraries/MLXLMCommon/ChatSession.swift:
- [x] Checkpoints: the fork takes checkpoints only when `speculativeDecoding == nil` (~1365-1371). Allow them for the MTP strategy. `prefillToCheckpoint` (~1562-1642) must feed the drafter (per-chunk hidden-row hook of the chunked-prefill task) so that the continuation stays aligned across that boundary.
- [x] An incompatible drafter: PR 581 throws from `SpeculativeDecodingConfig(mtpDrafter:)`. Make the session fall back to plain decode with a fallback reason (keep a throwing validation API for callers that want it).
- [x] `unavailableContinuation`: use the cold start from the drafter task instead of target-only. (Hybrid rebuild now restores a checkpoint and cold-starts there; the trimmable-target rewind `.trimToCommonPrefixMainOnly` is the reuse-policy change of ^ahnb0j9, per the scope of this pass.)
- [x] `memoryPolicy` for the MTP strategy (PR 581 sets it to nil): report the drafter bytes to `SpeculativeDecodingMemoryPolicy`.

## Acceptance Criteria
- [x] A two-turn `ChatSession` with a system prompt at temperature 0 gives the same text with MTP as without speculation, the checkpoints are taken with MTP on, and turn 2 feeds the drafter only over the new suffix.
- [x] An incompatible drafter gives plain decode with a fallback reason, and does not throw.
- [x] A memory policy that does not admit the drafter gives plain decode (fallback action) or throws (fail action).

## Tests
- [x] Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift: checkpoints with MTP, incompatible drafter, cold start, memory policy.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session

## Review Findings (2026-10-27 20:47)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 6 not reviewed.

> 6 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWGWNFDVP77YAW9ACHVXY1.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWGWNFDVP77YAW9ACHVXY1.md` — no validator matches this file
> - `.kanban/tasks/01M5QWHA34WP7DQMWQ4AHNB0J9.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWHA34WP7DQMWQ4AHNB0J9.md` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5SJPFM01WFRYWMQXVGMA4MQ.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/ChatSession.swift:1874` `duplication/duplication` — The two generation branches repeat the same prompt-end sequence: add `checkpointPrefillTime` to the iterator, take the prompt-end checkpoint from `iterator.state`, then call `startGeneration`. They differ only in the drafter-continuation argument, so the sequence can drift between the MTP and plain paths. Move the shared three steps into one local helper that takes the iterator and the drafter continuation, for example `func finishPrompt<I: TokenIteratorProtocol>(_ iterator: inout I, drafterContinuation: @autoclosure () -> MTPDrafterContinuation?) -> GenerationRun`. Call it from both branches so each passes only its own continuation value.
- [x] `Libraries/MLXLMCommon/ChatSession.swift:2431` `reuse/reuse` — `mtpFallbackReason` rebuilds the memory-policy outcome handling that the draft-model path already implements. It evaluates the policy, throws `SpeculativeDecodingMemoryError` on `.fail`, and otherwise maps a refusal to a fallback. The existing draft-model branch does the same thing with `speculativeMemoryEvaluation`. Two copies of that decision can drift, for example if the `.fail` rule or the error payload changes in one place only. Move the fail-or-refuse decision into one helper on `SpeculativeDecodingMemoryPolicy` (for example a method that returns the evaluation and throws on `.fail`) and call it from both `mtpFallbackReason` and the draft-model branch. Each caller then maps the refusal to its own fallback reason. Alternatively, extend the existing `evaluate` path so the MTP check reuses it instead of re-deriving the outcome.
