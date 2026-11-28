---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8gsphyyb9hxe5ztrytb04gr
  text: |-
    Research (picked up):
    - Rank 0 keeps a RemoteDrafterTrack per handle in RemoteModelExecutor. A cold restart is visible as the track's `state.cacheStartPosition` > 0: `didForward` on a non-aligned phase builds `coldState(at: position)`. A track that has followed the handle since position 0 keeps start 0 (same witness the local ^33h7w11 tests use: `cacheStartPosition == 0`).
    - Plan for observation without a wire change: `RemoteModelExecutor.drafterStartPosition(of:)` (package), and InProcessRemoteModelWorker records it after each `.speculate` op, so a test reads the start of rank 0's drafter at the first round of the next turn.
    - Ops per flow: guided cluster turn ends with `trimSpeculated(rejected)` right after the last verify (settles: kept >= 1). Plain cluster turn ends with `finalizeGeneration` trim right after the last verify. A checkpoint of a remote hybrid cache is a worker `copy` taken after `prefillToCheckpoint` forwards end at the boundary (aligned). A restore copies that copy. A prompt-cache trim of the live cache (divergence / re-render) is a trim outside `.verifying` → `.cold` today.
    - Flows to test: GuidedPromptCacheReuseTests (clustered, speculative) for schema→plain, schema→schema, plain→schema; PromptCacheDivergentCopyTests (Inkling + DSpark on rank 0) for the seeded copy.
  timestamp: 2026-11-28T10:02:34.334379+00:00
- actor: claude-code
  id: 01m8gtnjqkbmp009xytmxh6kvg
  text: |-
    Flows measured (before any fix): ALL WARM. Witness = rank-0 drafter `cacheStartPosition` at the first round of the next turn (0 = followed since position 0; a cold start sets it to the position where it restarted).
    - Qwen 3.5 MTP, clustered (GuidedPromptCacheReuseTests), each x rewritten tail 0/1/2/3 chars (re-render of the end of the first response): schema->plain, schema->schema, plain->schema: start 0 every case. cached prefix: tail 0 extends (44/297/77), tails >= 1-2 restore a checkpoint (14 prompt-end, 267) — the hybrid remote cache cannot trim the live cache, so every rewind is a worker `copy` of a checkpoint taken where forwards ended, i.e. aligned.
    - Inkling DSpark, clustered (PromptCacheDivergentCopyTests), sliding window 8 and 64: template-seeded copy (whole ledger, and keepingTokens = instructions) diverging after the instructions: restores the instructions checkpoint (cached 12), start 0. Re-rendered tail of 2 tokens in the same session: restores the prompt-end checkpoint (cached 20), start 0. The conv layers keep even the wide-window cache from trimming.
    - End-of-generation trims (plain finalizeGeneration, guided trimSpeculated) are the op right after the last verify, so the verify settles aligned.
    - Proof the witness catches cold: temporary sabotage `copy()` -> `.cold` made the warm tests fail with start 12/14/20/267 (every flow that restores a checkpoint); reverted.
    - Acceptance counts (second turn, warm; identical with the sabotaged cold copy on these random-weight fixtures): Qwen schema->plain 28/34, schema->schema 8/22, plain->schema 8/22; Inkling DSpark seeded 0-1 of 36-39, re-render 1/36-38.
    - No cold flow, so no production fix and no protocol change (RemoteModelProtocol.version unchanged, no #cluster card). A trim that is not right after a verify still makes the track cold (`didTrim` outside `.verifying`), but no flow here reaches it: every shardable family is hybrid, so the prompt cache rewinds by checkpoint copy, not by trim.
  timestamp: 2026-11-28T10:19:30.931426+00:00
- actor: claude-code
  id: 01m8gv3zr22wr9yphgme8bwdcs
  text: |-
    ### implement — changed
    - evidence: before any change, every flow on the card was already warm (see the previous comment). There is no cold flow, so nothing needed a production fix and there is no protocol change: RemoteModelProtocol.version is unchanged and no #cluster card was filed. `swift build --build-tests` is clean. `swift-format format` and `lint` are clean on the changed files. xcrun xctest results: MLXLMTests full bundle in one process, 2849 Swift Testing tests (3 known issues, which existed before this change: AttentionKernelTolerance and MiniCPM5) and 1261 XCTest tests with 0 failures; MLXFoundationModelsTests 770 passed; MLXGuidedGenerationTests 96 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 passed.
    - files: Libraries/MLXLMCommon/Remote/RemoteDrafterTrack.swift (`startPosition`), Libraries/MLXLMCommon/Remote/RemoteModelExecutor.swift (package `drafterStartPosition(of:)`), Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift (`drafterStartPositionsOfRounds`, recorded after each speculate op), Tests/MLXLMTestSupport/Inkling2TinyModels.swift (`draftJSON(slidingWindowSize:)`, default 8), Tests/MLXFoundationModelsTests/GuidedPromptCacheReuseTests.swift, Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift.
    - tests added:
      - Qwen, clustered: `aClusteredSecondTurnStartsTheDrafterOfRankZeroWarm` and `aClusteredSecondTurnWritesTheTextOfAColdTurn` (schema→plain, schema→schema, plain→schema, each with a rewritten tail of 0/1/2/3 characters), and `aClusteredSecondTurnReportsTheDraftsTheTargetAccepted`.
      - Inkling DSpark, clustered, sliding window 8 and 64: seeded copy (whole ledger and instructions only) starts warm, answers as a cold session does, and reports its draft counts. A re-rendered tail starts warm and answers as a cold session does.
      - The warm tests were checked for a red result by a temporary sabotage that makes `copy()` set the track cold; that sabotage is reverted.
    - next: /review
  timestamp: 2026-11-28T10:27:23.010637+00:00
- actor: claude-code
  id: 01m8gvrmjmxa47cpwt7se9kpe3
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` complete, 0 compiler warnings (one SwiftPM build-system notice: "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS", not from source). `swift-format lint` on the 8 changed Swift files: clean. Bundles via `xcrun xctest .build/out/Products/Debug/<Bundle>.xctest`, all exit 0, each ended with its summary line:
      - MLXLMTests: XCTest 1261 tests 0 failures; Swift Testing 2849 tests in 282 suites passed (3 known issues)
      - MLXGuidedGenerationTests: Swift Testing 96 tests in 23 suites passed
      - MLXFoundationModelsTests: Swift Testing 770 tests in 99 suites passed
      - CXGrammarTests: Swift Testing 7 tests in 5 suites passed
      - MLXHuggingFaceMacrosTests: XCTest 11 tests 0 failures (no Swift Testing tests, so no "Test run" line)
    - known issues (pre-existing, in files this change does not touch): `AttentionKernelToleranceTests.aNearTieRejectsATokenOutsideTheTwoBest` (1) and `MiniCPM5ToolCallParserTests` CDATA close-tag test (2). Both are marked as expected issues, and both passed.
    - no skipped tests in the logs.
    - next: none

    step: test
    outcome: green
    evidence: build-tests clean, swift-format lint clean; MLXLMTests 1261 XCTest + 2849 ST pass; MLXGuidedGenerationTests 96 pass; MLXFoundationModelsTests 770 pass; CXGrammarTests 7 pass; MLXHuggingFaceMacrosTests 11 pass; 0 failures, 0 skipped
    task: ^md0yqyr
  timestamp: 2026-11-28T10:38:39.700101+00:00
- actor: claude-code
  id: 01m8gvxyp0wgzqeqkh6fpqjj87
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (549de38d); 7 findings — Tests/MLXFoundationModelsTests/GuidedPromptCacheReuseTests.swift:522, Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:305, :306, :351, :356, :384, :484
    - next: implement the 7 unchecked items in `## Review Findings (2026-11-28 04:39)`; task stays in review
  timestamp: 2026-11-28T10:41:33.888260+00:00
- actor: claude-code
  id: 01m8gvyewth49vnxjr3kexb3rd
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — every flow was already warm; read-only start-position hook + warm-start tests
    - test: green — all 5 bundles (MLXLMTests 2849+1261, MLXFoundationModelsTests 770, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 549de38d
    - review: findings — GuidedPromptCacheReuseTests.swift:522; PromptCacheDivergentCopyTests.swift:305, :306, :351, :356, :384, :484
  timestamp: 2026-11-28T10:41:50.490594+00:00
- actor: claude-code
  id: 01m8gwnrxvp9026m9g7tcac9h8
  text: |-
    ### implement — changed
    - evidence: fixed all 7 findings of `## Review Findings (2026-11-28 04:39)`, all flipped to [x].
      - reuse: new `InProcessRemoteModelWorker.drafterStarts(during:) async rethrows -> (result:, starts:)` (generic over the pass result). It counts the recorded rounds, runs the pass, and returns the starts the pass added. `ClusteredSecondTurn`, `SeededSpeculativePass` and `RerenderedSpeculativePass` all call it, so no count-then-dropFirst code is left.
      - magic numbers: the window sizes 8/64 are now `narrowWindowSize`/`wideWindowSize` static constants on the suite. `followUpPrompt` moved from the extension into the suite body beside the other token prompts. swiftlint `no_magic_numbers` does not count a static let array inside an extension as a named value, but it does count one in the type body. A swiftlint run with the validator's config (allowed [0,1,-1,100]) on all 3 files now reports 0.
      - test cheating: `theSpeculativeSeededSessionReportsItsDraftCounts` now also expects `accepted >= 0` and `accepted <= drafted`. A comment says why zero is legitimate: random weights, and the earlier measurement accepted 0-1 of 36-39. The Qwen `aClusteredSecondTurnReportsTheDraftsTheTargetAccepted` had the same gap (only `accepted > 0`), so it now also expects `accepted <= drafted`. The pre-existing `aSpeculativeSecondSchemaTurnSpeculates` already checks `drafted > 0` and never reads accepted, so it does not have this weakness.
      - all existing assertions are kept.
    - verification: swift-format format plus lint are clean on the 3 files. `swift build --build-tests` is clean. MLXFoundationModelsTests: 770 ST tests passed. MLXLMTests in one process: 1261 XCTest tests with 0 failures, and 2849 ST tests passed (3 known issues, which existed before).
    - files: Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift, Tests/MLXFoundationModelsTests/GuidedPromptCacheReuseTests.swift, Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift
    - next: /review
  timestamp: 2026-11-28T10:54:34.427763+00:00
- actor: claude-code
  id: 01m8gxenzb7xc9ah50z9jnajcd
  text: |-
    step: test
    outcome: green
    evidence: swift build --build-tests complete (no warnings from sources; only the SwiftPM bundle-node notice "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS", a build-system message, not source); swift-format lint on the 3 changed files clean; xcrun xctest per bundle, each ended with its summary line, exit 0: MLXLMTests 1261 XCTest 0 failures + Swift Testing 2849 tests/282 suites passed (3 known issues, pre-existing withKnownIssue in AttentionKernelToleranceTests and MiniCPM5ToolCallParserTests, files not touched by this change); MLXGuidedGenerationTests 96 tests/23 suites passed; MLXFoundationModelsTests 770 tests/99 suites passed; CXGrammarTests 7 tests/5 suites passed; MLXHuggingFaceMacrosTests 11 XCTest 0 failures. 0 skipped (only matches are test names containing "Skipped").
    task: ^md0yqyr
  timestamp: 2026-11-28T11:08:10.603966+00:00
- actor: claude-code
  id: 01m8gxh4an80w1v5hjag6c2e14
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (9aa2c35d) — 0 findings (7 attempted, 2 refuted, 0 failed); 3 files reviewed, 2 .kanban files skipped (no validator); all 8 prior findings from 2026-11-28 04:39 checked
    - next: task moved to done
  timestamp: 2026-11-28T11:09:30.837403+00:00
- actor: claude-code
  id: 01m8gxhewe05gnrqcyrdzrajrb
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 3 test files, all 7 findings checked
    - test: green — all 5 bundles (MLXLMTests 2849+1261, MLXFoundationModelsTests 770, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 9aa2c35d
    - review: clean — 0 findings
  timestamp: 2026-11-28T11:09:41.646319+00:00
position_column: done
position_ordinal: ffffff9680
title: 'Cluster MTP: test (and fix) that rank 0''s drafter starts warm on the turn after a guided turn, a template-seeded copy, and a checkpoint restore'
---
## What
foundationmodelsrouter-22 asked: on a clustered model, after a guided turn under the ^9mvg23k design (rank 0 drafts unmasked and the host keeps what the grammar accepts), does the next turn's RemoteMTPDrafter start warm? ^33h7w11 fixed only the local path. Benches run Inkling and Qwen with MTP on the cluster, and SelectionTier makes many guided turns, each a new Router session seeded from a template copy (^jxyd0ja, ^3wjdpa4).

## What the code does today
Read in Libraries/MLXLMCommon/Remote/RemoteDrafterTrack.swift; none of this is tested for these flows:
- Rank 0 keeps a RemoteDrafterTrack for each cache handle.
- A forward on an aligned track extends it, so the drafter stays warm.
- A copy copies the track.
- A trim right after a verify settles the track and keeps it aligned.
- Any other trim makes it cold: `didTrim` outside `.verifying` sets `.cold`, and so does a trim longer than the verify. The next forward then starts from `coldState`.

So a turn that extends the ledger after a guided cluster turn should be warm. The drafter likely goes cold when:
- the next turn rewinds to a divergence point or to an end-of-message re-render (Qwen/DSML/Inkling splices);
- the end-of-generation trim of the last round's rejects isn't right after a verify;
- a restored checkpoint (a worker-side copy taken mid-prefill) carries a track that isn't aligned at its position.

## Do
1. Test it with the in-process cluster worker and the MTP fixtures (Qwen 3.5 MTP, Inkling DSpark). For each of these flows, check whether rank 0's track is aligned (not cold) at the start of the next speculative turn:
   - a speculative guided turn followed by a speculative turn, plain or guided;
   - a template-seeded copy whose prompt diverges after the instructions (checkpoint restore);
   - a plain speculative turn followed by a guided one.

   Expose the track state to tests if needed, without a protocol change if possible.
2. Fix each flow that goes cold. For example:
   - take worker checkpoints where the track is aligned;
   - let a trim into an aligned region rewind the track while the hidden rows are known.

   Output must not change, and no runtime guard.
3. If a fix needs a protocol change, bump the version and file a FoundationModelsCluster card for foundationmodelscluster-4e.

## Tests
- Each flow above reports a warm rank-0 drafter on the next turn.
- Output equals a cold run.
- Acceptance counts are reported.
#cluster

## Review Findings (2026-11-28 04:39)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8GQBM7P3THHNRX4NMD0YQYR.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8GQBM7P3THHNRX4NMD0YQYR.md` — no validator matches this file

- [x] `Tests/MLXFoundationModelsTests/GuidedPromptCacheReuseTests.swift:522` `reuse/reuse` — `ClusteredSecondTurn` re-implements the per-pass drafter-start capture: it records `worker.drafterStartPositionsOfRounds.count` before a pass, then slices the array after it. `PromptCacheDivergentCopyTests.swift` repeats the same pattern in `SeededSpeculativePass` (lines 349 and 351) and `RerenderedSpeculativePass` (lines 379 and 384). Three copies of the same count-then-dropFirst logic will drift if the recording changes. Give `InProcessRemoteModelWorker` (Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift) one helper that runs a pass and returns the starts of the rounds it added, for example `func drafterStarts(during pass: () async throws -> MTPTurn) async rethrows -> (MTPTurn, [Int])`. Then have all three structs call it, or extend the existing `drafterStartPositionsOfRounds` with a `since:` parameter so the callers share one slicing rule.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:305` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:306` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:351` `reuse/reuse` — `SeededSpeculativePass.init` repeats the count-then-slice capture of drafter starts that `ClusteredSecondTurn` already implements in `GuidedPromptCacheReuseTests.swift`. The two copies live in different files, so the shared form is not visible to either author. Use the same worker helper as the finding on `GuidedPromptCacheReuseTests.swift:522`, so this initializer reduces to one call.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:356` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:384` `reuse/reuse` — `RerenderedSpeculativePass.init` repeats the same drafter-start capture a third time, with its own local `firstRounds` counter, instead of using a shared worker helper. Replace the counter and slice with the shared helper proposed for `GuidedPromptCacheReuseTests.swift:522`.
- [x] `Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift:484` `test-integrity/no-test-cheating` — The test `theSpeculativeSeededSessionReportsItsDraftCounts` checks only that the accepted-draft count is non-nil. Any integer satisfies that, including 0 and a wrong count, so the assertion does not prove the target's accepted count is correct. This is a weakened assertion, and it is the only check on the accepted count in this test. Assert a bound that the protocol guarantees, such as `(accepted ?? -1) >= 0` and `(accepted ?? Int.max) <= (drafted ?? 0)`. If the drafter can legitimately accept zero drafts, say so in the test and check the accepted count against the number of drafts the speculative round reports.
