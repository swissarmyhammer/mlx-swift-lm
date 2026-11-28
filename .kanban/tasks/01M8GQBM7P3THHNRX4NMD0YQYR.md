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
position_column: doing
position_ordinal: '80'
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