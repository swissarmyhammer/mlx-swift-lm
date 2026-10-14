---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4y74g1btvwmsrq01p6kccrf
  text: |-
    Measurement before change (HEAD 155eec4, M3 Ultra, GPU idle, bench mode `compproj` in scratchpad dsv4bench, 124 compressor projections: 84 on ratio-4 layers, 40 on ratio-128 layers, all 4-bit affine).
    - Sync only (one tiny op): 0.221 ms.
    - Decode, all 124 projections of one token, one eval: 0.767 ms (about 0.55 ms after the sync). 16 tokens in one graph: 6.849 ms, thus 0.43 ms for each token.
    - Decode, ratio-4 projections of one token: 0.670 ms. Ratio-128 projections of one token: 0.460 ms.
    - Boundary step, ratio-4 projections of 4 rows: 0.892 ms.
    - Prefill, all projections of one 2048-row block: 52.6 ms, thus about 0.84 s of the about 58.5 s 32k prefill (1.4 %). The change does not remove prefill work, because each row is projected one time in both designs.
    Estimate: decode saves the ratio-128 projections on most steps (about 0.24 ms) and 3 of 4 ratio-4 projections (about 0.28 ms), thus about 0.4 to 0.5 ms/token of 30.4 ms. This is more than the usual decode noise, thus the work continues.
  timestamp: 2026-10-14T22:04:42.923495+00:00
- actor: claude-code
  id: 01m4y9qg1vx756jhk4q7vvq3tf
  text: |-
    Implementation landed (not committed). `DeepSeekV4ChunkCache` keeps the raw block rows of a run that no block has ended (`unprojectedRows`, a row buffer that grows by `chunkWidth` rows). The block that ends a run projects those rows and its own tokens in one `wkv` and one `wgate` call (`projectRun(endingWith:through:)`), then pools as before. A block that ends no run does no projection and no pooling. Rewind truncates the unprojected rows when the position is past the projected rows, else it truncates the projected rows and empties the unprojected rows. The branch state has five slots now (unprojected rows at slot 3, carry at slot 4); the format is `DeepSeekV4Cache.unprojectedRun.v4`, and the v2 and v3 formats fail closed. `DeepSeekV4RowBuffer` has an `allocationStep` for each buffer and a `removeAll()` that keeps a small storage.

    Tests (TDD): RED on HEAD was `aStepProjectsTheRowsOfARunOnlyWhenItEndsTheRun` (18 issues), the premise of `theCacheRewindsInsideTheRunItHasNotProjected`, and the v3 case of `aPromptCacheOfAProjectedRunFormatFailsClosed`. New tests also: `aRunOfOneTokenStepsPoolsTheChunksOfRowsProjectedOneTokenAtATime` (compares with the old one-token projection path, tolerance 1e-5) and `aStateSavedInsideARunPoolsTheChunksOfAnUnbrokenRun`. The branch-state fixture has three row slots.

    Measurements, 32k, GPU idle, bench args `<snap> 32768 128 8192 1024`, DSV4_PREFILL_STEP default:
    - Before (HEAD file): full 559.7 / 559.8 tok/s prefill, 30.4 / 30.3 ms/tok decode, peak 143 / 143 GiB. Profile 31.7 / 31.7 ms/step. Step mode B (pipelined) 30.9 / 31.1 ms, C 31.3 / 31.1 ms.
    - After: full 559.6 / 559.4 tok/s, 30.8 / 29.8 ms/tok (third run 29.9 ms/tok, 559.5 tok/s), peak 143 / 143 GiB. Profile 31.5 / 31.2 ms/step. Step mode B 30.4 / 30.7 ms, C 31.1 / 30.3 ms.
    - Result: decode about 0.3 to 0.5 ms/token faster in each pipelined measure; prefill and peak memory do not change. The full-mode decode has noise of about 1 ms (one 30.8 ms run), thus the profile and step numbers are the better evidence. Prompt cache checks of the bench: HARD reuse PASS and tie-aware PASS in all runs.
    Logs: scratchpad `mtxa-before-*.log`, `mtxa-after-*.log`.
  timestamp: 2026-10-14T22:50:02.683022+00:00
- actor: claude-code
  id: 01m4y9qk58a25j602yyx25rebn
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, Tests/MLXLMTests/DeepSeekV4CacheTests.swift, Tests/MLXLMTests/DeepSeekV4BranchStateFixture.swift, Tests/MLXLMTests/KVCacheByteCountTests.swift, Tests/MLXLMTests/PromptCacheTemplateRestoreTests.swift. `swift build --build-tests` clean (only the SwiftPM "missing creator for mutated node" note that HEAD also gives). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest 0 failures, 1453 Swift Testing tests in 115 suites passed. Greedy parity `greedyFirstTokensMatchThePythonFixture()` passed. 32k decode 30.35 -> about 30.0 ms/tok (profile 31.7 -> 31.35 ms/step), prefill and peak unchanged.
    - next: review
  timestamp: 2026-10-14T22:50:05.864702+00:00
- actor: claude-code
  id: 01m4y9wh0gtcseyncatk4mdfp9
  text: |-
    ### review — findings
    - evidence: `review sha HEAD~1..HEAD` (commit c97d071): 1 finding, 1 confirmed, 0 refuted — Tests/MLXLMTests/DeepSeekV4CacheTests.swift:328 (`reuse/reuse`).
    - next: implement the open item in the "Review Findings (2026-10-14 17:50)" section. The task stays in review.
  timestamp: 2026-10-14T22:52:47.504428+00:00
- actor: claude-code
  id: 01m4y9ww78zwdak8e0z44rsbmx
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 files; 32k profile window 31.7 → 31.2-31.5 ms/step, prefill and peak no change
    - test: green — xcrun xctest MLXLMTests, 716 XCTest and 1453 Swift Testing passed, 0 failures; greedy parity passed
    - commit: c97d071
    - review: findings — Tests/MLXLMTests/DeepSeekV4CacheTests.swift:328
  timestamp: 2026-10-14T22:52:58.984960+00:00
- actor: claude-code
  id: 01m4ya484nnxd506f4jar3j3ke
  text: 'Review finding fixed (not committed). `DeepSeekV4ChunkCache.projectedEnd` is internal now (it was `private`), thus `@testable import` reads it. The test helper `projectedEnd(of:)` is deleted. The two tests `aStepProjectsTheRowsOfARunOnlyWhenItEndsTheRun` and `theCacheRewindsInsideTheRunItHasNotProjected` read `cache.projectedEnd`. I examined all the other test helpers and computations in the file: `lastChunkEnd`, `wholeCount` and `leading` are independent expected values or fixture premises, not copies of a cache value, thus they stay. RED: the build failed with "''projectedEnd'' is inaccessible due to ''private'' protection level". GREEN: after the access change, the build passed.'
  timestamp: 2026-10-14T22:57:00.565880+00:00
- actor: claude-code
  id: 01m4ya49ttph1eth840s70jhhh
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXLLM/Models/DeepSeekV4Cache.swift, Tests/MLXLMTests/DeepSeekV4CacheTests.swift. `swift build --build-tests`: 0 warnings (only the known "missing creator" note). `xcrun xctest`: XCTest 716 tests, 0 failures; Swift Testing 1453 tests in 115 suites passed.
    - next: /review. The task stays in doing.
  timestamp: 2026-10-14T22:57:02.298012+00:00
- actor: claude-code
  id: 01m4ya61j7534g4q0z4tkj5801
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (f205a3d). Counts: 0 findings, 0 confirmed, 0 refuted. 7 validator runs, 0 failed. The prior finding at DeepSeekV4CacheTests.swift:328 is checked.
    - next: none. The task is in done.
  timestamp: 2026-10-14T22:57:59.367205+00:00
- actor: claude-code
  id: 01m4ya6fcc82c38ffzk9w0nfwm
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 2 files; projectedEnd is internal, the test helper is removed
    - test: green — xcrun xctest MLXLMTests, 716 XCTest and 1453 Swift Testing passed, 0 failures
    - commit: f205a3d
    - review: clean — 0 findings; prior finding checked
  timestamp: 2026-10-14T22:58:13.516612+00:00
position_column: done
position_ordinal: ffd080
title: 'DeepSeek-V4 decode speed: project the compressor rows only at a chunk boundary'
---
## What

Found on 2026-10-14 in a study of osaurus (scratchpad `osaurus/REPORT.md`, idea 2).

Our cache projects each new token through `wkv` and `wgate` on every decode step (`DeepSeekV4Cache.swift` about line 375). Osaurus keeps the raw rows and projects the whole chunk once at the chunk boundary (vmlx-swift commit fd93476a, `DeepseekV4Compressor.swift:1125-1172` and 1324-1379). They measured +8.3 % decode at 2k context.

Estimate at 32k: about 136 MB less weight read and about 120 fewer dispatches on most steps, decode -0.3 to -0.6 ms/token. A block projection can use a different matmul kernel than the one-token path, thus the bits can change. Rewind and the cache state (format version) must change. Risk: low to medium.

## Acceptance Criteria

- [x] A unit test compares the pooled rows with the current path (tolerance if not bit-exact).
- [x] Decode ms/token at 32k, before and after, GPU idle.
- [x] Cache save, load and rewind tests pass. The greedy parity test passes. #deepseek-v4 #performance

## Review Findings (2026-10-14 17:50)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 6 not reviewed.

> 6 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4XSZAC5BBG53QG7XNGJPZMQ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XSZAC5BBG53QG7XNGJPZMQ.md` — no validator matches this file
> - `.kanban/tasks/01M4XSZE6QXG6H3SK9V4N7MTXA.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XSZE6QXG6H3SK9V4N7MTXA.md` — no validator matches this file
> - `.kanban/tasks/01M4Y6YAFH5N7R8P3XAJYFKA2H.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4Y6YAFH5N7R8P3XAJYFKA2H.md` — no validator matches this file

- [x] `Tests/MLXLMTests/DeepSeekV4CacheTests.swift:328` `reuse/reuse` — The test helper `projectedEnd(of:)` recomputes the position after the last projected row from `carryStart` and the projected values count. The cache already computes this in its private `projectedEnd` property. The test copy can drift from the cache rule if the cache rule changes. Expose the cache's `projectedEnd` to the test module, for example as `private(set) var` or an internal getter, and delete the test helper. Then the test reads the same value the cache uses.