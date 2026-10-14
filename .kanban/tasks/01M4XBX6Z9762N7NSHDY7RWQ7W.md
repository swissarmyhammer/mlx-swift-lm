---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4xn80v37aq0fzzzbwg04njm
  text: 'Research: the three scale Gathers come from `DeepSeekV4Math.hcSplitOperands` (DeepSeekV4MathHelpers.swift): `scaleF32[preScaleIndex]`, `scaleF32[postScaleIndex]`, `scaleF32[combScaleIndex]`. In mlx-swift an `Int` subscript calls `take`, and `take` makes a `Gather` node (MLXArray+Indexing.swift). The decode path calls `hcSplitOperands` inside the compiled `collapseFields`, and compile keeps the Gather. 43 layers x 2 hyper-connections x 3 scales = 258 Gathers. The prefill path calls the same function, thus one change covers both. `DeepSeekV4HyperHead.callAsFunction` also does `mixScale.asType(.float32)[Self.scaleIndex]`, which is one more Gather for each step. Plan: a range slice of one element (shape `(1)`) in place of the Int subscript. A slice makes no kernel dispatch, and a `(1)` scale broadcasts to the same values, thus the bits stay the same. The head scale already has shape `(1)` and needs no index.'
  timestamp: 2026-10-14T16:52:04.067980+00:00
- actor: claude-code
  id: 01m4xpnpagb6a1ewg5kfa0hmay
  text: |-
    Implementation landed (not committed). `DeepSeekV4Math.hcSplitOperands` now reads each field scale through a new private helper `fieldScale(_:at:)`, which is the range slice `scale[index ... index]` (shape `(1)`, a view, no kernel). `DeepSeekV4HyperHead.callAsFunction` now multiplies by `mixScale` as it is (shape `(1)`), and the unused `scaleIndex` constant is gone. The same `hcSplitOperands` serves decode (compiled `collapseFields`) and prefill (op path plus shapeless split activations), thus the change covers both.

    Tests (TDD): new shared helper `DeepSeekV4TestArrays.gatherNodeCount(_:)` counts `"Gather"` nodes in `exportToDot`. New tests `theCompiledCollapseFieldsReadTheScalesWithNoGather`, `theCollapseFieldsOfAPrefillBlockReadTheScalesWithNoGather`, `theHeadReduceReadsItsScaleWithNoGather`. RED: the 3 tests failed on HEAD. GREEN: all pass. The decode collapse-field inputs moved into the shared `collapseInputs(tokens:)` with the same seeds, and the existing parity test reads them.

    Graph (PROF4_DOT, 32k): decode `Gather` 265 -> 6 (GatherAxis 64 and GatherQMM 129 unchanged). Last prefill chunk `Gather` 386 -> 132.

    Measurements at 32k, GPU idle each time:
    - bench `<snap> 32768 128 8192 1024`: before prefill 416.8 tok/s, decode 33.2 ms/token; after prefill 411.1 tok/s, decode 33.2 ms/token. Generated text and every cache-check line are the same before and after. Cache checks: HARD reuse PASS, tie-aware PASS for both thinking modes.
    - profile window (20 steps): before 35.2 and 35.1 ms/step; after 34.8 and 34.5 ms/step, thus about 0.5 ms/step less. Profile prefill 415.6 / 416.0 before, 415.4 / 415.8 after.
    The bench decode number does not move at its 0.1 ms resolution; the profile window shows about 0.5 ms/step, which is in the expected range. The change is kept: same bits, 259 fewer dispatches per step, and the head code is simpler.

    Note: the `swift build --build-tests` log has one line `warning: missing creator for mutated node: ... mlx-swift_Cmlx.bundle/Contents/MacOS`. It is a build-system message on HEAD too, not a compiler warning.
  timestamp: 2026-10-14T17:17:00.624825+00:00
- actor: claude-code
  id: 01m4xpnvf9ngt3ywna0jy2969f
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift, Tests/MLXLMTests/DeepSeekV4TestArrays.swift. Decode Gather 265 -> 6, prefill-chunk Gather 386 -> 132. 32k bench: decode 33.2 -> 33.2 ms/token, prefill 416.8 -> 411.1 tok/s; profile window 35.2/35.1 -> 34.8/34.5 ms/step. xctest: 716 XCTest 0 failures, 1424 swift-testing tests passed, 0 compiler warnings. Greedy parity greedyFirstTokensMatchThePythonFixture passed. Bench cache checks PASS.
    - next: review
  timestamp: 2026-10-14T17:17:05.897769+00:00
- actor: claude-code
  id: 01m4xptg8x6q0nqc3xwqmmma6f
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD; 3 findings (3 confirmed, 2 refuted) — Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift:31, Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift:268, Tests/MLXLMTests/DeepSeekV4TestArrays.swift:54. Acceptance criteria checked against the implement comments and ticked: (1) decode Gather 265 -> 6, and the new no-Gather tests for the compiled collapse fields and the head pass; (2) xctest 0 failures, the compiled decode parity test reads the same shared inputs and passes; (3) 32k decode before and after recorded with GPU idle (bench 33.2 -> 33.2 ms/token, profile window 35.2/35.1 -> 34.8/34.5 ms/step).
    - next: implement the 3 open findings
  timestamp: 2026-10-14T17:19:38.269037+00:00
- actor: claude-code
  id: 01m4xptzr9tpvq3x4qet004wpb
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 files; decode Gather 265 -> 6; 32k decode 33.2 -> 33.2 ms/token (profile window 35.2 -> 34.7 ms/step); greedy parity pass
    - test: green — 716 XCTest + 1424 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 1ae9cc9
    - review: findings — DeepSeekV4CompiledDecodeTests.swift:31, DeepSeekV4CompiledDecodeTests.swift:268, DeepSeekV4TestArrays.swift:54
  timestamp: 2026-10-14T17:19:54.121668+00:00
- actor: claude-code
  id: 01m4xq43c41esvppner7s932yn
  text: |-
    Review findings fixed (not committed).
    - Finding 1 (reuse): `decodeShape` is now `blockShape(decodeTokens)`. The same cause was also at `onlyABlockOfOneTokenReadsTheFunctionsCompiledWithAShape`, which built `[1, 2, hiddenSize]` by hand. It now reads `blockShape(decodeTokens + 1) + [hiddenSize]`. No other hand-built block shape stays in the file.
    - Findings 2 and 3 (positive control): `DeepSeekV4TestArrays` has a private control graph `MLXArray.zeros([controlVectorLength])[controlElementIndex]` (an `Int` subscript, thus `take`, thus one `Gather`). New shared check `expectCounterSeesTheControlGather(sourceLocation:)` asserts that `gatherNodeCount` of the control graph is exactly `controlGatherCount` (1). New shared check `expectNoGather(_:sourceLocation:)` runs the control check first, then asserts 0 for the given arrays. The three no-Gather tests (decode collapse, prefill collapse, head reduce) now call `expectNoGather`, so no assertion code is duplicated. New suite `Tests/MLXLMTests/DeepSeekV4TestArraysTests.swift` holds the dedicated control test `theGatherCounterFindsTheOneGatherOfTheControlGraph`.
    - RED (mutation): with the label changed to `"GatherBroken"`, the run of the 3 suites failed with 4 issues: the control test and the 3 no-Gather tests. Then the label went back to `"Gather"`.
    - GREEN: full `xcrun xctest`: 716 XCTest 0 failures, 1425 Swift Testing tests in 112 suites passed (one more than before: the new control test). Build: 0 compiler warnings (only the known `missing creator for mutated node` build-system line). GPU idle (0%) before each run.
  timestamp: 2026-10-14T17:24:52.740780+00:00
- actor: claude-code
  id: 01m4xq457511mgrdy1hzx7dwpf
  text: |-
    ### implement — changed
    - evidence: Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift, Tests/MLXLMTests/DeepSeekV4TestArrays.swift, Tests/MLXLMTests/DeepSeekV4TestArraysTests.swift (new). xctest: 716 XCTest 0 failures, 1425 Swift Testing passed, 0 compiler warnings. 3/3 findings checked.
    - next: review
  timestamp: 2026-10-14T17:24:54.629343+00:00
- actor: claude-code
  id: 01m4xq6hcnf0g43beb9m56f3fk
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (0e5b63c): 0 findings, 0 confirmed, 0 refuted; 4 files reviewed, 2 .kanban files not reviewed (no validator matched). All 3 findings of iteration 1 are checked.
    - next: task moved to done.
  timestamp: 2026-10-14T17:26:12.629886+00:00
- actor: claude-code
  id: 01m4xq6vyhsr4et9y9exqzndqk
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 test files, 3 findings fixed (Gather counter positive control)
    - test: green — 716 XCTest + 1425 Swift Testing pass, 0 warnings (run by the implement step on the same tree)
    - commit: 0e5b63c
    - review: clean — 0 findings
  timestamp: 2026-10-14T17:26:23.441436+00:00
position_column: done
position_ordinal: ffca80
title: 'DeepSeek-V4 decode speed: remove the 258 scale Gather dispatches in each step'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`, `prof4/dot_decode_counts.txt`).

`scaleF32[preScaleIndex]`, `scaleF32[postScaleIndex]` and `scaleF32[combScaleIndex]` in `DeepSeekV4MathHelpers.swift` (lines 144, 149, 155) each make a `Gather` node. A decode step has 265 `Gather` nodes; mlx-vlm has 25. Make the three scale values once at load, or use a slice.

Expected gain: about 0.3-0.6 ms/token.

## Acceptance Criteria

- [x] No `Gather` node for the scales in the decode graph.
- [x] The compiled decode parity tests (same bits) pass.
- [x] Decode ms/token at 32k, before and after, GPU idle. #deepseek-v4 #performance

## Review Findings (2026-10-14 12:17)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4XBX6Z9762N7NSHDY7RWQ7W.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4XBX6Z9762N7NSHDY7RWQ7W.md` — no validator matches this file

- [x] `Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift:31` `reuse/reuse` — The new `decodeShape` property rebuilds the shape `[1, decodeTokens]` by hand. The same file already has the helper `blockShape(_:)` at line 43, which returns `[1, tokens]`. `decodeShape` is `blockShape(decodeTokens)`, so it repeats the helper's logic instead of calling it. If the block shape layout changes, the two definitions can drift apart. Define `decodeShape` as `blockShape(decodeTokens)`. Note that `decodeShape` is a stored property and `blockShape` is a static function, so the order of the two declarations does not matter for static members, but check that `decodeTokens` is declared before its first use in a static initializer.
- [x] `Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift:268` `test-integrity/no-test-cheating` — This test's only assertion is that the gather count of the prefill collapse fields equals 0. The same vacuous-pass risk applies: a broken counter makes this test pass with no real check. Cover the counter with the same positive control as the decode test, so this zero-count assertion is shown to be able to fail.
- [x] `Tests/MLXLMTests/DeepSeekV4TestArrays.swift:54` `test-integrity/no-test-cheating` — The new gather counter matches the exact label "Gather" in the exportToDot output. If the label format changes, or the export produces no match, gatherNodeCount returns 0. The two new no-gather tests then pass with no check that the counter can see a gather. Those tests can never fail, so they prove nothing about the scale reads. Add a positive control test. Build a small graph that holds one known general Gather, for example an Int subscript on a vector, and assert that gatherNodeCount returns exactly 1. Keep the zero-count assertions. Then a broken counter fails the suite instead of passing every no-gather test.
