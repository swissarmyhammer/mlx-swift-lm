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
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 decode speed: remove the 258 scale Gather dispatches in each step'
---
## What

Found on 2026-10-14 (profile at 32k, scratchpad `prof4/REPORT.md`, `prof4/dot_decode_counts.txt`).

`scaleF32[preScaleIndex]`, `scaleF32[postScaleIndex]` and `scaleF32[combScaleIndex]` in `DeepSeekV4MathHelpers.swift` (lines 144, 149, 155) each make a `Gather` node. A decode step has 265 `Gather` nodes; mlx-vlm has 25. Make the three scale values once at load, or use a slice.

Expected gain: about 0.3-0.6 ms/token.

## Acceptance Criteria

- [ ] No `Gather` node for the scales in the decode graph.
- [ ] The compiled decode parity tests (same bits) pass.
- [ ] Decode ms/token at 32k, before and after, GPU idle. #deepseek-v4 #performance