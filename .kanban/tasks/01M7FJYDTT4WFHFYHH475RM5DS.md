---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7fnczcb9c52tcab8bv7xdz0
  text: 'Research: loadModelCheckpoint returns evaluated tensors; prepareCheckpoint (Flint fuse + sanitize stacking) builds lazy concatenations that keep the parts referenced through the graph until materializeModelForInference evaluates the whole model in one eval, so every fused buffer coexists with every part. Fix plan: evaluate the prepared checkpoint weights in byte-bounded groups with Memory.clearCache() between groups, right after prepareCheckpoint returns (the loader''s own pre-prepare checkpoint value is replaced by then, so evaluated concats detach and the parts free). Baseline at HEAD 2a0fd237 (/usr/bin/time -l flintcli <dir> chat 200 hello): oQ8e peak 70.17 GB, load 4.1 s; oQ4e peak 38.65 GB, load 2.2 s. MLX''s isEvaluated is internal to mlx-swift, so the helper evaluates all prepared tensors (evaluating an already-evaluated array is a no-op).'
  timestamp: 2026-11-15T13:12:29.835698+00:00
- actor: claude-code
  id: 01m7fp2sz8x45805yyfxa9wvah
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLMCommon/ModelCheckpoint.swift (new `ModelCheckpoint.materializeWeights(groupByteLimit:)`, default `materializationGroupBytes` = 1 GiB: evaluates prepared tensors in byte-bounded groups, Memory.clearCache() after each), Libraries/MLXLMCommon/Load.swift (installWeights calls it right after prepareCheckpoint; loadWeights doc), Libraries/MLXLMCommon/ModelComponent.swift (the on-demand component load calls it too, so both loaders behave alike), Tests/MLXLMTests/LoadWeightsTests.swift (testMaterializingAPreparedCheckpointFreesEachGroupOfReplacedParts: RED with a single-eval stub, peak growth 268435456 not < 134217728; GREEN after grouping).
    - Peak footprint (/usr/bin/time -l flintcli <dir> chat 200 hello): oQ8e 70.17 GB -> 44.27 GB (unfused reference 42.7); oQ4e 38.65 GB -> 24.92 GB. Load: oQ8e 4.1 s -> 3.8 s, oQ4e 2.2 s -> 2.3 s (one sample each, within noise).
    - Tests: MLXLMTests 1090 XCTest (4 LFM2 env skips) + 2331 Swift Testing passed, 3 known issues (baseline); MLXGuidedGenerationTests 76 passed; MLXFoundationModelsTests 494 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 passed. Zero failures, zero warnings beyond the pre-existing build-system "missing creator" line.
    - flintcheck.sh: MATCH x4 (all 256 greedy ids identical on parity/long/prompt_87/prompt_706), TOOL CALL get_weather Paris. golden_base.json untouched; output in scratchpad/golden_75rm5ds.json.
    - flintbench.sh: decode 7.10/7.50/7.81 ms/tok at 2k/8k/16k (prefill 6251/6158/5746 tok/s), unchanged.
    - Note: mlx-swift's isEvaluated is internal, so the helper evaluates every prepared tensor; already-evaluated ones are no-ops.
    - next: /review
  timestamp: 2026-11-15T13:24:25.192108+00:00
- actor: claude-code
  id: 01m7fpjg49hdejgrzmk5gq5shh
  text: |-
    ### test — green
    - evidence: swift build --build-tests complete (no code warnings). xcrun xctest, one bundle at a time: MLXLMTests 1090 XCTest (4 skipped = LFM2 env baseline, 0 failures) + Swift Testing 2331 tests in 221 suites passed, 3 known issues (baseline); MLXGuidedGenerationTests 76 passed; MLXFoundationModelsTests 494 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed, 0 failures. swift-format lint -s clean on Load.swift, ModelCheckpoint.swift, ModelComponent.swift, LoadWeightsTests.swift. flintcheck.sh: MATCH x4 (256 greedy ids identical each) plus TOOL CALL get_weather line; golden_base.json untouched.
    - next: review / commit
  timestamp: 2026-11-15T13:32:59.401054+00:00
- actor: claude-code
  id: 01m7fpkfsg34yxz4hqwyrpae6a
  text: |-
    ### commit — changed
    - evidence: 8ab425ee perf(load): free the tensors a checkpoint preparation replaces before the next group evaluates (^75rm5ds); files: Libraries/MLXLMCommon/Load.swift, ModelCheckpoint.swift, ModelComponent.swift, Tests/MLXLMTests/LoadWeightsTests.swift, .kanban/tasks (01M7FHFQT76AN9817CJJ4PQTWR, 01M7FJYDTT4WFHFYHH475RM5DS)
    - next: none (local commit only, not pushed)
  timestamp: 2026-11-15T13:33:31.824562+00:00
position_column: doing
position_ordinal: '80'
title: 'Loader: release checkpoint tensors that prepareCheckpoint replaces before materialization (Flint fusion load peak)'
---
## Why
^j4pqtwr fuses Flint q/k/v and gate/up projections in `FlintModel.prepareCheckpoint`. The concatenated tensors are new buffers, and the parts stay alive until materialization: `installWeights` (Libraries/MLXLMCommon/Load.swift) keeps its own `checkpoint` value while `prepareCheckpoint` runs, and the freed parts then sit in the MLX buffer cache until `materializeModelForInference` calls `Memory.clearCache()`.

Measured with `/usr/bin/time -l flintcli <dir> chat 200 hello` on the M5: oQ8e peak memory footprint 42.7 GB before fusion, 70.2 GB after (weights 39 GB); oQ4e 38.6 GB after (weights 22 GB). Steady-state RSS is unchanged. The allocator releases its cache near the GPU limit, but the transient peak is close to the 77.8 GiB working set on the M5 Ultra for oQ8e.

## What
- Let a model's checkpoint preparation free the tensors it replaces before the new ones evaluate (for example, hand `prepareCheckpoint` the only reference, or evaluate replaced tensors in bounded groups with a cache clear between them).
- Keep `ModelCheckpoint` semantics for other models.

## Acceptance
- oQ8e load peak footprint back near the unfused 42.7 GB (measure the same way).
- Flint greedy ids unchanged (flintcheck.sh MATCH on all four prompts).
- All five test bundles green.
#flint-perf