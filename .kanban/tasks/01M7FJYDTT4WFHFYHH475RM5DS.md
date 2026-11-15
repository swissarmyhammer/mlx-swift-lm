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
- actor: claude-code
  id: 01m7fpnr6m3xf8zz5m2ykcs767
  text: |-
    ### review — findings
    - evidence: review sha 8ab425ee~1..HEAD — 1 finding (1 confirmed, 0 refuted): Libraries/MLXLMCommon/ModelCheckpoint.swift:95
    - next: implement the Review Findings item (replace `[MLXArray]()` with `: [MLXArray] = []`, and every other such empty-collection init in the file), then re-review
  timestamp: 2026-11-15T13:34:45.972629+00:00
- actor: claude-code
  id: 01m7fpp2g6tkvtm9navq37kybj
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — ModelCheckpoint.swift, Load.swift, ModelComponent.swift, LoadWeightsTests.swift (+1 test)
    - test: green — MLXLMTests 1090 XCTest + 2331 ST, Guided 76, FM 494, CXGrammar 7, Macros 11; flintcheck MATCH x4 + TOOL CALL
    - commit: 8ab425ee (+47b06e2c kanban)
    - review: findings — Libraries/MLXLMCommon/ModelCheckpoint.swift:95 (AlwaysUseLiteralForEmptyCollectionInit)
  timestamp: 2026-11-15T13:34:56.518786+00:00
- actor: claude-code
  id: 01m7fq6p9jkmxka7bttb8ja8x6
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXLMCommon/ModelCheckpoint.swift — every empty collection written as an initializer call in the file now uses a typed literal: `var group: [MLXArray] = []` (materializeWeights, the finding), `var sources: [String: String] = [:]` and `var settings: [String: BaseConfiguration.QuantizationOption] = [:]` (mapNames). Load.swift, ModelComponent.swift, LoadWeightsTests.swift: the lines this task added contain no empty-collection initializer call, so no edit. swift-format: only ModelCheckpoint.swift changed. swift build --build-tests complete (only the pre-existing "missing creator" build-system line). xctest one bundle at a time: MLXLMTests 1090 XCTest (4 skipped = LFM2 env baseline, 0 failures) + 2331 Swift Testing passed, 3 known issues (baseline); MLXGuidedGenerationTests 76; MLXFoundationModelsTests 494; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 — 0 failures. flintcheck.sh: MATCH x4 (256 ids each) + TOOL CALL get_weather Paris; golden_base.json md5 unchanged (3630e415…).
    - Review finding item flipped to [x]. Not committed; task left in doing.
    - next: /test, /commit, /review
  timestamp: 2026-11-15T13:44:01.074870+00:00
- actor: claude-code
  id: 01m7fqn476dn91th8xwwgfeycr
  text: |-
    ### test — green
    - evidence: swift build --build-tests complete (only a build-system "missing creator for mutated node" notice, no compiler warnings); xcrun xctest per bundle, all exit 0: MLXLMTests 2331 Swift Testing tests in 221 suites passed with 3 known issues, XCTest 1090 executed with 4 skipped (LFM2 env skips) and 0 failures; MLXGuidedGenerationTests 76 passed; MLXFoundationModelsTests 494 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed, 0 failures. swift-format lint -s on Libraries/MLXLMCommon/ModelCheckpoint.swift clean. flintcheck.sh: MATCH x4 (all 256 greedy ids identical) and TOOL CALL get_weather line present.
    - next: review/commit
  timestamp: 2026-11-15T13:51:54.086499+00:00
- actor: claude-code
  id: 01m7fqp7stvdyaj9s9j06xnfrv
  text: |-
    ### commit — changed
    - evidence: commit "refactor(load): write the empty collections of ModelCheckpoint as literals, found in review of ^75rm5ds" (this comment ships in that same commit, so the sha is in git log by that subject); files: Libraries/MLXLMCommon/ModelCheckpoint.swift, .kanban/tasks/01M7FJYDTT4WFHFYHH475RM5DS.jsonl, .kanban/tasks/01M7FJYDTT4WFHFYHH475RM5DS.md
    - next: none
  timestamp: 2026-11-15T13:52:30.522548+00:00
- actor: claude-code
  id: 01m7fqqq2ptwdw8p3dn5pxn7jk
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (6b450a82): 0 findings (7 validator passes attempted, 0 failed); 1 file reviewed (Libraries/MLXLMCommon/ModelCheckpoint.swift), 2 .kanban files had no matching validator. The prior finding (2026-11-15 07:33) is checked.
    - next: none — task moved to done
  timestamp: 2026-11-15T13:53:18.934403+00:00
- actor: claude-code
  id: 01m7fqrc6jvtz0xye49xpmfana
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — ModelCheckpoint.swift empty-collection literals (1/1 finding)
    - test: green — MLXLMTests 1090 XCTest + 2331 ST, Guided 76, FM 494, CXGrammar 7, Macros 11; flintcheck MATCH x4 + TOOL CALL
    - commit: 6b450a82
    - review: clean — 0 findings; task done
  timestamp: 2026-11-15T13:53:40.562865+00:00
position_column: done
position_ordinal: ffffa580
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

## Review Findings (2026-11-15 07:33)

> Scope: `review sha 8ab425ee~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7FHFQT76AN9817CJJ4PQTWR.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FHFQT76AN9817CJJ4PQTWR.md` — no validator matches this file
> - `.kanban/tasks/01M7FJYDTT4WFHFYHH475RM5DS.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7FJYDTT4WFHFYHH475RM5DS.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/ModelCheckpoint.swift:95` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[MLXArray]()' with ': [MLXArray] = []'.
