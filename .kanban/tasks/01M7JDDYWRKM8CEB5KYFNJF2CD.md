---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7k2v0m2detamxzme6dpn7x1
  text: 'Research: FusedQuantizedProjectionGroup.callAsFunction uses the fused matmul only for < fewestMatrixRows (6) input rows; both GDN twins today use the fused matmul for every call, prefill included, and check `fusedInputProjectionEnabled` at call time. To keep output and prefill performance unchanged the group needs (a) a row policy that fuses every call and (b) an enable switch that both `prepare` and the call honour. Plan: add `FusedRows { vectorCalls, everyCall }` (default vectorCalls, so attention/MLP groups are untouched) and `isEnabled` to the group; GDN `fusedInputProjectionEnabled` forwards to it. Qwen35FusedGDNProjectionTests needs `registerFusedProjectionTrace(_:)` on the VLM GDN, so it stays as a thin forward to the group''s trace registry. LLM decoder layer then passes `trace` to the GDN as the VLM one does. Qwen3Next GDN has no input fusion; out of scope.'
  timestamp: 2026-11-16T21:05:04.642960+00:00
- actor: claude-code
  id: 01m7k4gnk8zfqjjtstbetkeefp
  text: |-
    Implementation landed. Both GDN twins now use FusedQuantizedProjectionGroup(keys: in_proj_qkv/z/b/a, fusedRows: .everyCall, isEnabled: qwen35FourGDNEnabled). `fusedInputProjectionEnabled` forwards to `group.isEnabled`. The update/updateModule overrides wrap `invalidating(afterUpdating:)` and `invalidating(afterReplacing:)`. The VLM lock and weak trace registry are gone. The LLM decoder layer passes `trace` to the GDN, so the LLM GDN now registers its traces. The group gained `FusedRows { vectorCalls (default), everyCall }` and `isEnabled`. Attention and MLP groups keep the defaults, so their behaviour is unchanged. VLM `registerFusedProjectionTrace(_:)` stays as a one-line forward because Qwen35FusedGDNProjectionTests (which must stay unchanged) calls it.
    TDD: the new Qwen35GDNInputProjectionGroupTests fails on the old behaviour. With the LLM GDN not registering, `testLLMGatedDeltaNetParameterUpdateDropsTheCompiledDecode` reports segment count 2 != 0. With the change it passes.
    Golden: qwencheck matches qwen_golden_mn44v3n.json id for id (all 8 label/mode pairs, 257 ids each). qwencheck still prints CHECK FAIL only because its built-in compare uses qwen_golden_base.json, whose 32k entries ^mn44v3n changed. Session and toolchat both PASS.
    Bench (qwen_bench_fnjf2cd.txt vs mn44v3n): plain 17.13/19.85/22.78 ms, MTP 9.56/12.51/13.58 ms, prefill 1464/1270/796 tok/s. MTP 2k came out +1.1%, so I ran a same-session A/B. HEAD (stashed) gave 9.53/9.52/9.46 ms; the change gave 9.61/9.55/9.49 ms. That is within run-to-run noise.
  timestamp: 2026-11-16T21:34:22.824544+00:00
- actor: claude-code
  id: 01m7k4grm8t5y2cem4dpccsr01
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/FusedQuantizedLinear.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift (new). Ran swift build --build-tests (no warnings), then xctest on all five bundles: MLXLMTests 1123 XCTest + 2369 Swift Testing, MLXGuidedGenerationTests 78, MLXFoundationModelsTests 496, CXGrammarTests 7 and MLXHuggingFaceMacrosTests 11, with 0 failures. MLXLMTests skipped 4 LFM2 checkpoint/benchmark tests, which an environment variable gates and which existed before this change. It also recorded 3 withKnownIssue expectations. Golden output is identical to qwen_golden_mn44v3n.json. Decode did not regress.
    - next: /review
  timestamp: 2026-11-16T21:34:25.928745+00:00
- actor: claude-code
  id: 01m7k4p3aqgr9r0ca1q0dqagbn
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD — 8 findings (8 confirmed, 2 refuted): Libraries/MLXLLM/Models/Qwen35.swift:201, :209, :214, :310, :323; Libraries/MLXVLM/Models/Qwen35.swift:1040, :1048, :1179
    - next: implement the 8 items of Review Findings (2026-11-16 15:35); task stays in review
  timestamp: 2026-11-16T21:37:20.727750+00:00
- actor: claude-code
  id: 01m7k4phnbbgnb7ff0abs559hn
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 4 files; golden identical to qwen_golden_mn44v3n.json; bench unchanged
    - test: green — 5 bundles 0 failures (run by implement)
    - commit: 22915621
    - review: findings — MLXLLM/Models/Qwen35.swift:201, :209, :214, :310, :323; MLXVLM/Models/Qwen35.swift:1040, :1048, :1179
  timestamp: 2026-11-16T21:37:35.403203+00:00
- actor: claude-code
  id: 01m7k53jq3vaqb0m3q399j6xdk
  text: 'Findings pass 1 (2026-11-16 15:35 section). Invalidation: `FusedQuantizedProjectionGroup.isEnabled` now has a `didSet` that calls `isEnabledDidChange(from:)`, which drops every reading trace when the value changes. It keeps the preparation, so a re-enable needs no new prepare (Qwen35FusedGDNProjectionTests flips the switch off then on around prepare). `invalidate()` now does projection.invalidate() + the shared `dropReadingTraces()`. TDD: testLLMGatedDeltaNetRollbackSwitchDropsTheCompiledDecode and testVLMGatedDeltaNetRollbackSwitchDropsTheCompiledDecode both failed first with segment count 2 != 0, then passed. Duplication: new `package enum Qwen35GatedDeltaInputProjection` in FusedQuantizedLinear.swift (next to the group) holds the one order (private `Part: Int, CaseIterable` with `key`), `keys`, `makeGroup()`, `ordered(qkv:z:b:a:)` and `project(_:by:_:batch:sequence:heads:headWidth:)` (split + z reshape). Both twins use it; their InputProjectionPart enums and key literals are gone. New unit test testGatedDeltaProjectionSplitsZIntoHeads failed first (missing symbol), then passed.'
  timestamp: 2026-11-16T21:44:42.467079+00:00
- actor: claude-code
  id: 01m7k5cc7nbxsdbzjv672em0bw
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/FusedQuantizedLinear.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift. All 8 findings of Review Findings (2026-11-16 15:35) are checked. swift-format ran. swift build --build-tests finished with no compiler warnings. xcrun xctest MLXLMTests: 1126 XCTest tests, 0 failures, 4 skipped (the env-gated LFM2 checkpoint/benchmark tests, which existed before); Swift Testing ran 2369 tests and they passed, with 3 known issues that predate this change. Qwen35FusedGDNProjectionTests and Qwen35FusedProjectionTests are unchanged and green. qwencheck.sh (compares against qwen_golden_mn44v3n.json) printed CHECK PASS: golden ids identical, SESSION PASS, TOOLCHAT plain/mtp PASS. The other four bundles were not run in this pass.
    - next: /review
  timestamp: 2026-11-16T21:49:30.741205+00:00
- actor: claude-code
  id: 01m7k5h92nqfg43pa677zbn09c
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (5cfae684): 1 finding (1 confirmed, 2 refuted) — Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift:195 (completeness/inverse-operation-coverage). All 8 prior findings checked.
    - next: implement the 2026-11-16 15:49 finding (add the off-to-on fusedInputProjectionEnabled flip case asserting compiledDecodeSegmentCount == 0), then re-review.
  timestamp: 2026-11-16T21:52:11.349137+00:00
- actor: claude-code
  id: 01m7k5hkd83bjeb11f16dgykt1
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — shared Qwen35GatedDeltaInputProjection, isEnabled didSet invalidation + tests; qwencheck PASS
    - test: green — MLXLMTests 1126 XCTest 0 fail + 2369 Swift Testing (run by implement)
    - commit: 5cfae684
    - review: findings — Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift:195
  timestamp: 2026-11-16T21:52:21.928826+00:00
- actor: claude-code
  id: 01m7k5nmd8g8hzq97syt79qw7t
  text: 'Findings pass 2 (2026-11-16 15:49 section). Added testLLMGatedDeltaNetRollbackSwitchReenableDropsTheCompiledDecode and testVLMGatedDeltaNetRollbackSwitchReenableDropsTheCompiledDecode. Each one disables the fusion, runs a compiled decode, sets `fusedInputProjectionEnabled = true`, and asserts `compiledDecodeSegmentCount == 0`. decodedLLMModel and decodedVLMModel now take `fusionEnabled: Bool = true` and set the switch before the decode, so all four switch tests use one helper. A new `gatedDeltaNet(of: Qwen35Language.Model)` lookup replaces the inline lookup in the VLM on→off test, whose assertions are unchanged. Production already invalidates on both flips, so the new tests pass on HEAD. Mutation check: with the guard of `isEnabledDidChange` changed to `guard isEnabled != oldValue, !isEnabled` (on→off only), both new tests failed with `("2") is not equal to ("0")` and the on→off tests still passed. The mutation was then reverted. Note: `traceState(of:)` registers a trace even while the fusion is off, which is why the off→on flip can drop it.'
  timestamp: 2026-11-16T21:54:34.024389+00:00
- actor: claude-code
  id: 01m7k5nphd6q3gxyzyz153k2hg
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift. swift-format ran. swift build --build-tests: Build complete, no compiler warnings. xcrun xctest -XCTest MLXLMTests.Qwen35GDNInputProjectionGroupTests: 8 tests, 0 failures. Mutation RED: 2 new tests failed (2 != 0) on an on→off-only guard, then the guard was reverted. The finding of 2026-11-16 15:49 is checked.
    - next: /review
  timestamp: 2026-11-16T21:54:36.205680+00:00
position_column: doing
position_ordinal: '80'
title: Move the Qwen35 GDN input-projection fusion onto FusedQuantizedProjectionGroup
---
## Why
^8h1td1f added `FusedQuantizedProjectionGroup` (Libraries/MLXLMCommon/FusedQuantizedLinear.swift): a fused projection plus its view installation, update/updateModule invalidation and compiled-trace registration. The GDN input-projection fusion in both twins (`Qwen35Language.GatedDeltaNet` in Libraries/MLXVLM/Models/Qwen35.swift, `Qwen35GatedDeltaNet` in Libraries/MLXLLM/Models/Qwen35.swift) still carries its own copy of that lifecycle (lock + weak trace registry in the VLM, none in the LLM, hand-written update/updateModule overrides, hand-written split offsets).

## What
- Rebuild both GDN fusions on `FusedQuantizedProjectionGroup(keys: ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"])`, keeping the `MLX_QWEN_FOUR_GDN` rollback switch and `fusedInputProjectionEnabled`.
- The LLM GDN then also registers its traces, as the VLM one does.
- Keep `Qwen35FusedGDNProjectionTests` green unchanged.

## Acceptance
- [x] No duplicated fusion lifecycle code remains in either twin.
- [x] `Qwen35FusedGDNProjectionTests` and `Qwen35FusedProjectionTests` pass (xctest). #qwen-perf

## Review Findings (2026-11-16 15:35)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 8 not reviewed.

> 8 file(s) not reviewed — no validator matched:
> - `.kanban/tags/01M7K2PZ36BK105354QCTZG6AB.jsonl` — no validator matches this file
> - `.kanban/tags/01M7K2PZ36BK105354QCTZG6AB.yaml` — no validator matches this file
> - `.kanban/tasks/01M7J84FZ3SJF8MB599CDEN2RC.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7J84FZ3SJF8MB599CDEN2RC.md` — no validator matches this file
> - `.kanban/tasks/01M7JDDYWRKM8CEB5KYFNJF2CD.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7JDDYWRKM8CEB5KYFNJF2CD.md` — no validator matches this file
> - `.kanban/tasks/01M7K2NYGV3ANPCQGHS6XCB05K.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7K2NYGV3ANPCQGHS6XCB05K.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Models/Qwen35.swift:201` `duplication/duplication` — The same four checkpoint keys in the same order appear in the LLM and VLM `fusedInputProjection` declarations. The fused row order is coupled to `InputProjectionPart` and to `inputProjections`, so three copies of the order must agree. Give the four-key order one definition, for example a static on the shared group type or a shared constant, and reference it from both twins.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:209` `completeness/invariant-propagation` — The `fusedInputProjectionEnabled` setter turns the input-projection fusion off or on without dropping the compiled traces that read it. A trace captured while the fusion was on keeps running the fused matmul after the switch. Every other path that changes the fused projection (`update` and `updateModule`) goes through `fusedInputProjection.invalidating(...)`, so the setter is the one site that skips the invalidation. Route the setter through the same lifecycle as the parameter updates, e.g. a group method that sets `isEnabled` and drops the reading traces when the value changes. Alternatively, make `isEnabled` read-only after init and document that toggling needs a new model. Add a test that toggles the setter after a compiled decode and checks `compiledDecodeSegmentCount` drops to 0, as the update test does.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:214` `duplication/duplication` — The new `InputProjectionPart` enum copies the VLM file's existing `InputProjectionPart` (same name, same four cases, same doc comments, same indices). The change adds a second copy of the index map instead of sharing one. Declare the part indices once in MLXLMCommon next to `FusedQuantizedProjectionGroup`, and have both GDN twins use that declaration. Then delete the two local enums.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:310` `duplication/duplication` — `inputProjections` repeats the VLM twin's list of the four projections in the same order. That order must match the `keys` list and `InputProjectionPart`, so it is a third copy of an ordering that has to agree. Derive the projection order from the shared group, or from one shared list of the four `Linear` keys and modules, and have both twins use it instead of writing the list twice.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:323` `duplication/duplication` — `projectInputs` repeats the same logic as the VLM twin: one fused projection call, then the split into qkv, z reshaped to heads, b and a by the same index constants. The four-part split and the z reshape are real logic that can drift between the two copies, so a fix to one twin can miss the other. Move the split into one shared helper, for example a method on `FusedQuantizedProjectionGroup` or a free function in MLXLMCommon that takes the fused output and the batch, sequence, head count and head width, and returns the four parts. Both GDN twins then call it and keep only their own `inputProjections` list.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:1040` `duplication/duplication` — The VLM `fusedInputProjection` declaration repeats the LLM twin's key list and fusion configuration, so the checkpoint key order is written twice. Use the shared key order proposed for the LLM declaration and delete the local literal.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:1048` `completeness/invariant-propagation` — The VLM twin has the same gap: `fusedInputProjectionEnabled`'s setter changes the fusion mode without invalidating the compiled traces that captured the fused path, while the `update` and `updateModule` overrides invalidate. Apply the same fix as the LLM twin in one shared place (the group, not each model), so both models inherit the invalidation. Add a VLM test that toggles the setter after a compiled decode and checks the trace is dropped.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:1179` `duplication/duplication` — `inputProjections` in the VLM GatedDeltaNet repeats the LLM twin's ordered list. The same ordering is coupled to `keys` and `InputProjectionPart` in two places. Use the shared projection order proposed for the LLM copy and delete the local list.

## Review Findings (2026-11-16 15:49)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7JDDYWRKM8CEB5KYFNJF2CD.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7JDDYWRKM8CEB5KYFNJF2CD.md` — no validator matches this file

- [x] `Tests/MLXLMTests/Qwen35GDNInputProjectionGroupTests.swift:195` `completeness/inverse-operation-coverage` — The rollback-switch test only flips `fusedInputProjectionEnabled` from on to off. The change now drops reading traces on every flip (`isEnabledDidChange` guards only on `isEnabled != oldValue`), and the stated reason is that a trace captured under the old setting keeps computing the outputs the old way. A trace captured while the fusion is off and then re-enabled is never tested, so the off-to-on direction is unproven. Add a case that decodes with the fusion disabled (or flips it off and decodes), then sets `fusedInputProjectionEnabled = true` and asserts `compiledDecodeSegmentCount == 0`. Keep the existing off-direction assertion.
