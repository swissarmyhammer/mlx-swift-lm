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