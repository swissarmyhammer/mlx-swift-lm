---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: Move the Qwen35 GDN input-projection fusion onto FusedQuantizedProjectionGroup
---
## Why
^8h1td1f added `FusedQuantizedProjectionGroup` (Libraries/MLXLMCommon/FusedQuantizedLinear.swift): a fused projection plus its view installation, update/updateModule invalidation and compiled-trace registration. The GDN input-projection fusion in both twins (`Qwen35Language.GatedDeltaNet` in Libraries/MLXVLM/Models/Qwen35.swift, `Qwen35GatedDeltaNet` in Libraries/MLXLLM/Models/Qwen35.swift) still carries its own copy of that lifecycle (lock + weak trace registry in the VLM, none in the LLM, hand-written update/updateModule overrides, hand-written split offsets).

## What
- Rebuild both GDN fusions on `FusedQuantizedProjectionGroup(keys: ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"])`, keeping the `MLX_QWEN_FOUR_GDN` rollback switch and `fusedInputProjectionEnabled`.
- The LLM GDN then also registers its traces, as the VLM one does.
- Keep `Qwen35FusedGDNProjectionTests` green unchanged.

## Acceptance
- [ ] No duplicated fusion lifecycle code remains in either twin.
- [ ] `Qwen35FusedGDNProjectionTests` and `Qwen35FusedProjectionTests` pass (xctest). #qwen-perf