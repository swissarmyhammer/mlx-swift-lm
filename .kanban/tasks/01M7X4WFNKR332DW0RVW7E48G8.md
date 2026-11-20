---
assignees:
- claude-code
position_column: done
position_ordinal: ffffcc80
title: 'Qwen 3.5: limit the prefill step below the MLX dsplit kernel'
---
Request from FoundationModelsACPAgent (^xgqst2y there). On an Apple M5 Max, Qwen3.5-9B-4bit stops at the first long prompt: "Unable to build metal library from source", kernel steel_attention_dsplit_bfloat16_..._bd256. Cause: upstream issue ml-explore/mlx 4832 (fix PR 4903 is open). MLX selects dsplit with NAX, head dimension 256, causal, 1024+ query tokens. Qwen3.5-9B (16/4 heads) is not covered by PrefillAttentionKernel, thus it goes to MLXFast SDPA with steps of 2048.

Fix: Qwen35PrefillPolicy.maximumStepSize gives 1023 when the head dimension is 256 and the heads do not make whole head groups. The LLM text model, the wrapped model and the VLM give it as maximumPrefillStepSize. Qwen3.8-27B keeps 2048.

Acceptance: Qwen35PrefillStepTests cover the policy and the three models; the five bundles pass; send the sha on stable to foundationmodelsacpagent-9e.