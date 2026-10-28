---
assignees:
- claude-code
position_column: todo
position_ordinal: '9780'
title: Route Qwen3Next and VLM Qwen35 decodeStep through a shared schedule runner, drop their cache force unwraps
---
## What
Found while doing ^5fq4w2k. `Qwen3NextModelInner.decodeStep` (Libraries/MLXLLM/Models/Qwen3Next.swift) and `Qwen35Language.Model.decodeStep` (Libraries/MLXVLM/Models/Qwen35.swift) each re-implement the schedule loop that `Qwen35TextModelInner.runSchedule` / `scheduleCaches` / `linearCaches` / `commit` already hold in Libraries/MLXLLM/Models/Qwen35.swift: gather `[carry] + pendingAttention + states`, run the segment, write the states back, run the KV write between segments.

The copies still carry `mambaCaches[layerIndex]!`, `mambaCache[0]!`, `mambaCache[1]!`, `cache[pre]!`, and index the outputs with literals (`1 + 2 * stateIndex`, `outputIndex + 2`) instead of `CompiledDecodeSegment.statesPerLinearLayer` / `CompiledDecodeSegment.AttentionOutput`. The VLM `DecoderLayer.decodeLinearLayer` also force-unwraps `cache[0]!, cache[1]!`.

- [ ] Move the schedule runner (cache gathering, segment loop, state commit) into MLXLMCommon next to `CompiledDecodeSegment`, parameterized by the attention cache step of each model (Qwen3Next: rope + `attentionWithCacheUpdate`; VLM: M-RoPE `attend` with `positionIds`; LLM Qwen35: `attentionCacheStep(mask:)`).
- [ ] Route all three models through it; delete the per-model loops.
- [ ] No force unwraps left in the three decode paths.

## Acceptance Criteria
- [ ] Plain decode logits bit-identical before/after (Qwen3NextCompiledDecodeTests, Qwen35VLMCompiledDecodeTests, Qwen35 MTP tests).
- [ ] Release `qwen38FourBitMTPGeneration` within 3% of the same-session baseline; `qwen38FourBitMTPSystemPromptCache` passes with no known issues.

## Tests
- [ ] `swift build --build-tests` zero warnings; all five bundles green via `xcrun xctest`.

## Workflow
- Use `/tdd`. #mtp #perf