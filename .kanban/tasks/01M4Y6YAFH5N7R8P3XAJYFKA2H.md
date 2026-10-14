---
assignees:
- claude-code
position_column: todo
position_ordinal: '8e80'
title: 'DeepSeek-V4: let the model give a preferred prefill step (4096 measured +3.5 % at 32k)'
---
## What

Found on 2026-10-14 in ^ngjpzmq (osaurus idea 4). After the fused sparse attention kernel (^f4ztxgj), the attention cost per query row does not grow with the prefill block size. A larger prefill step thus gives the MoE a larger batch.

Measured on M3 Ultra, DeepSeek-V4-Flash-4bit, 32k bench (`<snap> 32768 128 8192 1024`), GPU idle:
- step 2048: 559.5 to 559.9 tok/s, peak 143 GiB, decode 30.4 ms/token.
- step 4096: 579.0 to 579.6 tok/s, peak 145 GiB, decode 30.3 to 30.4 ms/token.

Now the prefill step is only a caller parameter (`GenerateParameters.prefillStepSize`). The library default is 512 for all models (`PrefillParameters.defaultStepSize`), and `PrefillParameters.resolvedStepSize` / `forEachChunk` already take a `defaultStepSize` argument that `LLMModel.prepare` does not pass. A caller that sets no step gets 512 for DeepSeek-V4.

Decision for a person: should a model give its own preferred step (for example 4096 for DeepSeek-V4) that `LLMModel.prepare` uses when the caller sets none? Peak memory goes up by about 2 GiB at 32k.

## Acceptance Criteria

- [ ] A unit test shows that a DeepSeek-V4 prepare with no caller step uses the model step, and that a caller step still wins.
- [ ] Prefill tok/s, peak memory and decode ms/token at 32k with no caller step, before and after, GPU idle.
- [ ] The greedy parity test on the real checkpoint passes. #deepseek-v4 #performance