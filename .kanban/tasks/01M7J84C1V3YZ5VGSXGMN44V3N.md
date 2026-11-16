---
assignees:
- claude-code
depends_on:
- 01M7J847TB7CBFPZ2BQNSH02J2
position_column: todo
position_ordinal: 8a80
title: 'Qwen3.8 prefill: default step 2048'
---
## Why
Measured prefill at 2k: step 512 runs at 1326 tok/s and step 2048 at 1427 tok/s (+8–10%), for about 1.6 GB more peak memory. lm_head already does not run on prefill. Flint commit 6adfaa83 set FlintModel.prepare to step 2048, and commit 792cd657 made guided passes use the model's own prefill step.

## What
- Give Qwen35 (VLM and LLM) a default prefill step of 2048 in `prepare`, keeping any caller-specified step.
- Check that the hybrid checkpoint and prompt-cache boundaries still line up. GDN state is snapshotted at chunk boundaries.

## Acceptance
- [ ] `qwencheck.sh`: greedy ids identical in plain and MTP modes, and cache reuse passes, including checkpoint restore on follow-ups. bf16 output can depend on chunk boundaries; if ids change, record the near-tie evidence here.
- [ ] Peak memory at a 64k prefill is recorded. All five bundles are green. `qwenbench.sh` delta on this card. #qwen-perf