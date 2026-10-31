---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'MTP router path: compile the MLXVLM Qwen35 verify forward (2.0–2.4 ms of host graph build per round while the GPU idles; the text path''s compiled verify builds in 1.0 ms)'
---
From ^1abqd92's round profile (Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, bs4 pMin0.75, after the buffer-pool and device-side causal-mask fixes): on the router load path (MLXVLM `Qwen35`, VLM drafter) the verify forward still takes the general per-layer path, and building its graph costs 2.03 ms per round at 32k and 2.44 ms at 100k, against 1.02-1.03 ms for the text path's compiled verify schedule (`Qwen35TextModelInner.verifyStep` + `KeyedCompiledDecodeSegmentCaches`). The GPU is idle during that build: the drafter chain's last confidence read has just drained it. That is the largest host-only piece left in a router round (36.6 ms at 32k: pre 0.19, chain build 0.39, verify build 2.03, commit 0.40, book 0.18).

- [ ] Give MLXVLM `Qwen35Language.Model` the compiled verify schedule the MLXLLM text model has (M-RoPE position ids stay outside the trace with the attention step), or show why it cannot be shared; coordinate with ^8p0rfxx (router verify mask as `.causal`).
- [ ] Verify output within `AttentionKernelTolerance` of the general path for S = 2..6; greedy MTP equals plain on the VLM fixture.
- [ ] Re-measure the router sweep and MTPCPULoadTests (verify build per round, tok/s at 32k/100k).

#mtp #perf