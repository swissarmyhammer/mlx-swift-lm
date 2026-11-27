---
position_column: todo
position_ordinal: '8e80'
title: 'Clustered models: support a quantized KV cache (kvBits 8/4) on the workers'
---
## What
ClusterLanguageModel.newCache (Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift:102) throws unsupported(.cacheConfiguration) for any KV-cache configuration. Our best Qwen profile (q8kv8: Qwen3.8-27B-8bit, kvBits 8) failed on its first call in the cluster1q SWE-bench run (2026-11-27).

## Fix
Carry the KV-cache quantization (bits, group size) in the newCache op to each rank, so each worker makes a quantized cache for its heads, as the local path does (KV8 decode ^2h5wtct). Keep maxKVSize refused if it is not needed. Protocol version bump if the op changes.
Tests: the 2-Mac parity suite with kvBits 8 on Qwen3.8-27B matches the local greedy output under the same top-5 rule; a unit test that newCache with kvBits sends the config.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster