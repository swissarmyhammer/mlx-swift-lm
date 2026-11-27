---
position_column: todo
position_ordinal: '8f80'
title: 'Clustered models: the prompt cache store drops a RemoteKVCache at every check-in, so each turn re-prefills'
---
## What
In the cluster2i SWE-bench run (2026-11-27, Inkling 2 Mini over m5u1+m5u2), every generation_call reported tokens.cached = 0. Each turn logs (subsystem com.apple.FoundationModels-MLX, category ExecutorPromptCache):
```
prompt cache commit … Inkling-2-Mini-4bit#cluster=908feca3833d627d ledger=54401
prompt cache evict  … bytes=1938063360
prompt cache spill  … result=failed: KVCacheError(message: "RemoteKVCache has no serializedClassName, so it cannot be saved")
prompt cache plan   … source=none rendered=54772 reused=0 rule=cold
```
ExecutorPromptCacheStore.keepInMemory measures the entry by `RemoteKVCache.residentByteCount`, which is the bytes on the workers, against this process's memory budget. When the entry is over the budget it goes to `spill`. A RemoteKVCache can't be saved, so the spill drops it, and the next turn of the session starts cold. TTFT was 49.6 of 88.9 model-minutes.

## Fix
- A RemoteKVCache counts no bytes of the head process. ExecutorPromptCacheEntry.byteCount counts only local caches, so the memory budget doesn't apply to cluster caches; the cluster plan reserves their bytes on the workers.
- An eviction never drops the only entry of a live session.
- Spilling a remote cache means dropping it, and only when no session holds it. Never call ExecutorPromptCacheFile.prepare for it.

Tests: a store test with a stub remote cache whose residentByteCount is over the budget. After check-in, the entry stays and the next checkOut is `.memory`. A remote entry is never spilled to disk.

The Router half (the derived window leaves the store almost no budget) is FoundationModelsRouter ^n98hpry.
Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/preds.cluster2i.telemetry.jsonl; build bench/bin/cluster-0b560b5. #cluster