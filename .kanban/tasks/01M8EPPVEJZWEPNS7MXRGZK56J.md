---
position_column: todo
position_ordinal: '9180'
title: 'Clustered models: support speculative decoding with the MTP drafter (Qwen MTP, Inkling DSpark)'
---
## What
ClusterModelFeature.speculativeDecoding is unsupported (Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift:112-119). In cluster1i (Inkling-2-Mini-4bit + oQ4e DSpark drafter) every one of 19 generation calls ran without MTP: 67 tok/s, against ~105 tok/s for the same model on one Mac with MTP (about 63% of drafts accepted). So on one task the cluster is slower than one Mac for Inkling, and Qwen loses its 1.7x MTP gain too.

## Fix
Run the drafter on rank 0 (it is small) and verify the drafted block with one clustered forward of block+1 tokens, with the cache trim on every rank on reject, as the local MTP path does. Expose the same generation.mtp.* telemetry.
Tests: parity suite with the drafter on Inkling and Qwen: greedy output equals the clustered no-MTP output; accepted-token counts reported.

Goal: one SWE-bench task (django__django-13447) runs to a patch on the 2-Mac cluster with Qwen3.8 27B and with Inkling 2 Mini.
Evidence (ACPAgent repo): bench/run.cluster1i.log, bench/preds.cluster1i.telemetry.jsonl, bench/run.cluster1q.log, bench/preds.cluster1q.transcripts/. Build bench/bin/cluster-40d7f3a: Router f3624eb9, Cluster 0b8d4848, mlx-swift-lm stable 3bd174ed, mlx-swift 2d5a12b7. #cluster