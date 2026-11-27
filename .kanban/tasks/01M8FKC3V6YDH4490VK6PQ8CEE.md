---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: 'FoundationModelsCluster: adopt remote protocol v4 (MTP drafter on rank 0 + draft exchange)'
---
## What
^rgzk56j (mlx-swift-lm) adds remote model protocol v4: `RemoteModelOp.speculate(RemoteDraftRound)`, `RemoteModelReply.drafted`, `RemoteModelCapabilities.drafter`, and `RemoteModelExecutor(model:configuration:drafter:draftExchange:)`. The cluster worker in ../FoundationModelsCluster (Sources/fmcluster-worker/Serve.swift `load(directory:)`) still builds `RemoteModelExecutor(model:configuration:)`, so a real cluster reports no drafter and every turn decodes plainly (fallback incompatibleDrafter).

## Fix
- On rank 0, load the MTP drafter of the checkpoint (Qwen MTP head in the checkpoint, Inkling DSpark mtp.safetensors) beside the sharded target (`loadMTPDrafter` / drafter registry) and pass it as `drafter:`.
- On EVERY rank pass `draftExchange: .group(group)` (or `.alone` with no group). Each rank must call the exchange once per round whose draft limit > 0, so all ranks need the same exchange kind.
- `forwardTokens(_:)` in Serve.swift / WorkerCachePolicy should also count `.speculate` rounds (verify tokens = drafts + 1).
- Bump the worker's control version / pin mlx-swift-lm to the commit of ^rgzk56j; protocol version is now 4.

## Goal
django__django-13447 on m5u1+m5u2 with Qwen3.8 27B and Inkling 2 Mini, with generation.mtp.* telemetry showing accepted drafts. #cluster