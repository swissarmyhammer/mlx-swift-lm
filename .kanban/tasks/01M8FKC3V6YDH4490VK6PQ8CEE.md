---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8fs0ycs92018fz50jm2ghe7
  text: |-
    ### Scope grows: protocol v5 (from ^2ry6yz7)
    ^2ry6yz7 raises RemoteModelProtocol.version from 4 to 5. `newCache` now carries the KV quantization (`.newCache(cache, quantization)`), and there is a new `.quantize` op. FoundationModelsCluster must adopt v5 too, because the version check is an exact match:
    - Sources/ClusterProtocol/ForwardTokenCount.swift: add a `.quantize` case to the switch.
    - IntegrationTests/Sources/fmcluster-parity/DecodeRun.swift: `.newCache(let cache)` becomes `.newCache(let cache, _)`.
    - IntegrationTests/Sources/fmcluster-parity/Recording.swift: add `.speculate` and `.quantize` to the `name` switch.
    - ModelProtocolCheckTests may pin the version.
    After that, run the real 2-Mac parity checks: Qwen3.8-27B with kvBits 8, and MTP on Qwen and Inkling.
  timestamp: 2026-11-28T00:31:31.737131+00:00
- actor: claude-code
  id: 01m8fv3j63fnhtxht0np23g462
  text: |-
    ### Owner: the FoundationModelsCluster session (foundationmodelscluster-4e)
    Per foundationmodelsacpagent-a1 (2026-11-27), this work is in the FoundationModelsCluster repo and that repo's session owns it. The v4 drafter half landed there as ^h3kq14q (main da4c10b), and foundationmodelsacpagent-a1 has sent it the v5 list. The mlx-swift-lm /finish run skips this card; close it here when the Cluster side lands v5.
  timestamp: 2026-11-28T01:07:54.691845+00:00
- actor: claude-code
  id: 01m8g2m113xzabc696r42nct50
  text: |-
    ### Closed: done in FoundationModelsCluster
    foundationmodelsacpagent-a1 reported on 2026-11-27 that the Cluster session did this work as ^h3kq14q:
    - 2adceb2: the v4 drafter
    - 8299246: v5

    It is proven on the real bench. In cluster4i (Inkling) and cluster5q (Qwen), rank 0 loaded the MTP drafter and drafts were accepted. Inkling resolved django-13447 in 32.9 min at 94.9 tok/s with 61% of drafts accepted. Qwen q8kv8 resolved it in 37.9 min at 71.5 tok/s with 87.8% accepted. No code change in mlx-swift-lm, so there is no review here.
  timestamp: 2026-11-28T03:19:14.211283+00:00
position_column: done
position_ordinal: ffffff8e80
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