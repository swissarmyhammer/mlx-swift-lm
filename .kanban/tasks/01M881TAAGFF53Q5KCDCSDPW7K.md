---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: 'Executor follow-ups: park request-id check, loaded model_type in describe'
---
Small follow-ups from 8aeeqed (RemoteModelExecutor, 38ff740d).
- [ ] A PARK whose request id doesn't match the open request gets a `.failed` reply on rank 0 and an error on ranks > 0, so a desynced rank fails loudly.
- [ ] `describe`'s capabilities also name the loaded `model_type` (and quantization), not only the shardable family list. Bump `RemoteModelProtocol.version` to 2, since nothing has shipped. Update ClusterLanguageModel and the tests.
- [ ] Note for the worker session: ClusterLanguageModel sends `end` only at the next `prepare`, so the worker's PARK-after-~250 ms idle is what releases members between turns. Document that in RemoteModelProtocol's DocC.
- [ ] Note: `prism_hadamard_qwen35` and `qwen3_5_moe` aren't shardable yet (rotated projections, MoE). That's for the larger-models phase. #cluster