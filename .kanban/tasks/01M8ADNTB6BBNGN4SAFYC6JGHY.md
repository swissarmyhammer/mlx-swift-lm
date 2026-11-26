---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: Greedy argmax breaks exact bf16 ties differently on the local and remote paths
---
Found by the FoundationModelsCluster parity suite (pne5cfr, 2026-11-25). Exact bf16 logit ties are common, and greedy decoding picks a different tied token depending on the path:
- Qwen3.8 guided, step 44: tokens 17 and 18 have margin 0.0, bit-identical on the local and one-rank lanes. Local greedy picked 18; the worker path picked 17.
- Inkling guided, step 49: tokens 38 and 40172 both at logprob -0.813. One rank picked 40172; the cluster picked 38.

Correctness doesn't need this: parity passes under vLLM's check_logprobs_close, since both tied tokens are in both top-5s. But local and clustered greedy runs would agree token for token if the greedy sampler broke exact ties deterministically, for example by the lowest token id, whatever the logits' dtype, shape or path (the local logits vs the float32 RemoteLogits copy).

To do: find why argMax differs on equal values (dtype conversion before the argmax? different reductions?), make greedy tie-breaking deterministic, and test it with a logits row that holds an exact tie in bf16 and in float32. Low priority. #cluster