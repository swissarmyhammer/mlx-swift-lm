---
assignees:
- claude-code
depends_on:
- 01M5SK9FZ2HTB7XJDAPCBRDDM5
position_column: todo
position_ordinal: '8180'
title: 'MTP: chunked prompt prefill that feeds the drafter for each chunk'
---
## What
`MTPSpeculativeTokenIterator.init` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift) sets `mtpPrefill.stepSize = Int.max` and `.unchunked` when `drafter.requiresPromptPrefill`. The target then runs the full prompt in one forward pass. For a long prompt the peak memory is very large.

The chunk loop must stay inside the model `prepare`, not in the iterator: the MLXVLM Qwen35 `prepare` merges images and computes the M-RoPE positions. Also, on a warm cache the MLXVLM Qwen35 `prepare` goes to `prepareContinuation` (branch `cacheOffset > 0 || len > window`, Libraries/MLXVLM/Models/Qwen35.swift), whose `forward` passes `state: nil`. The emit flag is then lost, the iterator gets one hidden row, fails `targetHidden.dim(1) >= promptLength` (~370), and every continuation goes to passthrough.

llama.cpp uses the same design: a hook runs after each target ubatch (`common_speculative_impl_draft_mtp::process`). It loses about one third of its prefill speed with MTP, mostly from device-to-host copies of the hidden rows. Keep the hidden rows on the GPU.

- [ ] Add a per-chunk hook to `LanguageModel.prepare` (for example an `onChunkHiddenStates: (tokens, hidden) -> Void` in the prepare state, called after each chunk when `mtpEmitFlagKey` is set). Implement it in MLXLLM Qwen35 and MLXVLM Qwen35, including `prepareContinuation`.
- [ ] In the iterator, feed the drafter append (from the model-side task) from the hook, with the chunk tokens shifted by one and the chunk hidden rows; keep the last hidden row as the pending hidden. Remove the forced `stepSize = Int.max` / `.unchunked`.
- [ ] Do not keep the hidden rows of earlier chunks alive (evaluate the drafter cache and the main cache after each chunk). Do not copy hidden rows to the CPU.

## Acceptance Criteria
- [ ] Chunked MTP prefill (stepSize 64 on a prompt of more than 300 tokens) gives the same greedy tokens and the same drafter cache contents (within float tolerance) as unchunked prefill, for the MLXLLM and the MLXVLM Qwen35 targets.
- [ ] A warm-cache continuation on the MLXVLM target runs MTP (no passthrough).
- [ ] Peak memory during the MTP prefill of a prompt of 4000+ tokens with stepSize 512 is lower than unchunked (`Memory.peakMemory`).
- [ ] Release benchmark on the M5 Max (`qwen38FourBitMTPGeneration`): decode tok/s within ±3% of the baseline, and MTP prefill tok/s ≥ 85% of plain prefill tok/s.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared fixture: chunked against unchunked equality (LLM and VLM targets), the VLM warm continuation, and a test that the hook is called one time for each chunk.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38