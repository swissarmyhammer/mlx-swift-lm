---
assignees:
- claude-code
depends_on:
- 01M5QWJGM1679S6CWHFCST7WAN
position_column: todo
position_ordinal: '9480'
title: 'MTP: compiled multi-token verify schedule for Qwen35 (LLM and VLM targets)'
---
## What
A verify pass has `x.dim(1) >= 2`. The guards in `Qwen35DecoderLayer.callAsFunction` (Libraries/MLXLLM/Models/Qwen35.swift ~797) and `Qwen35TextModelInner.forward` (~988) send it to the general path, which has no compiled segments. Measured: the 2-row matmuls cost only 1.10× a 1-row step (scratchpad rows.py), but a full MTP round costs 1.31× a plain step.

- [ ] Add a fixed-shape decode schedule for S = 2..6 to `Qwen35TextModelInner` (reuse `CompiledDecodeSegment` / `CompiledDecodeSegmentCache`, keyed by S). It uses the T-token GDN kernel with snapshot slots (previous task) and emits the drafter hidden rows (the `applyFinalNorm: false` path of `Qwen35TextModel.forward` ~1191).
- [ ] Use `DecodeAttentionKernel` / `PrefillAttentionKernel` for the S-query attention step where they apply.
- [ ] The MLXVLM Qwen35 model (Libraries/MLXVLM/Models/Qwen35.swift) has its own emit and checkpoint path (`Qwen35VLMMTPRegistration`). Give it the same schedule, or keep it on the general path and test that it still works.

## Acceptance Criteria
- [ ] The verify output (logits, hidden rows, caches, and snapshot slots) of the compiled path is within the existing `AttentionKernelTolerance` against the general path, for S = 2..6.
- [ ] Greedy MTP output equals plain decode on the shared fixture, for the LLM and VLM targets.
- [ ] Release benchmark on the M5 Max (`qwen38FourBitMTPGeneration`, `#expect`): MTP tok/s on the math category improves by ≥ 8% against the baseline of 52.3 tok/s, and the output still matches plain decode.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: compiled S = 2..6 verify against the general path, for rounds that accept and rounds that reject; a VLM-target MTP equality test.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes; then the Release benchmark.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf