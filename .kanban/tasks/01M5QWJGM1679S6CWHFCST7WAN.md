---
assignees:
- claude-code
depends_on:
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8680'
title: 'MTP: T-token GDN kernel with per-token state snapshot slots, and MambaCache slot selection'
---
## What
A verify pass sets `mtpCacheCheckpointIndexKey`, and `recurrentBody` (Libraries/MLXLLM/Models/Qwen35.swift ~450) splits the input at `checkpointAfter` and runs `gatedDeltaUpdate` two times to take one checkpoint. `GatedDeltaDecodeKernel` (Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift) handles only T = 1.

### Design from llama.cpp (PR 22673 and PR 22400)
The Metal kernel `kernel_gated_delta_net_impl` (ggml/src/ggml-metal/kernels/gated_delta_net.metal) keeps the state in registers, loops over the T tokens in one dispatch, and writes the state after each of the last K tokens into K snapshot slots. The conv state of each slot is a view of the conv input window (src/models/delta-net-base.cpp). A rollback of r tokens only selects slot r. Cost for Qwen3.8: one slot is 48 layers × 48 heads × 128 × 128 × 4 B ≈ 151 MB, ≈ 0.28 ms of write traffic for each slot at 546 GB/s.

- [ ] Extend `GatedDeltaDecodeKernel` to T ≤ 6 tokens: one dispatch, the state stays in registers, and the kernel writes the recurrent state after each token into a `[T, …]` snapshot output. Get the conv state for each slot from the conv input window. Keep the T = 1 output bit-identical to the current kernel.
- [ ] Use it in the general multi-token GDN path when the speculative flag is set (replace the two-pass split in `recurrentBody`), in Libraries/MLXLLM/Models/Qwen35.swift and Libraries/MLXVLM/Models/Qwen35.swift.
- [ ] In `MambaCache` (Libraries/MLXLMCommon/KVCache.swift ~1640-1875), keep the snapshot slots of the last verify as transient speculative data, next to the current `speculativeCheckpoint`. Selecting slot r makes the current state a separate array (an evaluated copy of the slot, not a slice view, so that the `[T, …]` buffer is released). `saveSpeculativeCheckpoint` keeps working for the current blockSize 2 iterator (it selects the slot after token 1).
- [ ] The slots are excluded from `copy()`, `copyContents`, `PromptCacheCheckpoint`, serialization, and `byteCount`. `discardSpeculativePromptCacheCheckpoints` and the end of the round release them.

## Acceptance Criteria
- [ ] Each snapshot slot and its conv state match the op path within the existing GDN tolerance for T = 2..6, and T = 1 stays bit-identical.
- [ ] A rollback by r = 1..T−1 tokens gives the same recurrent and conv state as a run of the T − r accepted tokens only.
- [ ] After a round, no `MambaCache` holds a reference to a snapshot buffer (`byteCount` and a copy of the cache do not change with T).
- [ ] Greedy MTP output still matches plain decode (existing Qwen35MTPTests).

## Tests
- [ ] Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift: T = 2..6 kernel against the op path, every snapshot slot and conv state.
- [ ] Tests/MLXLMTests/KVCacheTests.swift: slot selection, release, and exclusion from copy, checkpoint, serialization, and byte count.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf