---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: GatedDelta ops fallback uses CustomFunction, which creates a second GPU stream
---
Found by FoundationModelsCluster's GPU-stream guard (wevkejv, 13d83dd).
- MLX's `custom_function` calls `stop_gradient` on the calling thread's C++ default stream, so each call creates a second GPU stream. That stream is the known cause of the FAST_SYNCH hangs (exo#2391).
- `GatedDeltaRecompute.masked/unmasked` (Libraries/MLXLMCommon/GatedDelta.swift:416) are CustomFunctions. They're reached only through `gatedDeltaOps`, which runs when neither fastGatedDelta nor the Metal kernel applies: training (`useKernel: false`), or Dk % 32 != 0.
- Inference with Qwen3.8 and Inkling (Dk = 128) never reaches it, so clustered workers are safe today.

Fix: when a distributed group is active, keep the ops fallback off CustomFunction (plain ops with no VJP wrapper), or pass the stream explicitly. Add a test that asserts one GPU stream after a forward through the fallback, using `mlx_get_streams`. Low priority. It matters only if training or an odd Dk ever runs clustered.

#cluster