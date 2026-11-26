---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8c8b5rxcs258gq8e8751vme
  text: |-
    Research:
    - Confirmed in mlx core (transforms.cpp custom_function): outputs pass through `stop_gradient(out)` with no stream, so `default_stream(default_device())` runs; default streams are thread_local (stream.cpp default_stream_storage), so the first CustomFunction call on each thread makes a new GPU stream. The mlx-c / Swift CustomFunction API has no way to pass a stream, so "pass the stream explicitly" is not available from this repo; taking the plain-ops route.
    - No MLX API reports "a distributed group is active" without side effects (a non-strict `MLXDistributed.initialize` caches a size-1 group under jaccl and would break a later strict init). The group is known where the model shards: `Qwen35DecoderLayer.shard(_:with:group:)` -> `Qwen35GatedDeltaNet.adoptShardedHeads(from:)`, only reached when plan.groupSize > 1. Qwen35 (MLXLLM) is the only TensorParallelShardable model that uses gatedDeltaUpdate (Inkling2 does not; Qwen3Next and VLM Qwen35 are not shardable).
    - Plan: `gatedDeltaOps`/`gatedDeltaUpdate`/`Snapshots.gatedDeltaUpdate` gain a defaulted `recomputeChunks: Bool = true`; false runs the chunk steps as plain ops (same forward, plain-ops VJP keeps every step: correct gradients, more memory). The sharded GDN layer turns it off in adoptShardedHeads.
    - Test: forward through the ops fallback on a fresh Thread (fresh C++ thread-local default stream), compare the GPU-stream count from `mlx_get_streams` before/after.
  timestamp: 2026-11-26T15:42:17.885888+00:00
- actor: claude-code
  id: 01m8c8mypz0p7w8why97r9yht8
  text: |-
    Implementation landed (TDD):
    - RED 1: `GatedDeltaGPUStreamTests` "forward through the ops fallback" on a fresh Thread added GPU stream(s) (CustomFunction path). GREEN after adding `recomputeChunks: Bool = true` to `gatedDeltaUpdate` / `gatedDeltaOps` / `Snapshots.gatedDeltaUpdate`; false runs each chunk as plain `GatedDeltaRecompute.steps` (new `GatedDeltaRecompute.chunk(masked:recomputing:)`).
    - RED 2: a `Qwen35GatedDeltaNet` after `adoptShardedHeads` (Dk = 16 → fallback in inference) added GPU stream [2]. GREEN: layer gets `private(set) var recomputesChunks = true`, set false in `adoptShardedHeads` (only reached for a group of >1 in `Qwen35DecoderLayer.shard`), passed to both gatedDeltaUpdate call sites in `recurrentBody`.
    - Gradient guard: ops gradients (q,k,v,a,b) with recomputeChunks false equal the recomputing path exactly (largest diff 0).
    - Test pitfall for the next agent: measure on a fresh Thread that runs ONLY the forward. Evaluating inputs on that thread first (or eval of already-evaluated arrays, which falls back to `default_stream`) creates the thread's C++ default stream outside the measured span and hides the bug.
    - `swift build --build-tests` prints `warning: missing creator for mutated node: (.../mlx-swift_Cmlx.bundle/Contents/MacOS)` even on a no-op rebuild; it is a SwiftBuild warning about the Cmlx resource bundle, not from these files.
  timestamp: 2026-11-26T15:47:38.335353+00:00
- actor: claude-code
  id: 01m8c93jrhh8tybhxtckt3pqkd
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLLM/Models/Qwen35.swift, Tests/MLXLMTests/GatedDeltaGPUStreamTests.swift (new, 3 tests). `swift build --build-tests` clean apart from the SwiftBuild Cmlx-bundle "missing creator" warning that also appears on a no-op build. xctest: MLXLMTests 1261 XCTest + 2749 Swift Testing passed (3 known issues, all pre-existing withKnownIssue: AttentionKernelToleranceTests, MiniCPM5 CDATA); MLXFoundationModelsTests 618 passed; MLXGuidedGenerationTests 95 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 passed. All existing GatedDelta tests pass, including testGatedDeltaOpsPathIsDifferentiable and the fast-path gradient test.
    - next: /review
  timestamp: 2026-11-26T15:55:37.617403+00:00
position_column: doing
position_ordinal: '80'
title: GatedDelta ops fallback uses CustomFunction, which creates a second GPU stream
---
Found by FoundationModelsCluster's GPU-stream guard (wevkejv, 13d83dd).
- MLX's `custom_function` calls `stop_gradient` on the calling thread's C++ default stream, so each call creates a second GPU stream. That stream is the known cause of the FAST_SYNCH hangs (exo#2391).
- `GatedDeltaRecompute.masked/unmasked` (Libraries/MLXLMCommon/GatedDelta.swift:416) are CustomFunctions. They're reached only through `gatedDeltaOps`, which runs when neither fastGatedDelta nor the Metal kernel applies: training (`useKernel: false`), or Dk % 32 != 0.
- Inference with Qwen3.8 and Inkling (Dk = 128) never reaches it, so clustered workers are safe today.

Fix: when a distributed group is active, keep the ops fallback off CustomFunction (plain ops with no VJP wrapper), or pass the stream explicitly. Add a test that asserts one GPU stream after a forward through the fallback, using `mlx_get_streams`. Low priority. It matters only if training or an odd Dk ever runs clustered.

#cluster