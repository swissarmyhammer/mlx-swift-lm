---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rzcbvfde73qv29ajxjtfxa
  text: |-
    ### Research 2026-10-12 (session scratchpad dl/, Python sources)
    - Give the position offset to a compiled function as an `MLXArray` input. Never capture it as a Swift `Int`. A captured Int either adds one compile-cache entry for each token, which made Python slow down linearly (PR #1189 comments, `c1189.md:1039-1094`), or freezes the value.
    - Compile targets: gate select, SwiGLU, hyper-connection expand, compressor pooling (`mlx_vlm .../language.py:390-414`), and the split softmax.
    - The gate weight cast and the query norm moved to task ^tj8ds2m. Do not do them twice.
  timestamp: 2026-10-12T21:13:00.015334+00:00
- actor: claude-code
  id: 01m4v40fkgtgag94kv8vc39d66
  text: |-
    ### Profile after ^9r9yxh9 — 2026-10-13, commit 05475ad, GPU idle, 128 context, 21 pipelined decode steps
    Tools: `xcrun xctrace record --template 'Metal System Trace' --instrument 'Points of Interest'` and `--template 'Time Profiler'` on `dsv4bench <snapshot> profile 128` (scratch bench, signpost "decode" marks the window). Scripts: scratchpad `scripts/mst_analyze.py`, `scripts/tp_analyze.py`.
    - Window 1154 ms for 21 steps, about 53-55 ms for each step.
    - GPU busy 187 ms, 16 % of the window, about 9 ms for each step. That is near the memory-bandwidth floor for 13B active 4-bit weights. The GPU waits for the CPU 84 % of the time.
    - Compute thread CPU 836 ms, about 40 ms for each step:
      - `mlx::core::async_eval` 640 ms: `eval_impl` 600, of which `gpu::eval` 454 (encoding each primitive: `copy_gpu_inplace` 101, `binary_op_gpu` 76, `dispatch_threads` 66, `set_compute_pipeline_state` 61, `concatenate_gpu` 53, `QuantizedMatmul::eval_gpu` 31).
      - Graph build in Swift, `DeepSeekV4Model.callAsFunction` 195 ms: attention 115, of which `partialRoPE` 39, `attentionOutput` 57; MoE 32; hyper-connection collapse 36.
    - Other threads: about 320 ms in command-buffer submit and completion (about 180 command buffers for each step).
    Conclusion: the step cost scales with the count of MLX primitives, not with arithmetic. Fewer primitives is the goal. Compile of small elementwise chains fuses primitives; also try a compiled decoder layer for the L=1 decode path, with the offset given as an MLXArray input.
  timestamp: 2026-10-13T17:12:22.384332+00:00
- actor: claude-code
  id: 01m4v46154aepnc3d2yrbs7z5c
  text: |-
    ### Target: mlx-vlm on this machine — 2026-10-13, GPU idle
    Blaizzy/mlx-vlm commit 2b93e7782e20f1883138402db12bc1c1428cfe93, MLX 0.32.3 (the same MLX as our Swift build), same checkpoint snapshot, script scratchpad `scripts/bench_dsv4_decode.py`, venv scratchpad `py/venv-vlm`.
    - 128 context: sync 27.4 ms, pipelined 26.9 ms (37.2 tok/s). Prefill note: 128 tokens is too short to measure.
    - 4k: sync 31.1 ms, pipelined 31.1 ms (32.1 tok/s), prefill 418.5 tok/s.
    - 16k: sync 31.8 ms, pipelined 31.5 ms (31.8 tok/s), prefill 433.5 tok/s.
    - Ours at commit 05475ad: 53 / 56 / 68 ms.
    mlx-lm PR 1195 (commit 905df9c) cannot load this checkpoint: `ValueError: Unable to quantize model of type QuantizedSwitchLinear`.
  timestamp: 2026-10-13T17:15:24.196777+00:00
- actor: claude-code
  id: 01m4v693qzx7bjr8hdvbm32qxc
  text: |-
    ### Research and baseline — 2026-10-13, HEAD e08347b, GPU idle (0-1 %)
    Baseline `dsv4bench <snapshot> step` (scratch `dsv4bench/step-base.txt`), two rounds each:
    - 128: A sync 50.8 / 50.1 ms, B pipelined 49.9 / 49.0 ms, C generate 58.6 / 57.5 ms.
    - 4k: A 52.5 / 52.7, B 51.8 / 52.4, C 58.5 / 58.3.
    - 16k: A 65.1 / 64.6, B 63.3 / 63.3, C 67.8 / 71.3.
    Greedy baselines: `greedy-base.txt` (`32 64` text and cache check) and `ids-base.txt` (`ids fixture/vlm-transpose.json 64`). The cached ids on HEAD are the same as the `generated_token_ids` of the mlx-vlm fixture.
    Findings:
    - MLX `compile` inlines a compiled function that is called during a trace of another compiled function (`compile.cpp`: "If the inputs are tracers, trace the original graph"). Thus a small compiled function can stay in place inside a larger compiled region.
    - MLX JIT kernels use `MathMode::Safe`, but the Metal compiler still contracts `a * b + c` to `fma` (the Sinkhorn task saw this). A fused kernel can thus change bits against the op path. The parity tests must check bits.
    - The prevailing pattern (Qwen35 `CompiledTrace`, `CompiledDecodeSegments`) compiles the L=1 decode path only. Prefill stays on the op path, because each new prefill length would add one trace for each layer and the compile cache keeps each one.
    - Plan: compile the clamped SwiGLU, the gate selection (hash and top-k), the hyper-connection expand, the collapse up to the Sinkhorn kernel (rms reduction, mixes projection, split softmax) and the routed/shared combine of the MoE. Each takes every array as an argument; only per-model constants are captured. Then evaluate a larger compiled region of the decode layer.
    - Out of scope here (other tasks): gate weight cast at load, query-head norm, masks at L=1, sink cast (^tj8ds2m); rope (^phf8pwt); compressor (^cm8fq2q).
  timestamp: 2026-10-13T17:52:02.303680+00:00
- actor: claude-code
  id: 01m4v6xwzqjqyh71w6ee49g6mq
  text: |-
    ### Small compiled functions landed — 2026-10-13
    - New file `Libraries/MLXLLM/Models/DeepSeekV4CompiledDecode.swift`: compiled clamped SwiGLU, expand, collapse fields (rms reduction + mixes projection + split softmax), top-k and hash routing, MoE combine. A block of one token reads them; prefill keeps the op path.
    - `DeepSeekV4CompiledDecodeTests` (9 tests): every compiled function gives the SAME BITS as its op path (float32 and bfloat16). No fma contraction showed up in these fused kernels.
    - Greedy ids on the real checkpoint (`dsv4bench ids fixture/vlm-transpose.json 64`): cached and full lists are the same as on HEAD.
    - Step bench (GPU idle), `step-small.txt`: 128: A 48.5/48.8, B 47.3/47.7, C 57.2/57.8 ms; 4k: A 49.9/50.8, B 48.9/49.6; 16k: A 63.0/62.8, B 61.3/61.2. That is about 2 ms better than HEAD (50 / 52 / 65 ms sync). The small functions alone do not remove enough primitives. Next: evaluate a compiled region of the decoder layer (all but the cache work).
  timestamp: 2026-10-13T18:03:23.511162+00:00
- actor: claude-code
  id: 01m4v88ypezxtxdczzms8z02j3
  text: |-
    ### Compiled decoder layer evaluated and kept — 2026-10-13
    What I tried, and the numbers (GPU idle, scratch `dsv4bench step`, A = direct sync, B = direct pipelined, C = library generate; two rounds each):
    - HEAD e08347b (`step-base.txt`): 128 A 50.8/50.1 B 49.9/49.0 C 58.6/57.5; 4k A 52.5/52.7 B 51.8/52.4 C 58.5/58.3; 16k A 65.1/64.6 B 63.3/63.3 C 67.8/71.3 ms.
    - Small compiled functions only (`step-small.txt`): 128 A 48.5/48.8 B 47.3/47.7; 4k A 49.9/50.8 B 48.9/49.6; 16k A 63.0/62.8 B 61.3/61.2 ms. About 2 ms.
    - Small functions + two `CompiledTrace`s for each decoder layer (`step-final.txt`): 128 A 45.5/45.7 B 43.8/44.2 C 49.0/47.9; 4k A 48.4/48.0 B 47.1/47.2 C 52.2/53.5; 16k A 60.6/61.1 B 58.6/58.2 C 62.4/62.8 ms. The layer traces beat the small functions alone by about 3 ms more, thus I kept them.
    - Prefill unchanged (0.4 / 10.0 / 47.6 s). The traces add about 20 ms to the first decode step of a process (88-90 ms first step against 68-70 ms warm).
    Design: `DeepSeekV4DecoderLayer.decodeStep` runs `compiledAttentionInput` (collapse, attn norm, q/kv projections and rope) and `compiledLayerTail` (inverse rope, grouped output projection, expand, whole MoE half). The stage between them (pooled chunks, cache write, masks, indexer, SDPA) stays outside, because the caches hold Swift `Int` state and grow each step. The rope tables are trace ARGUMENTS (computed from the cache offset outside the trace), so no offset becomes a trace constant. Weights reach the traces as `CompiledTrace` state. A compiled function called inside a trace is inlined by MLX (`compile.cpp`: tracer inputs trace the original graph).
    Bits: the fused kernels gave the SAME bits as the op path in every test (float32 and bfloat16, small functions and whole layer step). No fma contraction showed up, thus no `fp contract(off)` work was needed.
    Profile after (Time Profiler, 21 steps at 128, scratch `prof2/`): graph build in Swift 131 ms (was 195), `gpu::eval` 378 ms (was 454). Of what is left: `DeepSeekV4ChunkCache.pooled` 43 ms, `copy_gpu_inplace` 76 ms, `concatenate_gpu` 43 ms (the compressor pooling, task ^cm8fq2q), and `compile_replace` 46 ms (86 trace calls per step; follow-up task ^4dx93af merges the tail of a layer with the head of the next).
    Correctness on the real checkpoint: `dsv4bench ids fixture/vlm-transpose.json 64` gives the same cached and full id lists as HEAD (both equal the mlx-vlm fixture for the cached run); `dsv4bench 32 64` text, token counts and cache-check text are the same as HEAD; integration test `greedyFirstTokensMatchThePythonFixture()` passed (xcodebuild, alone).
    Unit tests: `swift build --build-tests` clean; `xcrun xctest MLXLMTests.xctest`: 716 XCTest and 1375 Swift Testing tests, 0 failures, 0 skips.
  timestamp: 2026-10-13T18:26:54.286309+00:00
- actor: claude-code
  id: 01m4v899kjy2nb164v066n754c
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLLM/Models/DeepSeekV4CompiledDecode.swift (new), Libraries/MLXLLM/Models/DeepSeekV4.swift, Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Libraries/MLXLLM/Models/DeepSeekV4MoE.swift, Tests/MLXLMTests/DeepSeekV4CompiledDecodeTests.swift (new), Tests/MLXLMTests/DeepSeekV4ModelTests.swift, docs/deepseek-v4-support.md. Decode step sync before/after: 128 50.5 -> 45.6 ms, 4k 52.6 -> 48.2 ms, 16k 64.9 -> 60.9 ms (pipelined 49.5/52.1/63.3 -> 44.0/47.2/58.4). Unit suite 716 XCTest + 1375 Swift Testing, 0 failures; greedy ids and text same as HEAD e08347b; integration greedy parity passed.
    - next: /review. Not committed.
  timestamp: 2026-10-13T18:27:05.458704+00:00
- actor: claude-code
  id: 01m4v8kx9gjztd1ytgakh5759c
  text: |-
    ### test - green
    - evidence: swift build --build-tests exit 0 (MLXLLM and MLXLMTests rebuilt after touch); only warning is the known "missing creator for mutated node" (mlx-swift_Cmlx.bundle). xcrun xctest on 5 bundles: CXGrammarTests 7 passed, MLXFoundationModelsTests 367 passed, MLXGuidedGenerationTests 70 passed, MLXHuggingFaceMacrosTests 5 XCTest passed, MLXLMTests 716 XCTest + 1375 Swift Testing passed. 0 failures, 0 skipped. xcodebuild build-for-testing IntegrationTesting: TEST BUILD SUCCEEDED, no new warning. Real-weights integration tests not run, as ordered.
    - verify-docs: scripts/verify-docs.sh exit 1. It builds DocC for Swift targets only. It does not read docs/deepseek-v4-support.md. Every DocC error is in code that is not in this change (for example DeepSeekV4.swift symbol link from commit fdfdecf, DeepSeekV4Cache.swift, Tokenization, SwitchLayers). One more error: ExtractAPI of MLXCXGrammar fails with picojson.h 'algorithm' file not found (toolchain). No error in DeepSeekV4CompiledDecode.swift or in the changed hunks.
    - next: none for this change. Pre-existing DocC link errors need their own task.
  timestamp: 2026-10-13T18:32:53.296778+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 speed: compile the small hot functions of the decode step'
---
## What

The DeepSeek-V4 port has no compiled path. `Libraries/MLXLLM/Models/DeepSeekV4.swift:47-50` records this as a known difference from the Python reference, which uses `compile(shapeless:)`.

Functions to compile:
- The gate selection: `DeepSeekV4MoE.swift:309-346`. Also move the float32 cast of the gate weight at line 312 to load time.
- `clampedSwiGLU`: `DeepSeekV4MathHelpers.swift:341-350`.
- The hyper-connection `expand`: `DeepSeekV4HyperConnection.swift:286-298`.
- Replace the manual query-head norm at `DeepSeekV4Attention.swift:673-674` with `MLXFast.rmsNorm`.

Python equivalents in Blaizzy/mlx-vlm `deepseek_v4/language.py`: `_expert_select`, `_limited_swiglu`, `_hc_expand_op`.

Note (comment of 2026-10-12): the gate weight cast at load time and the query norm moved to task ^tj8ds2m.

## Warning

Give the position offset to a compiled function as an array, not as an `Int`. In mlx-lm PR 1189, an `Int` offset made a new compile-cache entry for each token, and decode fell from 29 to 17.7 tok/s after 6K tokens. `Libraries/MLXLMCommon/CompiledTrace.swift` already exists.

## Acceptance Criteria

- [x] The listed functions run through `compile`. No compiled function takes a value that changes on each step as a Swift scalar.
- [x] Parity tests show the same results as the current path.
- [x] The existing DeepSeek-V4 unit tests pass.
- [x] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after, at 128 and at 16k context.
#deepseek-v4 #performance