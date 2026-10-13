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
position_column: todo
position_ordinal: '8280'
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

## Warning

Give the position offset to a compiled function as an array, not as an `Int`. In mlx-lm PR 1189, an `Int` offset made a new compile-cache entry for each token, and decode fell from 29 to 17.7 tok/s after 6K tokens. `Libraries/MLXLMCommon/CompiledTrace.swift` already exists.

## Acceptance Criteria

- [ ] The listed functions run through `compile`. No compiled function takes a value that changes on each step as a Swift scalar.
- [ ] Parity tests show the same results as the current path.
- [ ] The existing DeepSeek-V4 unit tests pass.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after, at 128 and at 16k context.
#deepseek-v4 #performance