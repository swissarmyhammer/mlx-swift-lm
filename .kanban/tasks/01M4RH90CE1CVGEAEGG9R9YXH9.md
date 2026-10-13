---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rz1st5s739g021f73gp5fs
  text: |-
    ### Baseline before this task — 2026-10-12, GPU idle, commit 4041c5b
    Release build, mlx-community/DeepSeek-V4-Flash-4bit, greedy, scratch bench `scratchpad/dsv4bench`.
    - Step bench (`dsv4bench <snapshot> step`), one decode step: A direct sync 104 ms @128 ctx, 109-110 ms @4k, 126-135 ms @16k; C generate 126 ms @128, 110-118 ms @4k, 123-129 ms @16k.
    - Full bench (`dsv4bench <snapshot> 32,2048,8192,32768 128 8192 1024`): decode 110.7 ms/tok @32, 121.2 @2k, 128.4 @8k, 133.0 @32k. Prefill 397 tok/s @2k, 375 @8k, 276 @32k.
    - Cache check @8k: chat and thinking both reuse (turn 2 prefilled 10), warm == cold.
    Measure the same two commands after this task, with the GPU idle.
  timestamp: 2026-10-12T21:07:13.861215+00:00
- actor: claude-code
  id: 01m4rz68m82wm7x4xpbthvqzfx
  text: |-
    ### Research
    - The op path is `DeepSeekV4Math.hcSplitSinkhorn`. Only `DeepSeekV4HyperConnection.collapse` calls it. The Sinkhorn part is one column normalization and then `iters - 1` pairs of a row and a column normalization. Each normalization is 3 ops (sum, add eps, divide). With 20 iterations that is 39 normalizations, about 117 dispatches for each hyper-connection.
    - The mlx-lm kernel `deepseek_v41_sinkhorn` does only the Sinkhorn loop: one thread for each matrix, the same order, the same eps placement (`m / (sum + eps)`). Its input is the softmax output plus eps. This keeps the exact math of our path, thus the port uses this kernel.
    - The mlx-vlm kernel `hc_sinkhorn_split` also fuses the softmax, but it changes the math: it adds EPS to the softmax denominator and it uses `fast::exp`. It also needs `hc_mult == 4`. Thus it is not used.
    - Plan: put the kernel in a new file `Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift`. Move the op loop into an internal `DeepSeekV4Math` function so that a parity test can compare the two paths. `hcSplitSinkhorn` uses the kernel when the default device is the GPU, `iters >= 1`, and the matrix count is not zero; otherwise it uses the op path (the same gate as mlx-lm).
    - eps goes to the kernel as the bit pattern of the float32 in an `Int` template argument, thus the kernel needs no extra input array and eps is exact.
  timestamp: 2026-10-12T21:09:40.104429+00:00
- actor: claude-code
  id: 01m4v35vpbjatyfncebyvveq8x
  text: |-
    ### Implementation — fused Sinkhorn + collapse kernel
    - New file `Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift`: one `MLXFast.metalKernel` dispatch for each collapse. One threadgroup (256 threads) for each token. Thread 0 does the Sinkhorn loop (the order of the mlx-lm `deepseek_v41_sinkhorn`). All threads do the weighted sum of the copies by `pre` (float4 when the width is a multiple of 4, else scalar). Template args: HC, ITERS, D, THREADS, EPS_BITS (the float32 bit pattern of `hc_eps`, thus exact).
    - On coordinator instruction, the collapse (weighted sum) is fused into the same kernel. The `comb @ residual` orientation of `expand` is not changed.
    - The softmax and the sigmoids stay on the op path (`DeepSeekV4Math.hcSplitFields`, split out of `hcSplitSinkhorn`). The mlx-vlm kernel changes that math (EPS in the softmax denominator, `fast::exp`), thus it was not copied.
    - `DeepSeekV4HyperConnection.collapse` uses the kernel when the default device is the GPU, `iters >= 1` and the stream is not empty. Otherwise it uses the old op path (`hcSplitSinkhorn` + `ManifoldStream.weightedSum`). `ManifoldStream` is now internal so that the parity test can read the op path.
    - Discovery: the first version moved the weighted sum by 1 ULP, because the Metal compiler fused the multiply and the add into an `fma`. That 1 ULP changed the greedy text of the thinking-mode 4k cache check. `#pragma METAL fp contract(off)` in the kernel header (it must end in a newline, because MLX writes the signature directly after the header) makes the kernel bit-exact. The parity test now asserts exact equality and fails without the pragma.

    ### Tests
    - `DeepSeekV4HyperConnectionTests.theFusedSinkhornKernelAgreesWithTheOpPath` (3 cases: hc 4/20 steps/width 64, hc 3/1 step/width 6, hc 3/7 steps/width 8): bit-exact against `DeepSeekV4Math.sinkhorn` + `ManifoldStream.weightedSum`.
    - `theCollapseOnTheCPUAgreesWithThePythonReference`: the CPU device takes the op path and matches the Python fixtures.
    - Full MLXLMTests: 716 XCTest + 1364 Swift Testing, 0 failures. The only build warning is the SwiftPM "missing creator for mutated node ... mlx-swift_Cmlx.bundle", which is not from this change.

    ### Real weights, after this change — 2026-10-13, GPU idle (0 % before the run)
    Step bench (`dsv4bench <snapshot> step`), one decode step:
    - @128: A direct sync 53.0 / 53.3 ms (before 104), C generate 59.4 / 58.3 ms (before 126)
    - @4k: A 56.7 / 55.4 ms (before 109-110), C 59.2 / 56.5 ms (before 110-118)
    - @16k: A 68.4 / 67.7 ms (before 126-135), C 68.5 / 73.4 ms (before 123-129)
    Thus about 1.9x faster decode for each step.
    - Greedy check (`dsv4bench <snapshot> 32 64`): every text line and every token count is the same as the run on HEAD 4041c5b (the 64-token text, "Gull." in the 4k chat check, and the thinking-mode turn-2 texts). Note: on HEAD the thinking-mode 4k check already gives warm != cold (first difference at char 55). That is not from this change.
    - The full bench (`32,2048,8192,32768 128 8192 1024`) was not run again.
  timestamp: 2026-10-13T16:57:50.027703+00:00
- actor: claude-code
  id: 01m4v365h1hw5pch60bfjfykzq
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLLM/Models/DeepSeekV4SinkhornKernel.swift (new), Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift, Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift, Tests/MLXLMTests/DeepSeekV4HyperConnectionTests.swift. `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 716 XCTest + 1364 Swift Testing, 0 failures. Step bench @128/4k/16k A: 53/56/68 ms (before 104/109/126-135). Greedy text identical to HEAD.
    - next: /review. ARCHITECTURE.md does not exist, thus no update is necessary.
  timestamp: 2026-10-13T16:58:00.097461+00:00
- actor: claude-code
  id: 01m4v3gfg4xvbgp1emn8v4667n
  text: |-
    ### test — green
    - evidence: `swift build --build-tests` exit 0; `xcrun xctest` on all 5 bundles, exit 0 each. CXGrammarTests 7 Swift Testing; MLXFoundationModelsTests 367; MLXGuidedGenerationTests 70; MLXHuggingFaceMacrosTests 5 XCTest; MLXLMTests 716 XCTest + 1364 Swift Testing. 0 failures, 0 skipped. `xcodebuild build-for-testing` (IntegrationTesting, macOS): TEST BUILD SUCCEEDED, 0 compiler warnings, 0 errors (the changed files compiled there).
    - warnings: only the old SwiftPM "missing creator for mutated node" (mlx-swift_Cmlx.bundle). No Metal kernel warning in test output. SwiftPM compiled 0 files in this run (cache), so the compiler check of the changed files comes from xcodebuild.
    - next: review.
  timestamp: 2026-10-13T17:03:37.988962+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 speed: fused Metal kernel for the hyper-connection Sinkhorn steps'
---
## What

The hyper-connection Sinkhorn normalization runs as many small MLX ops. With `hc_sinkhorn_iters: 20`, each hyper-connection runs about 120 kernels. DeepSeek-V4-Flash has 86 hyper-connections, thus about 10,000 kernel dispatches for each decoded token.

Our code: `Libraries/MLXLLM/Models/DeepSeekV4MathHelpers.swift:89-132` (the Sinkhorn loop at line 126), called from `DeepSeekV4HyperConnection.swift:255`.

## Evidence from other ports

- ml-explore/mlx-lm PR #1189: the fused Sinkhorn kernel took decode on an M3 Ultra (4-bit) from 11 to 20 tok/s.
- mlx-lm PR #1192: the changes that include Sinkhorn fusion took decode from 22.4 to 26.7 tok/s.
- Reference kernels: `deepseek_v41_sinkhorn` in mlx-lm main `mlx_lm/models/deepseek_v41.py:150-216` (merged PR #1895), and `hc_sinkhorn_split` / `hc_sinkhorn_collapse` in Blaizzy/mlx-vlm `mlx_vlm/models/deepseek_v4/hyper_connection.py`.
- This repo already uses `MLXFast.metalKernel` in `Libraries/MLXLMCommon/TurboQuantKernels.swift`.

## Acceptance Criteria

- [ ] One Metal kernel (or a small fixed number) does all Sinkhorn iterations of one hyper-connection.
- [ ] A parity test shows that the kernel gives the same result as the current op path, within float32 tolerance.
- [ ] The existing DeepSeek-V4 unit tests pass.
- [ ] A decode-step measurement on the real checkpoint, with the GPU otherwise idle, shows the speed before and after.
#deepseek-v4 #performance