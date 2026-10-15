---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4zwxkvc3emwhnbhc1f5cyhm
  text: |-
    Research (dot mode at HEAD 94f3563, scratchpad/gdnfuse/base.decode.dot, 8k context): the decode graph has 1446 kernel-capable nodes (80 CustomKernel = 48 GDN + 32 decode attention). One GDN layer runs these non-matmul kernels: conv taps (1 compiled kernel), Concatenate (conv state), SiLU (CompiledSigmoidMultiply), q RMSNorm, q scale, k RMSNorm, k scale, decay gate (compiled), beta sigmoid (compiled), AsType, gated_delta_step, gated RMSNorm, gate multiply (compiled). That is 13 kernels for each layer, 624 for each token.

    Timing (new bench mode `qwen38bench gdn 48 N`, scratchpad GdnBench.swift / GdnFused.swift): 48 GDN layer bodies at real shapes, chained by the conv state, the recurrent state and the layer input (as in the model). Op path (compiled, as in the decode segments): 2.00-2.05 ms for 48 layers. State traffic floor (f32 state read + write only): 0.71-0.92 ms. Without the input link the op path is 1.67 ms, because the conv/norm chain of the next layer then overlaps the delta kernel; the model cannot do that, so the linked chain is the correct one.

    Prototypes: (1) one kernel up to y, then MLX rmsNorm + gate: 1.34-1.52 ms. (2) one kernel for each head (1024 threads, 32 SIMD groups, 4 value rows each) that also does the gated RMSNorm: 1.24-1.35 ms (most runs). Prefetch of the state rows before the conv did not help (1.39-1.74 ms). Expected gain: about 0.6-0.7 ms for each token (about 2 %). The card target of 2-3 ms is not possible: the GDN non-matmul work in the compiled decode step is about 2 ms, not 4.1 ms (the 4.1 ms of PROFILE.md came from the unfused split mode).

    Precision: with the sigmoid and softplus done in the input type (as the compiled MLX kernels do), prototype (2) is bit-exact with the op path on random bf16 input (y, conv state and recurrent state max |diff| = 0).

    GPU clock noise: the same configuration gives 1.3 or 1.7 ms in different runs; compare medians of several runs.
  timestamp: 2026-10-15T13:44:40.556219+00:00
- actor: claude-code
  id: 01m5001pf3df75d2jcjnvrf486
  text: |-
    Implementation landed (not committed).

    Files:
    - Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift (new): `GatedDeltaDecodeKernel`, one Metal kernel for the body of one GDN layer on one decode token: conv taps + SiLU, conv state shift, q/k RMSNorm and scale, beta and decay gates, delta rule, gated RMSNorm. One threadgroup for each value head (32 SIMD groups, 4 value rows each at the Qwen3.8 shape). `canRun` refuses other shapes and dtypes (op path stays).
    - Libraries/MLXLMCommon/GatedDelta.swift: the delta-rule row step is now the shared Metal helper `gated_delta_row` (header `gatedDeltaMetalHeader`), used by `gated_delta_step` and by the new kernel. Same operations in the same order.
    - Libraries/MLXLLM/Models/Qwen35.swift: `forward` calls `decodeKernelStep` when there is no mask; the op path moved to `recurrentBody` (returns the input of `out_proj`). `queryKeyNormEps` and `queryKeyScales` name the q/k constants that both paths read.
    - Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift (new): tolerance test kernel vs `recurrentBody` (Qwen3.8 shape and a 2-sequence small shape, bf16 and f16; output and recurrent state inside 2 steps of the dtype, conv state bit-equal), positive control (no recurrent state / no conv state moves the output past the tolerance), `canRun` refusals, training keeps the op path, `forward` routes one token through the kernel.

    Mutation check: a kernel that skipped the conv state rows failed 8 expectations of the tolerance test; restored.

    Precision: the sigmoid and softplus run in the input type (as the compiled MLX kernels). On random bf16 input the prototype was bit-exact with the op path. The 32k chat outputs of the bench are the same text as before the change.

    Measurements (qwen38bench, 32k, M3 Ultra; background load each time: fseventsd ~100 %, sourcekit-lsp ~89 %, SourceKitService 23-93 %, once another `sah` at 2235 % during step run 2):
    - dot (8192): 1446 -> 870 kernel-capable nodes per decode token (-576 = 12 x 48). RMSNorm 305 -> 161; Concatenate, AsType and the GDN compiled elementwise kernels gone.
    - profile, same-session A/B, interleaved: base B pipelined 29.97 / 29.81 ms, new 29.01 / 29.05 ms (-0.86 ms, -2.9 %). Sync A 31.80 / 31.50 -> 29.96 / 30.08 ms. Graph build 2.02 / 1.92 -> 1.36 / 1.35 ms. Prefill 398.4-399.0 tok/s both.
    - profile (new) runs 1-2: B 29.06 / 29.29 ms, prefill 398.8 / 398.5 tok/s, peak 20.9 GiB.
    - step (new) runs 1-2: B 29.00 / 29.12 / 29.12 / 29.03 ms, C generate 28.70 / 29.87 / 29.01 / 28.70 ms (previous after ^pqdshah: B 29.74-29.99, C 29.32-29.38).
    - full (new, QWEN38_THINKING_KEY=enable_thinking) runs 1-2: 128-token decode 28.77 / 35.07 ms (run 2 one-time transient; its chat turns decode at 28.3-29.5 ms, the same as run 1); chat-turn decode 28.3-29.2 ms vs 29.1-30.1 ms before; prefill 396.1 / 395.1 tok/s; peak 20.7 GiB; reuse check PASS, warm == cold YES / tie-aware PASS, thinking off and on.

    Card target (2-3 ms) not reached: the GDN non-matmul work in the compiled decode step was about 2.0 ms for 48 layers, not 4.1 ms (the 4.1 ms of PROFILE.md came from the unfused split mode). The state traffic floor is about 0.7-0.9 ms, thus about 1 ms was the most to gain; the kernel gets about 0.9 ms.

    Tests: `swift build --build-tests` clean (only the SwiftPM notice "missing creator for mutated node" for the Cmlx bundle, not a compiler warning); `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: XCTest 716 tests 0 failures, Swift Testing 1488 tests in 120 suites passed, 0 skipped. Real weights: `xcodebuild test -only-testing:IntegrationTestingTests/Qwen35SessionPromptCacheTests -only-testing:IntegrationTestingTests/Qwen35CheckpointIntegrationTests`: 7 tests in 2 suites passed.

    Follow-up: ^znzpvzb (use the kernel in Qwen3Next and in the MLXVLM Qwen 3.5 GDN layer).

    Logs: scratchpad/gdnfuse/ (ab-profile-*, profile-*, step-*, full-*, base/new .decode.dot, xctest-full.log, integ.log).
  timestamp: 2026-10-15T14:39:20.035645+00:00
- actor: claude-code
  id: 01m50021n0rzcmkbxg2ssmwtps
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift (new), Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLLM/Models/Qwen35.swift, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift (new). Decode at 32k, same-session A/B profile: 29.89 -> 29.03 ms/token (-0.86 ms); dispatches 1446 -> 870 per token; prefill and peak memory unchanged; reuse check PASS. MLXLMTests: 716 XCTest + 1488 Swift Testing, 0 failures, 0 skipped; Qwen35 real-weights integration 7/7 passed.
    - next: /review. Card target 2-3 ms not reached (about 1 ms was the most to gain, see the comment above). Follow-up ^znzpvzb.
  timestamp: 2026-10-15T14:39:31.488236+00:00
- actor: claude-code
  id: 01m5007sws2etyfr3bm3qcmk8y
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (a6c0dac). 1 finding, 1 confirmed, 0 refuted. Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift:496 code-hygiene/magic-numbers-swift.
    - next: Replace the magic numbers with named constants in all of GatedDeltaDecodeKernel.swift. Then run the review again. The task stays in review.
  timestamp: 2026-10-15T14:42:40.153489+00:00
- actor: claude-code
  id: 01m500869ev99tz7pc4gmt4qxx
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 4 files; 32k decode 29.89 → 29.03 ms/token, kernels per token 1446 → 870
    - test: green — MLXLMTests 716 XCTest and 1488 Swift Testing, 0 failures, 0 warnings; Qwen35 integration 7/7
    - commit: a6c0dac
    - review: findings — Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift:496
  timestamp: 2026-10-15T14:42:52.846543+00:00
- actor: claude-code
  id: 01m500t417jh3rnw3rj5e9qgyg
  text: |-
    Fix of the finding `code-hygiene/magic-numbers-swift` (pure refactor, same values):

    - The rule runs `swiftlint no_magic_numbers` (allowed: 0, 1, -1, 100). Before the fix it found 37 violations in GatedDeltaDecodeKernel.swift (1) and GatedDelta.swift (36). After the fix: 0 in the 5 changed files.
    - New Libraries/MLXLMCommon/MetalSimdWidth.swift: `metalSimdWidth = 32`, the one shared SIMD width. It replaces `gatedDeltaSimdWidth` (GatedDelta.swift) and the private `DecodeAttentionKernel.simdWidth`. Both kernels write it into their Metal source as `SIMD_WIDTH`. New file in MLXLMCommon; no ARCHITECTURE.md exists.
    - GatedDeltaDecodeKernel.swift: `Output` enum names the output places (the finding line). `mostSimdGroups = mostThreadgroupThreads (1024) / metalSimdWidth`, `mostThreadgroupNumbers = mostThreadgroupBytes (32_768) / MemoryLayout<Float>.size`. Metal: `constexpr int STATE_ROWS = TAPS - 1` replaces the four `TAPS - 1`.
    - GatedDelta.swift: `GatedDeltaAxis` (batch, step, head, width), `GatedDeltaStateAxis` (head, valueRow, keyNumber), `GatedDeltaStepRank` (perSequence, perHead, perHeadNumber), `GatedDeltaRecompute.Input` (input places), `gatedDeltaStepSimdGroups = 4` (threadgroup of `gated_delta_step`); grid width and the `Dk % 32` check use `metalSimdWidth`. The `axis: -2` operands are now the positive axes of the same rank (`repeated(..., axis: 2)` on 4-D q/k; `expandedDimensions(..., axis: 2)` on 3-D k/q/g), which give the same shapes.
    - Qwen35GDNDecodeKernelTests.swift: the numbers of the JSON configuration have names (attentionHeads, attentionHeadDim, vocabularySize, fullAttentionInterval). swiftlint found no violation in this file before.
    - swift-format reformatted the `header` string of GatedDeltaDecodeKernel.swift (indentation of the multi-line literal only; the string content does not change) and one `guard` line.
  timestamp: 2026-10-15T14:52:40.359607+00:00
- actor: claude-code
  id: 01m500t9aqeeg6vbsyxc8pnzqm
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/MetalSimdWidth.swift (new), Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift, Libraries/MLXLMCommon/GatedDelta.swift, Libraries/MLXLMCommon/DecodeAttentionKernel.swift, Tests/MLXLMTests/Qwen35GDNDecodeKernelTests.swift. swiftlint no_magic_numbers: 37 -> 0. `swift build --build-tests`: 0 compiler warnings (only the SwiftPM notice "missing creator for mutated node"). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: XCTest 716 tests 0 failures; Swift Testing 1488 tests in 120 suites passed. Real weights (xcodebuild -only-testing Qwen35SessionPromptCacheTests + Qwen35CheckpointIntegrationTests): 7 tests in 2 suites passed. Finding flipped to [x]. Not committed.
    - next: /review
  timestamp: 2026-10-15T14:52:45.783443+00:00
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 decode speed: fuse the small GDN kernels of each linear-attention layer'
---
## What

At 32k, one decode token spends about 4.1 ms in the Gated DeltaNet (GDN) work that is not a matmul, over 48 layers. The token runs 1430 kernels, 416 of them compiled elementwise kernels. The GPU is busy 100 % of the time, so each small kernel costs time.

## Approach

1. Make a list of the kernels of one GDN layer in decode (conv state update, conv, SiLU, q/k norm and scale, beta sigmoid, decay gate, `gated_delta_step`, gated RMSNorm, output gate).
2. Put as many as possible into one Metal kernel (or into `gated_delta_step`), so that a GDN layer has the fused input projection, one kernel, and the output projection. Target: 2 to 3 ms less for each token.
3. Keep the prefill path as it is, unless the same kernel is also faster there.

## Measure

scratchpad/qwen38bench at 32k: profile and step modes for decode (two runs each, idle GPU), the dispatch count (dot mode), and full mode for prefill, peak memory and the reuse check.

## Acceptance Criteria

- [x] Decode at 32k is faster, measured.
- [x] A tolerance test compares the fused kernel with the op path, with a positive control.
- [x] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance

## Review Findings (2026-10-15 09:40)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 4 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M4ZD6HSN7V15THJC5Q0950ZV.jsonl` — no validator matches this file
> - `.kanban/tasks/01M4ZD6HSN7V15THJC5Q0950ZV.md` — no validator matches this file
> - `.kanban/tasks/01M5000W20Q4CW37GXFZNZPVZB.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5000W20Q4CW37GXFZNZPVZB.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/GatedDeltaDecodeKernel.swift:496` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
