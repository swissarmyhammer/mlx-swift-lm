---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4x4ww10c3j8hcdkrn62qq87
  text: |-
    Baseline at HEAD 8793212 (GPU idle, dsv4bench `step 128,16384`, 32 steps, 2 rounds). A = direct sync, B = direct pipelined, C = library generate, ms/token.
    - ctx 128: A 35.4 / 35.2, B 34.0 / 34.1, C 33.2 / 32.8.
    - ctx 16384: A 38.0 / 38.6, B 37.2 / 37.5, C 42.2 / 39.9.
  timestamp: 2026-10-14T12:06:21.472004+00:00
- actor: claude-code
  id: 01m4x5dypjfm80bw1ck11r29ke
  text: |-
    Prototype and measurement. Result: no gain. The code change is reverted.

    What I tried: one `CompiledDecodeSegmentCache<DeepSeekV4ModelInner>` with one trace for each layer boundary (42 traces). Each trace ran `layerTail` of layer i and `attentionInput` of layer i+1, with the weights of both layers as compile state. Layer 0 kept its head trace and layer 42 kept its tail trace. Thus a step made 44 trace calls, not 86. The attend stage (cache write) stayed outside the traces. An environment switch selected the path, so that one build measured both paths. The first decode step took 340 ms (baseline 80 ms), which shows that the new traces compiled.

    Measurement (GPU idle, dsv4bench `step 128,16384`, 32 steps, 2 rounds, ms/token; A = direct sync, B = direct pipelined, C = library generate):
    - Before, ctx 128: A 35.4 / 35.2, B 34.0 / 34.1, C 33.2 / 32.8.
    - After, ctx 128: A 35.3 / 35.7, B 34.0 / 34.2, C 33.2 / 32.6.
    - Before, ctx 16384: A 38.0 / 38.6, B 37.2 / 37.5, C 42.2 / 39.9.
    - After, ctx 16384: A 38.7 / 38.2, B 37.3 / 37.4, C 42.8 / 43.3.

    All differences are in the run-to-run noise (about 0.5 ms). C at 16k is not faster. Possible cause: the cost of `compile_replace` follows the number of tape nodes it replaces, not the number of calls. Two traces of one layer hold the same nodes as one fused trace across a boundary. The fusion of the expand of layer i with the collapse of layer i+1 removes only a few small kernels in each layer. Do not try this again without a new reason.

    Acceptance:
    - [x] Decode-step measurement before and after, at 128 and 16k: above.
    - [x] Parity tests of ^rnnxh8d: the tree is HEAD 8793212 again. `swift build --build-tests` and `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`: 1408 tests in 111 suites passed. `DeepSeekV4CompiledDecodeTests`: 9 tests passed.
    - [x] Greedy ids: no library code changed, thus the ids cannot change. I did not run the integration test again.

    The bench binary is rebuilt at HEAD.
  timestamp: 2026-10-14T12:15:41.266512+00:00
- actor: claude-code
  id: 01m4x5e31h7rwgt3q0v7ks2rcn
  text: |-
    ### implement — no-change
    - evidence: fused boundary trace prototype measured and reverted. ms/token before -> after: ctx 128 B 34.0/34.1 -> 34.0/34.2, C 33.2/32.8 -> 33.2/32.6; ctx 16384 B 37.2/37.5 -> 37.3/37.4, C 42.2/39.9 -> 42.8/43.3. Tree = HEAD 8793212. xctest: 1408 tests in 111 suites passed.
    - next: the measurement is the deliverable. A person decides whether to close the card.
  timestamp: 2026-10-14T12:15:45.713639+00:00
- actor: claude-code
  id: 01m4x5fs1s3q36dpk5kf25gdsy
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (675e223): 0 findings, 0 confirmed, 0 refuted. 0 files reviewed. 2 files not reviewed because no validator matches them: .kanban/tasks/01M4V88GSVC1BBY9QFR4DX93AF.jsonl, .kanban/tasks/01M4V88GSVC1BBY9QFR4DX93AF.md. The commit changes no library code. The task description has no prior Review Findings section.
    - next: none. The task is in done.
  timestamp: 2026-10-14T12:16:41.017403+00:00
- actor: claude-code
  id: 01m4x5g3vbbw3v0y4j5wrsrgac
  text: |-
    ### finish iteration 1 — clean
    - implement: no-change — fused boundary trace measured, no gain, code removed (ctx 128 generate 33.2/32.8 -> 33.2/32.6; ctx 16k generate 42.2/39.9 -> 42.8/43.3 ms/token)
    - test: green — xctest 1408 Swift Testing tests in 111 suites passed at HEAD; DeepSeekV4CompiledDecodeTests 9 passed
    - commit: 675e223 (task record only)
    - review: clean — 0 findings
  timestamp: 2026-10-14T12:16:52.075607+00:00
position_column: done
position_ordinal: ffc480
title: 'DeepSeek-V4 speed: one compiled trace from the tail of a layer to the head of the next layer'
---
## What

Task ^rnnxh8d runs each DeepSeek-V4 decode step through two compiled traces for each layer (`DeepSeekV4DecoderLayer.decodeStep` in `Libraries/MLXLLM/Models/DeepSeekV4.swift`): one before the cache stage and one after it. That is 86 trace calls for each step.

A Time Profiler run of 21 decode steps at 128 context (scratch `prof2/tp128.trace` of ^rnnxh8d) shows `mlx::core::detail::compile_replace` at 46 ms, about 2.2 ms for each step. Each call also passes every weight of the layer as a compile input.

Evaluate one trace that runs the tail of layer i and the head of layer i+1 (the pattern `CompiledDecodeSegment` of `Libraries/MLXLMCommon/CompiledDecodeSegments.swift` gives Qwen3.5). That halves the trace calls and lets the expand of layer i fuse with the collapse of layer i+1. Measure it before keeping it.

## Acceptance Criteria

- [x] A decode-step measurement on the real checkpoint, GPU idle, at 128 and 16k context, before and after.
- [x] The parity tests of ^rnnxh8d (same bits as the op path, float32 and bfloat16) still pass.
- [x] Greedy ids on the mlx-vlm fixture do not change.
#deepseek-v4 #performance