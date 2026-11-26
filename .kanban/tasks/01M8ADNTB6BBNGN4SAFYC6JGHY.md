---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8cdhkzb1hnv0bckyqbnsc4q
  text: |-
    ### Cause found: not the tie-break. The lanes' free-run logits are not bit-identical at the "tie" steps.

    **MLX argMax already picks the lowest tied id.**
    - The Metal kernel (`Cmlx/mlx-generated/metal/arg_reduce.metal`, `ArgMax`) has a strict `>` in `reduce_many`, so the earliest index wins inside a thread. `reduce` uses `best.val == current.val && best.index > current.index`, so the lower index wins in the SIMD and threadgroup reductions. It does not depend on the element type.
    - New `ArgMaxSamplerTieTests` (MLXLMTests): bf16, f16 and f32. Layouts [V], [1,V] and [1,1,V]. The float32 copy of a bf16 row, a `RemoteLogits` round trip, and the strided last-position slice of a prefill. Ties at 17=18, 40172=38, 4096=5 (the higher id reaches the lane reduction first), 248319=1, and a three-way tie, with V=248320. Result: 45/45 cases pass on the current `ArgMaxSampler`, so RED never happened.
    - New `GuidedGreedyTieTests` (MLXGuidedGenerationTests): `applyMaskAndSample` (the guided loop uses bare `argMax`, not `ArgMaxSampler`) with a float32 grammar mask plus a bias over a bf16/f16/f32 row picks 17 at a 17=18 tie. 3/3 pass.
    - A throwaway sweep (deleted afterwards) tried 1080 random tie pairs over V in {248320, 151936, 200064, 4097, 1025, 33}, each in 3 dtypes x 3 layouts. 0 failures.

    **The card's premise about RemoteLogits is wrong.** `RemoteLogits` keeps the element type (`RemoteCacheStatus.swift`: bf16 stays bf16). Only the parity suite's own rows are float32 copies (`TeacherForcing.lastRow`), and that conversion keeps ties exactly.

    **What the parity reports show** (`FoundationModelsCluster/.build/parity-reports`):
    - Qwen guided: teacher-forced (TF) local = TF one-rank bit for bit (KL 0 on all 105 steps). TF step 44 has an exact tie 17=18 (margin 0). The free-run one-rank lane picked 17 (nights 2) and the free-run local lane picked 18 (nights 3). Both lanes run the same `GuidedGenerationLoop` and the same deterministic argmax. So the two free runs did not see the same logits at that step, though their TF recomputations agree.
    - Inkling guided step 49: the TF one-rank row ties 38=40172 exactly, yet the free-run one-rank lane picked 40172, the *higher* id. The TF cluster row has 40172 ahead (-0.835 vs -0.979), yet the free-run cluster picked 38. Neither pick is possible on those logits with a lowest-id argmax. So in both lanes the free-run logits differ from the TF logits the suite measures ties on. The one-rank vs cluster pair also differs by sharding (KL 0.042 at that step).

    **Conclusion:** no deterministic tie-break can make these runs agree. The divergence comes from logits that differ between the free-running decode and the TF recomputation, and between lanes.

    Candidates for that difference, not verified because m5u1/m5u2 are off-limits for model runs:
    - run-to-run GPU nondeterminism (compare e8b5ea42, the gated-delta stream race that made Inkling parity flaky);
    - the free run's prefill remainder going through `callAsFunction` (`.allPositions`) while TF uses `nextTokenLogits` (`.lastPosition`);
    - the closing/whitespace bias, which the free run adds and the TF rows leave out (Inkling 38/40172).

    The next step is a parity-suite change: record each lane's free-run logits at the decision step, not only TF recomputations. That shows which lane's logits moved.
  timestamp: 2026-11-26T17:13:11.915159+00:00
- actor: claude-code
  id: 01m8ce0t5ady7029ypqk0yvmat
  text: |-
    ### implement — stuck
    - evidence: the cause is not the tie-break. MLX argMax (Metal `arg_reduce.metal`) already picks the lowest tied id for bf16, f16 and f32, in every layout, and for RemoteLogits too, which keep the element type and are not float32 copies. The parity reports show free-run picks that a lowest-id argmax cannot make on the teacher-forced logits: Inkling one-rank picked 40172 at an exact 38=40172 tie, and Qwen local picked 18 where TF local = TF one-rank bit for bit with an exact 17=18 tie. So the lanes' free-run logits are not bit-identical, and no sampler change can help. Production code is unchanged.
    - Added 2 test files, swift-formatted, and left uncommitted:
      - Tests/MLXLMTests/ArgMaxSamplerTieTests.swift (5 tests, 45 cases)
      - Tests/MLXGuidedGenerationTests/GuidedGreedyTieTests.swift (1 test, 3 cases)
      - Both pass on the unchanged code, so they guard the lowest-id contract and are not a TDD red.
    - Tests: `swift build --build-tests`, then xctest.
      - MLXLMTests: XCTest 1261, 0 failures; Swift Testing 2754 passed, with 3 known issues that were already there (AttentionKernelToleranceTests, MiniCPM5 CDATA).
      - MLXFoundationModelsTests: 618 passed.
      - MLXGuidedGenerationTests: 96 passed.
    - next: a person decides. Option 1: change the parity suite (FoundationModelsCluster) to record each lane's free-run logits at the decision step, which finds the lane whose logits moved (GPU nondeterminism, the prefill-remainder `.allPositions` vs `.lastPosition` path, or the bias that TF rows leave out). Option 2: close this card as not a sampler bug. Keep or drop the two guard tests.
  timestamp: 2026-11-26T17:21:29.770578+00:00
- actor: claude-code
  id: 01m8ce1ms4w6vc7yy7dyx0jdbg
  text: |-
    ### finish iteration 1 — stuck
    - implement: stuck — not a sampler bug. MLX argMax already picks the lowest tied id (bf16/f16/f32, all shapes, RemoteLogits keep the worker dtype, guided mask+bias). The parity reports show free-run picks that the teacher-forced logits can't produce (Inkling one-rank chose 40172 at an exact 38=40172 tie; Qwen local chose 18 where local/one-rank recomputations are bit-identical with 17=18), so the free-run logits differ from the teacher-forced ones and between lanes.
    - guard tests added (green on unchanged code): Tests/MLXLMTests/ArgMaxSamplerTieTests.swift, Tests/MLXGuidedGenerationTests/GuidedGreedyTieTests.swift; committed separately as test-only.
    - next (human decision): have the FoundationModelsCluster parity suite log each lane's free-running logits at each decision to find which moved (suspects: run-to-run GPU nondeterminism, prefill last-chunk path in free run vs recomputation, the closing/whitespace bias absent from teacher-forced rows), or close as not a sampler bug.
  timestamp: 2026-11-26T17:21:57.028853+00:00
position_column: doing
position_ordinal: '80'
title: Greedy argmax breaks exact bf16 ties differently on the local and remote paths
---
Found by the FoundationModelsCluster parity suite (pne5cfr, 2026-11-25). Exact bf16 logit ties are common, and greedy decoding picks a different tied token depending on the path:
- Qwen3.8 guided, step 44: tokens 17 and 18 have margin 0.0, bit-identical on the local and one-rank lanes. Local greedy picked 18; the worker path picked 17.
- Inkling guided, step 49: tokens 38 and 40172 both at logprob -0.813. One rank picked 40172; the cluster picked 38.

Correctness doesn't need this: parity passes under vLLM's check_logprobs_close, since both tied tokens are in both top-5s. But local and clustered greedy runs would agree token for token if the greedy sampler broke exact ties deterministically, for example by the lowest token id, whatever the logits' dtype, shape or path (the local logits vs the float32 RemoteLogits copy).

To do: find why argMax differs on equal values (dtype conversion before the argmax? different reductions?), make greedy tie-breaking deterministic, and test it with a logits row that holds an exact tie in bf16 and in float32. Low priority. #cluster