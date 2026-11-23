---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84drb71yg17sww2b64t8pcs
  text: |-
    ## Feasibility notes (mlx-swift 0.33.0)
    - `MLXFast.scaledDotProductAttention` takes `sinks`, but as one logit **per head** (`[H]`). Inkling's sink is per query per head (`[B, L, Hq]`, from `sinkProj`), and SDPA returns no LSE.
    - Far part: pass `keys[..<S-1024]` with `mask: .causal`. MLX aligns causal masks bottom-right, so query i sees keys ≤ S−L+i−1024, exactly the distance-≥1024 set. Rows with no far keys need a sink, or they produce NaN.
    - To recover the far denominator without an LSE: give SDPA a per-head sink c, and V an extra channel of ones (head dim padded 128→256). Then O_extra = D_far/(D_far+e^c), so D_far = e^c·O_extra/(1−O_extra). That cancels badly when D_far ≫ e^c, and the padding doubles the PV work and the V reads.
    - Cleaner options: (a) an upstream MLX change returning the LSE from the steel attention kernel; (b) port the steel/NAX tiling into `relativeAttentionSource`.
    - Either way, the summation order on long prompts changes. Any golden divergence must be shown to be a near tie.
    - Upside: SDPA is 22 ms against ~78 ms per 2048×64k chunk, about +40% prefill at 64k. Left open for the user to choose an approach.
  timestamp: 2026-11-23T14:42:56.865189+00:00
- actor: claude-code
  id: 01m84qzzvcvk3s0se38w30r9zz
  text: |-
    ## Implemented as option (c): the relative attention on the neural accelerators (NAX)
    SDPA with an LSE was not reachable: MLX only emits the LSE under `in_grad_tracing()`, as a sibling output the Swift API does not expose, and the ones-column trick loses the denominator in a bf16 output. Instead:
    - **A custom `MLXFast.metalKernel` can use Metal 4 tensor ops.** It includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and calls `mpp::tensor_ops::matmul2d` (16×32×16, execution_simdgroup). Proven with `inklingcli naxtest`: max |diff| 0.0 against `matmul`.
    - `relativeAttentionNAXSource` follows MLX's `attention_nax` layout:
      - 64 query rows per threadgroup, as 4 simdgroups of 16 rows; 32-key blocks.
      - QKᵀ and PV on `matmul2d`; Q fragments re-read per block; scores, probabilities and output in registers. No threadgroup memory, so each simdgroup walks only its own causal key range.
      - Inkling's own semantics: per-row sinks, the relative bias at d < R, the window, key start, keep/evicted.
      - A "far" fast path for blocks whose every key exists and lies past the bias extent of all 16 rows: scale only, with unguarded vec4 loads.
    - `inkling2HasNeuralAccelerators` mirrors MLX's `is_nax_available`: macOS 26.2+ and an Apple GPU of generation ≥17 (≥18 for `p` variants), read from `GPU.deviceInfo().architecture`. Other GPUs, and head dims that are not a multiple of 32, keep the old kernel.
    - `inkling2RelativeAttention(..., neuralAccelerators:)` defaults to that check. The unit test "Prefill attention matches the reference" runs every case on both kernels, and adds a 1100-cached-key case that takes the far path.

    ## Kernel microbenchmark (L=4096, 24 q heads, 8 kv heads, D=128, bf16)
    | S | old | NAX | speedup |
    |---|---|---|---|
    | 4096 | 6.7 ms | 3.8 ms | 1.8x |
    | 8192 (window 1024) | 3.4 ms | 2.3 ms | 1.5x |
    | 32768 | 84 ms | 39.5 ms | 2.1x |
    | 65536 | 174 ms | 83–88 ms | 2.0x |

    - Max |diff| vs the old kernel: 1e-4 to 8e-3, bf16 output rounding.
    - Simdgroups per threadgroup (2/4/8): flat, so 4 is kept.
    - MLX's own SDPA is still ~2x faster at this shape (~44 ms); there is more to find here.

    ## End to end (inklingbench vs the pushed e2debdb6)
    | ctx | prefill tok/s |
    |---|---|
    | 2k | 4778 → 5066 (+6%) |
    | 8k | 4927 → 5516 (+12%) |
    | 32k | 3781 → 4806 (**+27%**) |
    | 64k | 2850 → 4021 (**+41%**) |

    Decode is unchanged.

    ## Output
    The goldens differ at three near ties. `inklingcli gap`, top-2 logit gap at the divergence, old kernel → NAX:

    | prompt | gap |
    |---|---|
    | short@43 | 0.07 → 0.14 |
    | code@166 | 0.34 → 0.06 |
    | raw32768@53 | 0.00 → 0.17 |

    All are under the repo's 0.5 near-tie tolerance. The NAX kernel is at least as precise: it feeds the bf16 q and k to the matmul exactly, and keeps p in float rather than rounding it to half.

    ## Tests
    - MLXLMTests: 0 failures, including the 10 prefill-attention cases.
    - Inkling2CheckpointIntegrationTests and Inkling2MTPIntegrationTests: 7/7.
      - MTP equals plain over 160 tokens on 3/3 prompts.
      - MTP long block 3 went 1.23x → **1.43x**: acceptance rose 0.576 → 0.745 on the new prefill numerics.
      - Chat with MTP ~121 tok/s.
  timestamp: 2026-11-23T17:41:53.132940+00:00
position_column: done
position_ordinal: ffffec80
title: 'Inkling 2 prefill: full-attention far keys through MLXFast SDPA, near band through the relative kernel'
---
## Why
^4nfm0y0: at long contexts, prefill time is the full-attention relative kernel, at ~21 TFLOP/s. `MLXFast.scaledDotProductAttention` runs the same L=2048 × S=64k shape in 22 ms (~74 TFLOP/s), against ~78 ms. Estimated gain: +40% prefill at 64k, ~+20% at 32k.

## What
In a full layer the relative bias is zero past distance R=1024 (`rel_extent`).
- Far part: keys at distance ≥ 1024 have no bias. Run them through SDPA (causal with an offset, or an explicit key range).
- Near band: d < 1024 plus the sink, through the existing kernel, changed to also return each row's LSE.
- Merge the two by their log-sum-exps.

Open problem: MLX's SDPA returns no LSE. Options:
- (a) An upstream MLX change that exposes the LSE.
- (b) The sinks argument plus an extra value channel of ones, to recover the denominator.
- (c) Port the steel attention tiling (NAX) into the relative kernel.

## Acceptance
- [ ] The output matches the current kernel within bf16 tolerance. Any golden divergence must be shown to be a near tie.
- [ ] inklingbench shows a prefill gain at 32k and 64k. #inkling-perf