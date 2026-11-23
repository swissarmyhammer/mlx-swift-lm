---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84ajdn8b6nprwdebnjda3ca
  text: |-
    ## Implemented: compiled one-row decode, two traces per layer visit
    - `Inkling2Attention`:
      - `decodePrologue`: the projection plus the attention prologue, which reads no cache.
      - `decodeAttend`: `cache.update` plus `inkling2DecodeAttention`, kept outside the traces.
      - `decodeTau`: the log-length scaling as a `[1]` input array.
    - `Inkling2DecoderLayer.compiledDecode`, for a one-row decode, not a verify:
      - The `CompiledTrace` prologue covers the depth mix of the attention input, the message controllers and the attention prologue.
      - Then the attend step runs outside the traces.
      - The `CompiledTrace` epilogue covers o_proj, the depth mix of the MLP input, the controllers, the router, the experts and the convolution.
      - Separate traces with and without messages. The blocks, messages and tau are optional trailing arguments, fixed per layer.
    - `INK_COMPILE=0` turns it off for A/B. **Temporary; remove before the change lands.**
    - Made trace-safe: the constant cache (`Inkling2KernelConstants`) uses data-backed arrays with no eval; the stacked attention norms are built at `prepare()`.
    - The layer's `update` overrides call `invalidateCompiledTraces()`, and drop the stacked norms and controllers.

    ## Result vs stable base (includes the constant hoisting)
    - inklingcheck: CHECK PASS. The compile is bit-exact on all 4 goldens.
    - Profile at 2k: build 3.24 → **1.70 ms**, eval 10.29 → 9.72 ms per step.
    - GPU busy during decode: 68.7% → **83.1%**.

    | ctx | decode ms/tok base → now | tok/s |
    |---|---|---|
    | 2k | 12.610 → **10.597** | 79.3 → 94.4 (+19.0%) |
    | 8k | 12.639 → 10.701 | +18.1% |
    | 32k | 13.915 → 11.882 | +17.1% |
    | 64k | 15.726 → 13.885 | +13.3% |

    Prefill is unchanged, as expected.

    ## Follow-ups
    - Merge epilogue(i) and prologue(i+1) into one trace, so that each visit makes one compiled call. The GPU still idles ~17%.
    - Compile the verify (sequential) path for MTP.
  timestamp: 2026-11-23T13:47:17.032900+00:00
position_column: done
position_ordinal: ffffea80
title: 'Inkling 2 decode: compile the decode step in segments around the attention cache (CompiledTrace)'
---
## Why
Decode is CPU-bound (^xdrv99b): the GPU needs ~8.7 ms per token, but the main thread builds and encodes for ~12.6 ms. The Swift graph build and the per-call custom-kernel source rebuild (`write_signature` in mlx) are both skipped when a compiled trace replays. Qwen 3.5 got its decode gain from `CompiledTrace` / `CompiledDecodeSegment`.

## What
Per layer visit, the parts that do not touch the growing attention cache are pure functions of arrays and weights:
- **Pre:** depth mix of the attention input, message controllers, the fused q/k/v projection and the attention prologue. They give the q, relative and gate outputs, the new k and v, and the next key and value convolution states.
- **Outside the trace:** `cache.update` and `inkling2DecodeAttention`, whose shapes and template args change per step.
- **Post:** o_proj, the attention convolution, the depth mix of the MLP input, the message controllers, the router, the experts and the MLP convolution.

Compile Post(visit i) + Pre(visit i+1) as one `CompiledTrace` on the layer, with the weights as compile state. Pass the per-step scalars (tau) as arrays. Start with the non-sequential decode path (one row). A verify keeps the current path until the decode path is proven.

Step 1: prototype a compiled Post only (MoE side). Measure the CPU build time and ms/tok, to show that replay avoids the custom-kernel build cost.

## Acceptance
- [ ] inklingcheck.sh: CHECK PASS. Compile must not change the bits; the custom kernels are the same.
- [ ] inklingbench.sh shows a decode gain beyond the noise.
- [ ] The Inkling 2 unit tests pass, including verify-matches-decode. `invalidateCompiledTraces` covers the traces (LoRA and adapters). #inkling-perf