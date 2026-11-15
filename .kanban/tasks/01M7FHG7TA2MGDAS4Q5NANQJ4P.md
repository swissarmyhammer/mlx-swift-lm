---
assignees:
- claude-code
comments: []
position_column: todo
position_ordinal: '8380'
title: 'Flint: find and cut the cost of sliding-window attention in decode (~63 µs/layer)'
---
## Why
- **Measured:** skipping attention in the 35 sliding layers (window 513, RotatingKVCache) saves ~2.2 ms/token on oQ4e, about 63 µs per layer. The 14 full layers cost ~69 µs each even at 2k context.
- **Expected:** the bytes read are tiny, so most of this must be dispatch overhead: q/k/v projections, q and k RMSNorm, two RoPE calls, the rotating-cache update, the sinks cast, SDPA with sinks, and `o_proj`.

## What
1. **Measure the parts.** Do a micro-ablation or a shader-profiler trace of one layer:
   - the RotatingKVCache update at L=1 (slice or concatenate ops? temporaries?);
   - whether `MLXFast.scaledDotProductAttention` with `sinks` takes the fused vector kernel, or falls back to unfused attention for head dim 128 / GQA 8;
   - the norm and RoPE kernels.
2. **Fix what dominates.** Candidates:
   - Fuse q/k RMSNorm, RoPE and the KV write into one Metal kernel, like the Qwen3.8 decode kernels (`94f35637`, `a6c0dacc`).
   - Write the rotating cache in place.
   - Give SDPA sinks support in the fused kernel if it's missing.

## Acceptance
- Greedy ids are identical on oQ4e. Decode output at 16k context matches the old path, including after the rotating cache wraps.
- ms/token improves against a same-session baseline at 2k and 16k.
- Prompt-cache and ChatSession checkpoint behaviour is unchanged: rotated caches still trim or restore as now.
- All five bundles are green. #flint-perf