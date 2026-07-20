---
depends_on:
- 01KXX99P1H2Z3DV0TM0AGBFEBR
position_column: todo
position_ordinal: '8380'
title: 'MiniMax M3: attention with per-head QK norm, partial RoPE, MSA dense fallback'
---
#minimax-m3

## What
Add the attention and normalization layers to `Libraries/MLXLLM/Models/MiniMaxM3.swift`. Baseline: `MiniMax.swift` (M2) `MiniMaxAttention`; M3 deltas below. Reference implementations: upstream mlx-lm PRs #1398 (`minimax_m3_vl.py`) and #1401 (`minimax_m3`).

- `MiniMaxM3Attention`: GQA with 64 query heads / 4 KV heads, head_dim 128, scale headDim^-0.5, no biases.
- **Per-head QK norm** (`qk_norm_type: "per_head"`): RMSNorm over `headDim` applied after reshaping to `(B, L, heads, headDim)` — unlike M2's flat RMSNorm over `heads * headDim` before the reshape. Verify the norm weight shape against the checkpoint index / upstream PR #1401 before committing to the layout.
- **Partial RoPE**: `RoPE(dimensions: rotaryDim /* 64, half of head_dim */, traditional: false, base: ropeTheta /* 5e6 */)` — M2 already does this; keep the `applyRotaryPosition`/`ropeOffset` cache pattern from `MiniMax.swift`.
- **Gemma-style RMSNorm** (`use_gemma_norm: true`): `(1 + weight)` scaling with float32 normalization. Reuse the repo's existing Gemma norm (see `Gemma.swift` / `Gemma3Text.swift`) if accessible from this module; otherwise a local private copy. Used for input/post-attention layer norms (wired in the assembly task) and constructed here as the shared norm helper.
- **MiniMax Sparse Attention (MSA) → dense fallback**: layers 3-59 are trained with block-sparse attention (top-16 blocks x 128 tokens + init/local blocks); like both upstream PRs, implement full dense causal attention instead and do not construct the sparse index-head modules. Document in a comment: dense is numerically exact up to ~(sparse_topk_blocks x sparse_block_size = 2048) tokens plus the init/local windows, an accepted approximation beyond.
- `attention_output_gate` is false for M3 — no gate module.

## Acceptance Criteria
- [ ] Attention module builds from a tiny `MiniMaxM3Configuration` and produces correct output shape `(B, L, hidden)`
- [ ] Per-head QK norm applies over head_dim (verified by a shape/behavior test with a non-uniform norm weight)
- [ ] Rotary applies to only the first 64 of 128 head dims (unrotated tail passes through)
- [ ] KV cache path works via `attentionWithCacheUpdate` (same seam as M2)
- [ ] MSA fallback documented in code; no sparse-index parameters exist in the module tree

## Tests
- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: attention shape test (prefill + single-token decode with KV cache), per-head-norm behavior test, partial-RoPE tail-passthrough test (see `RoPEApplicationTests.swift` for conventions)
- [ ] Run `swift test --filter MiniMaxM3`; expect pass

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax