---
assignees:
- claude-code
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
position_column: todo
position_ordinal: '8480'
title: 'Port DeepseekV4 attention: partial RoPE, attn_sink, inverse-RoPE output, grouped low-rank O'
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` holding `DeepseekV4RoPE` and `DeepseekV4Attention`. This is the first half of the model forward; the mHC, MoE, and decoder-layer pieces are separate tasks.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4RoPE` at line 91, `DeepseekV4Attention` at line 133 (the file is 2509 lines total; take only these two types). Split into its own file rather than one 2500-line file, matching how this repo keeps model files navigable.

Shape facts from the real config: `head_dim=512`, 64 query heads, `num_key_value_heads=1` (one latent KV head broadcast to all 64 Q heads via GQA), RoPE applied to only the trailing `qk_rope_head_dim=64` dims, q/o LoRA rank 1024.

Features to implement, all NEW vs DeepseekV3:

1. **Per-layer RoPE theta** — `DeepseekV4RoPE` selects `rope_theta=10000` when `compress_ratio == 0` and `compress_rope_theta=160000` (with YaRN scaling) otherwise. Use `ropeTheta(forLayer:)` and `yarnInvFreq` from the config/math-helper tasks.
2. **Learned per-head `attn_sink`** — prepend a learned sink logit column before softmax, drop it after. `use_attn_sink` defaults **ON**. Flagged Bug 3 in the gap tracker.
3. **Inverse RoPE on the attention output** — rotate the trailing 64 dims backward via `conj(freqs_cis)` to strip positional info before the residual add-back. Uses the inverse partial RoPE from the math-helpers task.
4. **Grouped low-rank O projection** — `o_groups=8`: reshape O to `(B, L, 8, 4096)`, then `wo_a` einsum (`bsgd,grd -> bsgr`) followed by `wo_b(flatten(...))`.

**Critical — Metal buffer leak.** `ml-explore/mlx-lm` issue 1662: a model that discards the value returned by the cache's `updateAndFetch` leaks one Metal buffer per layer per generated token, and `deepseek_v4` deterministically crashes at ~11.5k generated tokens. Use this repo's existing `attentionWithCacheUpdate` helper (`Libraries/MLXLMCommon/AttentionUtils.swift:37`), which already handles the cache update correctly, rather than hand-rolling the cache interaction. If a custom path is unavoidable because of the sink column, the returned keys/values MUST be the ones fed to SDPA.

Note: `MLXFast` is already reachable from these targets (`AttentionUtils.swift:46`), so no `Package.swift` change is needed. Do not modify `DeepseekV3.swift`.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~91-756 (MIT; header attributes Osaurus AI).
- Bug numbering and feature list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`.
- Leak landmine: `ml-explore/mlx-lm` issue 1662.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4Attention.swift` exists with `DeepseekV4RoPE` and `DeepseekV4Attention`.
- [ ] Forward pass on synthetic weights returns the correct shape for both a prefill (L>1) and a decode (L=1) step.
- [ ] `attn_sink` is on by default; a test with `use_attn_sink=false` produces different output, proving the sink actually participates.
- [ ] Output dims 0..<448 are unaffected by position while dims 448..<512 are — i.e. the inverse RoPE demonstrably ran.
- [ ] Grouped low-rank O projection produces `hidden_size=4096` output from `o_groups=8` grouping.
- [ ] The value returned by the cache update is the value passed to SDPA (guards the mlx-lm 1662 leak).
- [ ] `DeepseekV3.swift` unmodified.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4AttentionTests.swift` using small synthetic config + random weights (no download).
- [ ] Test: prefill shape `[1, 8, 4096]` in to `[1, 8, 4096]` out; decode shape `[1, 1, 4096]` in to `[1, 1, 4096]` out; output `allFinite`.
- [ ] Test: `use_attn_sink` true vs false yields materially different output (assert not-almost-equal).
- [ ] Test: per-layer theta — a layer with `compress_ratio == 0` and one with nonzero ratio produce different RoPE output for the same input.
- [ ] Test: cache growth — after N decode steps the cache offset is exactly N, and the arrays SDPA consumed are the arrays the cache returned (regression test for the mlx-lm 1662 leak).
- [ ] Run: `swift test --filter DeepseekV4AttentionTests` — all pass.

## Workflow
- Use `/tdd` — write the shape, sink-participation, and cache-identity tests first, then port.
#deepseek-v4