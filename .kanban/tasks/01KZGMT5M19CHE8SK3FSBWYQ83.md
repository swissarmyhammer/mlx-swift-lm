---
assignees:
- claude-code
depends_on:
- 01KZGMQCH9PFY25Y3QXP34CRP6
position_column: todo
position_ordinal: '8580'
title: Port DeepseekV4 mHC hyper-connections (HyperConnection, HyperHead, head reduce)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` with `DeepseekV4HyperConnection` and `DeepseekV4HyperHead`. This implements DSV4's **mHC (manifold-constrained hyper-connections)** residual stream — a `hc_mult=4` set of parallel residual copies that are collapsed and expanded per block using a Sinkhorn-normalized mixing matrix. Nothing analogous exists in this repo.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4.swift` — `DeepseekV4HyperConnection` at line 1361, `DeepseekV4HyperHead` at line 1514.

Pieces:

1. `_hc_pre` / `_hc_post` — collapse the 4-way residual stream into the block input and expand the block output back out. The gap tracker flags **axis alignment as Bug 1**: getting the mixing axes transposed yields plausible-but-wrong activations that will not crash. Assert on shapes *and* values.
2. Sinkhorn normalization of the `comb` matrix — call `hcSplitSinkhorn` from the math-helpers task (already tested there); this task wires it in, it does not reimplement it.
3. `_hc_head_reduce` — the pre-norm mHC reduction at the top of the stack using the `hc_head_*` parameters, producing the single stream the final norm and LM head consume.

Uses `MLXFast.rmsNorm` (reference lines 912, 1554), already reachable from this target.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4.swift` lines ~1361-1568 (MIT; header attributes Osaurus AI).
- Bug 1 (axis alignment) and the mHC description: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.
- Numeric cross-check: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift` exists with both types.
- [ ] `_hc_pre` then `_hc_post` round-trips shape: `[B, L, hidden]` to collapsed to expanded `[B, L, hc_mult, hidden]` (or whatever the reference's exact layout is — assert it explicitly).
- [ ] A value-level test against Python-generated fixtures passes, proving axis alignment (Bug 1) is correct rather than merely shape-compatible.
- [ ] `_hc_head_reduce` reduces the 4-way stream to a single `[B, L, hidden]` stream.
- [ ] No changes to existing files.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift`, synthetic weights only.
- [ ] Test: `_hc_pre`/`_hc_post` shapes for `hc_mult=4`, B=1, L=4, hidden=64.
- [ ] Test: value parity against a checked-in Python fixture for a small deterministic input — this is the Bug 1 guard; a transposed implementation must fail it.
- [ ] Test: with an identity-like `comb` matrix, `_hc_post(_hc_pre(x))` is close to `x` (sanity path).
- [ ] Test: `_hc_head_reduce` output shape and `allFinite`.
- [ ] Run: `swift test --filter DeepseekV4HyperConnectionTests` — all pass.

## Workflow
- Use `/tdd` — write the fixture-based value test first; it is the only thing that catches a transposed axis.
#deepseek-v4