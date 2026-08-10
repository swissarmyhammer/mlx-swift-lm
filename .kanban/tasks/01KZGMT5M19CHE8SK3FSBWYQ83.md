---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp8tb6xh0hq04w94sj10ba2
  text: |
    ## Note carried from the math-helpers task `p34crp6` (review, 2026-08-10)

    This card says of the Sinkhorn step: "call `hcSplitSinkhorn` from the math-helpers task (already tested there); this task wires it in, it does not reimplement it." That is correct with one qualification.

    **The `eps` term is not pinned by any test on `p34crp6`.** `hcSplitSinkhorn` uses `eps` in three places -- `pre = sigmoid(...) + eps`, `comb = softmax(...) + eps`, and as the division guard inside each row and column normalization. The code holds all three and agrees with the Python. But the 3x3 fixture allows 1e-6 absolute, and removing the `+ eps` on `comb` moves the result by only 8.54e-7. The fixture thus stays green with the term deleted.

    Nothing is wrong in the delivered code. The point for this card is narrower: **do not treat "eps is wired in correctly" as already proven.** The `p34crp6` mutation table lists eight mutations and none of them touches `eps`.

    Two things follow.

    - Pass `hc_eps` from the configuration, as `hcSplitSinkhorn` requires it as an argument with no default. It does not carry a copy of the value.
    - If you want the term pinned, the robust detector is not a tighter tolerance. The measured float32 error on that fixture is 8.44e-8 against the 1e-6 limit, thus a 1e-7 limit would leave only about 1.2x headroom and would be fragile. Prefer a direct test: run `hcSplitSinkhorn` twice on the same input with `eps = 0` and with `eps = 1e-6` and assert the outputs differ. That detects the term itself rather than detecting it by coincidence.
  timestamp: 2026-08-10T16:41:42.877735+00:00
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