---
assignees:
- claude-code
depends_on:
- 01KZGMXEJ4A72EE95T2MJRZKGM
position_column: todo
position_ordinal: 8a80
title: Real-weights integration test for mlx-community/DeepSeek-V4-Flash-4bit
---
## What

Prove the port actually works against the real checkpoint, not just synthetic weights. Add an integration test that loads `mlx-community/DeepSeek-V4-Flash-4bit` and generates.

The model is a 284B-total / 13B-active MoE — expect roughly 91 GB installed for the 4-bit build. That will not run on ordinary CI, so the test must gate on available memory and on the weights being present locally, and skip cleanly otherwise. Follow the existing pattern in `Tests/MLXLMIntegrationTests/` and `Libraries/IntegrationTestHelpers/`.

Cover:
1. Load succeeds via `LLMModelFactory` from the repo id — the whole path: config decode, type registry, weight load, `sanitize`, quantize.
2. A short greedy generation produces coherent, non-repeating output. Compare against the same prompt run through a Python reference (`Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, or `ml-explore/mlx-lm` PR 1189) — assert the first N greedy token ids match exactly. Greedy decode is deterministic, so this is a real parity check, not a vibe check.
3. Both `chat` and `thinking` mode prompts render and generate.
4. **Long-generation stability** — the `ml-explore/mlx-lm` issue 1662 landmine: a mis-wired cache leaks one Metal buffer per layer per token and `deepseek_v4` deterministically dies at ~11.5k generated tokens. Generate past 12k tokens (or assert stable Metal buffer count over a long run if a full 12k generation is impractical in the test budget). This is the single most important assertion in the task — the synthetic-weight test in `ag7ant0` cannot catch it.

Note the prompt-cache caveat already recorded for this repo: generation-priming tokens can make round N's prompt not a prefix of round N+1, breaking cache reuse. If DSV4's encoder primes, verify caching with a real two-round test rather than assuming.

## Provenance
- Model: `mlx-community/DeepSeek-V4-Flash-4bit` (284B total / 13B active, 1M context).
- Python parity reference: `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` `mlx_lm/models/deepseek_v4.py`; `ml-explore/mlx-lm` PR 1189.
- Leak landmine: `ml-explore/mlx-lm` issue 1662.

## Acceptance Criteria

- [ ] New integration test loads the real repo id end to end with no missing/unexpected weight keys.
- [ ] First N greedy token ids match a Python-generated fixture exactly.
- [ ] Both `chat` and `thinking` prompts generate successfully.
- [ ] A >12k-token generation completes without a Metal buffer-count crash, or an equivalent buffer-stability assertion passes.
- [ ] The test skips cleanly (not fails) when the weights are absent or memory is insufficient, with a message saying why.
- [ ] Two-round prompt-cache behavior is asserted or explicitly documented as not-cacheable with the reason.

## Tests

- [ ] New `Tests/MLXLMIntegrationTests/DeepseekV4IntegrationTests.swift`.
- [ ] Test: load plus assert model type and layer count (43).
- [ ] Test: greedy first-N-token parity against the checked-in Python fixture.
- [ ] Test: chat vs thinking mode both produce non-empty output.
- [ ] Test: long generation past 12k tokens completes (regression guard for the mlx-lm 1662 leak).
- [ ] Test: two-round conversation — assert cache hit, or assert-and-document no-hit.
- [ ] Run: `swift test --filter DeepseekV4IntegrationTests` with weights present — all pass; without weights — all skip.

## Workflow
- Use `/tdd` — write the load plus parity test first; it will fail until every preceding task is green, which is the point.
#deepseek-v4