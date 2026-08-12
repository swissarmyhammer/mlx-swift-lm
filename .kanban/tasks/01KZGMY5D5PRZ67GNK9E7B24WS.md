---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzthrx95fw2gc45zpe025nw1
  text: |-
    Research results, before implementation:

    1. Weights: the checkpoint `mlx-community/DeepSeek-V4-Flash-4bit` is NOT on this machine. `~/.cache/huggingface/hub` holds only `models--deepseek-ai--DeepSeek-V4-Flash` (11 MB, tokenizer and configuration files only, no safetensors). `~/Documents/huggingface/models/mlx-community/` holds four small models and no DeepSeek-V4. The hard rule says: do not start the ~91 GB download. Thus the real-weights tests will show the skip path today.
    2. Location: the card names `Tests/MLXLMIntegrationTests/` and `swift test --filter`. That directory does not exist. The root package has NO swift-transformers dependency by design (the Downloader/TokenizerLoader abstraction keeps it out), thus a root SwiftPM test target cannot load a real Hugging Face tokenizer. Every real-weights suite lives in `IntegrationTesting/IntegrationTestingTests/` (Xcode project with swift-transformers, filesystem-synchronized groups). The new suite goes there as `DeepseekV4IntegrationTests`, per the dispatch instruction to follow what exists. Run: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests`.
    3. Load path with no download: `LLMModelFactory.shared.loadContainer(from: directory, using: #huggingFaceTokenizerLoader())` loads from a local directory and never touches the network. The gate finds a local snapshot via `IntegrationTestHelpers.hfSnapshotDir`, the swift-transformers download base, or an `MLX_DEEPSEEK_V4_CHECKPOINT` override, and requires at least one `*.safetensors` file.
    4. Weight-key completeness: `loadWeights` verifies with `[.all]`, thus a load that completes proves no missing and no unexpected keys.
    5. Layer count: `DeepSeekV4ModelInner.layers` is internal, but `DeepSeekV4Model.kvHeads` is public with one entry per layer (43), and `newCache` answers one cache per layer.
    6. Long generation: `TokenIterator.next()` never checks EOS (the generate loop does), thus a direct iterator run of more than 12,288 steps is a deterministic length regardless of content. `MLX.Memory.snapshot()` gives active/peak memory for the stability record. The mlx-lm issue-1662 leak kills the process at about 11.5k tokens, thus completion past 12k is the assertion.
    7. Two-round cache facts, derived from `DeepSeekV4ChatEncoder.render`/`tail`/`removingEarlierReasoning`: in thinking mode, round 2 re-renders round 1's user tail as `<assistant></think>` where round 1 primed `<assistant><think>`, and it drops the round-1 reasoning. Thus round 1 prompt + generation is NEVER a prefix of the round-2 rendering in thinking mode — cross-round prefix caching cannot hold there. In chat mode the tail is `</think>` in both rounds, and the prefix relation holds. `ChatSession` avoids the problem in-session: it keeps the live KV cache and renders only the new messages each turn. These facts get a weights-free encoder-level test; the recall behavior gets a real-weights two-round test.
    8. Parity fixture: no fixture exists and this machine cannot produce one (weights absent, no Python reference run possible). The parity test is gated on a fixture JSON at a `#filePath`-relative location and skips with the exact Python generation command in the message. No token ids get invented.
  timestamp: 2026-08-12T08:35:10.757175+00:00
- actor: claude-code
  id: 01kztj2vaepn5xq2ts53n6fzmq
  text: |-
    ### implement — changed
    - evidence: 1 new file — IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift (7 tests). Weights ABSENT on this machine (only the 11 MB deepseek-ai tokenizer snapshot is cached), thus no download was started per the hard rule. Full suite run via `xcodebuild test ... -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests`: TEST SUCCEEDED, 7 tests — 5 real-weights tests skip cleanly with reason messages (memory gate 160 GiB, local-checkpoint gate with safetensors check, `MLX_DEEPSEEK_V4_CHECKPOINT` override; the parity test also gates on a fixture file and prints the Python generation script). The 2 encoder-level cache tests pass: thinking mode is not prefix-cacheable across rounds (tail re-render `<think>` -> `</think>` plus dropped reasoning), chat mode is. A mutation run (both prefix assertions flipped) failed both tests, which proves the assertions bite; the assertions were then restored and the suite went green again.
    - unverified for lack of weights: real load (weight-key completeness, model type, 43 layers), greedy parity against a Python fixture (no fixture exists; none was invented), chat/thinking real generation, the >12k-token issue-1662 run, and the real two-round recall. Each stays gated and runs when a checkpoint is local.
    - next: /review. When someone places the checkpoint locally (or sets MLX_DEEPSEEK_V4_CHECKPOINT) and checks in the Python parity fixture at IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json, re-run the suite and record the real numbers.
  timestamp: 2026-08-12T08:40:36.430529+00:00
- actor: claude-code
  id: 01kztjb4jv261ahbdajrvf8p0v
  text: |-
    ### finish iteration 1 — stuck
    - implement: changed — 1 new file, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift (7 tests; 5 gated real-weights tests skip cleanly with reasons, 2 encoder cache tests pass with a mutation proof)
    - test: green — swift test, exit 0, 0 failures
    - commit: b98cac9
    - review: not run — the card cannot reach done in this state
    - stuck: five acceptance criteria need the real mlx-community/DeepSeek-V4-Flash-4bit checkpoint (~91 GB installed, plus a 160 GiB memory gate). The weights are not on this machine, and the agent must not start that download without the user's decision. The user has two options: (1) download the checkpoint (or point MLX_DEEPSEEK_V4_CHECKPOINT at a copy) and run `swift test` from the IntegrationTesting project — the gated tests then run for real; (2) accept the gated suite as the deliverable and defer the real-weights run. The greedy-parity fixture also needs one Python reference run; the fixture gate message names the script.
  timestamp: 2026-08-12T08:45:08.059225+00:00
depends_on:
- 01KZGMXEJ4A72EE95T2MJRZKGM
position_column: doing
position_ordinal: '80'
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

- [ ] New integration test loads the real repo id end to end with no missing/unexpected weight keys. (Test written and gated; NOT verified — the checkpoint is not on this machine and the hard rule forbids the ~91 GB download.)
- [ ] First N greedy token ids match a Python-generated fixture exactly. (Test written and gated on a fixture file; NOT verified — no fixture exists and this machine cannot run the Python reference without the weights. No token ids were invented.)
- [ ] Both `chat` and `thinking` prompts generate successfully. (Test written; NOT verified — weights absent.)
- [ ] A >12k-token generation completes without a Metal buffer-count crash, or an equivalent buffer-stability assertion passes. (Test written with a deterministic 12,400-step `TokenIterator` run; NOT verified — weights absent.)
- [x] The test skips cleanly (not fails) when the weights are absent or memory is insufficient, with a message saying why. (Verified: full suite run shows 5 clean skips with reason messages, 0 failures.)
- [x] Two-round prompt-cache behavior is asserted or explicitly documented as not-cacheable with the reason. (Verified at the encoder level, no weights needed: thinking mode is NOT prefix-cacheable across rounds — round 2 re-renders the tail as `</think>` where round 1 primed `<think>`, and it drops the round-1 reasoning; chat mode IS prefix-cacheable. Both assertions ran green and a mutation run proved both can fail. The real-weights two-round recall test stays gated.)

## Tests

- [x] New `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift`. (The card's original `Tests/MLXLMIntegrationTests/` path is not buildable: that target does not exist and the root package deliberately has no swift-transformers dependency, thus it cannot load the real tokenizer. All real-weights suites live in the IntegrationTesting Xcode project; see the comment trail.)
- [x] Test: load plus assert model type and layer count (43).
- [ ] Test: greedy first-N-token parity against the checked-in Python fixture. (The test exists and is fixture-gated; no fixture is checked in — it must come from the Python reference against real weights.)
- [x] Test: chat vs thinking mode both produce non-empty output.
- [x] Test: long generation past 12k tokens completes (regression guard for the mlx-lm 1662 leak).
- [x] Test: two-round conversation — assert cache hit, or assert-and-document no-hit.
- [ ] Run: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests` with weights present — all pass; without weights — all skip. (The without-weights half is verified: 7 tests, 5 clean skips, 2 encoder tests pass. The with-weights half stays unverified until someone downloads the checkpoint.)

## Workflow
- Use `/tdd` — write the load plus parity test first; it will fail until every preceding task is green, which is the point.
#deepseek-v4