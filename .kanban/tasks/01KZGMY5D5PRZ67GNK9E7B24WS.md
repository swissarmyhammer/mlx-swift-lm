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
- actor: claude-code
  id: 01kztksetzdp3ax8dy37w0np4n
  text: |-
    ### The user authorizes the download — 2026-08-12
    The user wrote: "let's download and really run --". Thus the agent downloads mlx-community/DeepSeek-V4-Flash-4bit and runs the gated tests for real. Machine check: Apple M3 Ultra, 512 GiB memory (the 160 GiB gate passes), 395 GiB free disk (the ~91 GB checkpoint fits).
  timestamp: 2026-08-12T09:10:25.887775+00:00
- actor: claude-code
  id: 01kztngskdcehbjadxf2k84f5w
  text: |-
    ### Download complete — 2026-08-12
    The checkpoint `mlx-community/DeepSeek-V4-Flash-4bit` is now in the Hugging Face cache at `~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/38c0bd20a6fba70f22c5ee2940ec0092b36ab936`. Facts: 141 GB on disk, 33 safetensors shards plus the index, zero `.incomplete` files, download time approximately 29 minutes, `hf download` exit 0. Note: the card's estimate of ~91 GB was too low — 284B parameters at 4 bits is approximately 142 GB, and that agrees with the disk size. The machine has 512 GiB memory, thus the 160 GiB memory gate passes. Next: run the gated tests for real.
  timestamp: 2026-08-12T09:40:39.149486+00:00
- actor: claude-code
  id: 01kztp4dyqyzxbssz3t71en6ad
  text: |-
    ### Real-weights run 2026-08-12 — results

    **1. Download: complete.** 141 GB on disk, 33 safetensors shards, 0 `.incomplete` files, ~29 minutes. Snapshot: `~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/snapshots/38c0bd20a6fba70f22c5ee2940ec0092b36ab936`.

    **2. Python parity fixture: produced.** The route that worked, and the routes that did not:
    - `Thump604/mlx-lm` @ `deepseek-v4-support-fixes` (0.31.3): FAILS to load this checkpoint — "Received 258 parameters not in model: model.layers.N.attn_hc.{base,fn,scale}, ...ffn_hc...". Its model predates the hyper-connection tensors of the published checkpoint.
    - Released `mlx-lm` and `ml-explore/mlx-lm@main` (254d153): no `deepseek_v4` module at all.
    - `ml-explore/mlx-lm` PR 1189 head (63a2662): supports this checkpoint, but does not import as published — `mlx_lm/models/deepseek_v4.py` uses `Any` without importing it. One-line fix applied in a local copy (`from typing import Any, Dict, List, Optional`); no other change.
    - Result: load + 64 greedy tokens in 238 s. Fixture written to `IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json` (18 prompt tokens, 64 generated ids; NOT committed, per the no-commit rule). First 16 ids: 671, 8343, 344, 260, 12596, 14, 34023, 106344, 396, 12927, 33168, 31760, 21537, 1009, 4433, 16. Greedy text starts: "The sea is a vast, mysterious expanse that holds countless secrets beneath its surface.</think>…" — coherent, non-repeating.

    **3. Swift suite run with weights present: the load FAILS. This is the real finding.**
    Command: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests`.
    Error: `keyNotFound(path: ["model","layers","0","hc_attn","base"], modules: [DeepSeekV4Model, DeepSeekV4ModelInner, DeepSeekV4DecoderLayer, DeepSeekV4HyperConnection])` from `verify: [.all]`.
    Cause: the mlx-community checkpoint spells each per-layer hyper-connection `attn_hc.{fn,base,scale}` / `ffn_hc.{fn,base,scale}` (43 layers x 6 = 258 tensors). The Swift `DeepSeekV4Model.modulePath(of:)` renames only the raw flattened form `.hc_attn_fn` -> `.hc_attn.fn`. It does not turn `attn_hc.X` into `hc_attn.X`. The Python PR 1189 applies exactly that rename ("attn_hc.X -> hc_attn.X — mlx-community naming order") after the underscore rename, so both orders converge.
    A second divergence is certain to bite after that fix: the checkpoint spells the routing bias `model.layers.N.ffn.gate.e_score_correction_bias` (40 top-k layers). The Swift gate declares `@ParameterInfo(key: "bias")` and `sanitize` has no rename for it. With `verify [.all]` those 40 tensors fail as unused keys; without the check they would leave the gate bias nil and corrupt expert routing. PR 1189 maps `gate.bias` <-> `gate.e_score_correction_bias` (its parameter carries the long name).
    Full inventory check of `model.safetensors.index.json` (2481 tensors): every other family matches the Swift module paths — `model.` prefix, `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` un-split, `q_norm`/`kv_norm`, `attn_sink`, `switch_mlp` pre-stacked `{gate,up,down}_proj` (weight+scales), `shared_experts.*_proj` (weight+scales+biases), `gate.weight` on all 43 layers, `gate.tid2eid` on the 3 hash layers, `model.hc_head.{fn,base,scale}`, `lm_head`/`embed_tokens`. The `compressor` (41 layers) and `indexer` (21 layers) tensors are dropped by design until tasks `^tty95f4`/`^r92pjcr` land sparse attention.

    **4. Consequence for the gated tests.** Because the shared load fails, all 5 real-weights tests skip through the load-failure path (message carries the error), and the 2 encoder cache tests pass. Thus these stay unverified: weight-key completeness, greedy parity, chat/thinking generation, the >12k-token issue-1662 run, and the real two-round recall. None was papered over.

    **5. Proposed next fix task (user decides).** In `DeepSeekV4Model.modulePath(of:)`: (a) add the rename `.attn_hc.` -> `.hc_attn.` and `.ffn_hc.` -> `.hc_ffn.` after the underscore rename; (b) resolve the gate-bias name (`e_score_correction_bias` in the checkpoint vs `bias` in the Swift gate) with a rename in `sanitize` or a key change on the parameter. Add synthetic-weight unit tests that feed the mlx-community key spelling. Then re-run this suite; the fixture and the downloaded weights are already in place.
  timestamp: 2026-08-12T09:51:22.583144+00:00
- actor: claude-code
  id: 01kztp4pdg857vwad0t6f5mw8d
  text: |-
    ### implement — stuck
    - evidence: download complete (141 GB, 33 shards, 0 incomplete files, ~29 min). Python parity fixture produced by ml-explore/mlx-lm PR 1189 head (63a2662, plus a one-line `Any` import fix) against the same weights: 18 prompt tokens, 64 greedy ids, 238 s, coherent text; fixture at IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json (not committed). Swift suite run with weights present: the load FAILS — keyNotFound model.layers.0.hc_attn.base. Cause: sanitize does not rename the checkpoint's `attn_hc.*`/`ffn_hc.*` spelling (258 tensors) to `hc_attn.*`/`hc_ffn.*`; a second divergence follows — `ffn.gate.e_score_correction_bias` (40 tensors) vs the Swift gate key `bias`. All 5 real-weights tests skip through the load-failure path; the 2 encoder cache tests pass. Unverified for the failing load: weight-key completeness, greedy parity, chat/thinking generation, the >12k issue-1662 run, real two-round recall.
    - next: a user-approved fix task on `DeepSeekV4Model.modulePath(of:)`/`sanitize` (the two renames, with synthetic-key unit tests), then re-run this suite — weights and fixture are already in place.
  timestamp: 2026-08-12T09:51:31.248555+00:00
- actor: claude-code
  id: 01kztp5y0xcbda6825ateq89y6
  text: |-
    ### finish iteration 2 — red (real load failure found)
    - implement: changed — download complete (141 GB, 33 shards, 0 incomplete, ~29 min); Python parity fixture produced via ml-explore/mlx-lm PR 1189 (18 prompt + 64 greedy ids); real Swift load FAILS with keyNotFound model.layers.0.hc_attn.base
    - root cause recorded on the card: the checkpoint spells attn_hc.*/ffn_hc.* (258 tensors) and gate e_score_correction_bias (40 tensors); the Swift sanitize/modulePath does not accept those names; all 2481 index keys checked, no other family diverges
    - next: fix the two weight-key renames with TDD unit tests, then re-run the real-weights suite. The user's delegation ("make DeepSeek work") covers this fix.
  timestamp: 2026-08-12T09:52:11.805060+00:00
- actor: claude-code
  id: 01kztw52jk5vh8bmtzweh8xf3b
  text: |-
    ### finish iteration 3 — stuck (the user stopped the agent mid-run)
    - implement: changed, in progress — the sanitize fix for the checkpoint key names is in the tree (Libraries/MLXLLM/Models/DeepSeekV4.swift + unit tests in Tests/MLXLMTests/DeepSeekV4ModelTests.swift; 117 DeepSeek-family tests pass), NOT yet committed
    - real-weights results so far: load PASSES (21.8 s, all keys accepted); chat and thinking modes PASS with coherent output ("The sea exhales a salty sigh against the shore."); greedy parity FAILED — but the analysis shows the FIXTURE is defective: the Python script made its prompt with ASCII pseudo-markers (<|user|>) instead of the real fullwidth markers, the reference output degenerates into repeated "2.0" loops, and the Swift output is the healthy one (first divergence at index 0). The fixture must be regenerated with the correct prompt token ids.
    - the >12k-token issue-1662 run STILL RUNS as a detached xcodebuild (pid 72429, ~1h38m elapsed at this note) — no crash so far. Its verdict lands in scratchpad/dsv4-real-weights-run.log.
    - two-round recall: not yet reached in this run.
    - next when work resumes: (1) read the long-run verdict from the log; (2) regenerate the parity fixture with the correct fullwidth-marker prompt via the mlx-lm PR 1189 environment; (3) re-run parity and the two-round test; (4) commit; (5) review.
  timestamp: 2026-08-12T11:36:35.155670+00:00
- actor: claude-code
  id: 01kztwj87gc4acngfyyfbnvw99
  text: |-
    ### Sanitize fix and first real-weights results — 2026-08-12

    **1. The sanitize fix landed, TDD.** Two new unit tests went in first and failed for the correct reason (the keys passed through unmapped): `sanitizeConvergesBothHyperConnectionSpellings` and `sanitizeMapsTheScoreCorrectionBiasOntoTheGateBias` in `Tests/MLXLMTests/DeepSeekV4ModelTests.swift`. The fix in `DeepSeekV4Model.modulePath(of:)` mirrors `ml-explore/mlx-lm` PR 1189 `Model.sanitize` (local copy at the scratchpad, `mlx-lm-pr1189`, head 63a2662): (a) rename `.attn_hc.` / `.ffn_hc.` to `.hc_attn.` / `.hc_ffn.` AFTER the underscore rename, thus the two spellings converge; (b) rename `.ffn.gate.e_score_correction_bias` to `.ffn.gate.bias` — the Swift gate holds the one score-correction tensor under `bias` and adds it to the expert scores only for the top-k selection, the same place the Python reference adds its `e_score_correction_bias`. Full family run: `swift test --filter DeepSeek` — 117 tests, 0 failures.

    **2. Real-weights run (full suite, later killed per the user's instruction to go one test at a time):**
    - `loadsTheRealCheckpointEndToEnd`: PASSED in 21.8 s. Model type `DeepSeekV4Model`, 43 layers. The load verifies with `[.all]`, thus no missing and no unexpected keys. The keyNotFound failure is gone.
    - `greedyFirstTokensMatchThePythonFixture`: FAILED. First differing index: 0. Swift generated id 455 (' The'); the fixture says id 671 ('The'). The full Swift stream is coherent: reasoning inside `<think>`, then `</think>`, then one good sentence, then end-of-sentence. The fixture stream answers with no reasoning, repeats itself, and degenerates after its end-of-sentence token.
    - `chatAndThinkingModesBothGenerate`: PASSED in 232.2 s. Thinking output starts "The user wants a short sentence about the sea. I need to generate a concise ...". Chat output: "The sea exhales a salty sigh against the shore." Both non-empty and coherent.
    - `longGenerationPastTwelveThousandTokensCompletes`: NOT COMPLETED — killed after ~83 minutes in progress. Decode speed is roughly 2-3 s/token on this machine, thus 12,400 tokens needs hours and exceeds the suite's 240-minute limit. This is its own finding (decode performance), separate from correctness.
    - `twoRoundConversationRecallsTheFirstRound`: NOT RUN (came after the killed long test in the serialized suite).

    **3. The parity failure is in the FIXTURE's generation path, not (so far) in the Swift port.** Probe against the same weights with the same PR 1189 Python model, full 18-token forward, top-10 logits at step 0: rank 0 = id 455 (' The') logit 24.25, rank 1 = id 671 ('The') logit 22.25 — margin 2.0, not a near-tie. Thus the Python model's own batch forward argmax AGREES with the Swift first token (455) and DISAGREES with the fixture (671). The fixture came from `generate_step`, whose step 0 goes through the cached S=1 decode path (prefill 17 tokens, then the last prompt token alone) — the exact area the PR 1189 head commit names "S=1 cache divergence". A second probe now runs both Python paths side by side (cached S=1 decode vs uncached full-forward greedy) to confirm where the divergence lives and whether the Python full-forward stream matches the Swift stream.
  timestamp: 2026-08-12T11:43:46.928704+00:00
- actor: claude-code
  id: 01kzty0a7w513xb5mahdnw5a8x
  text: |-
    ### Parity root cause found and fixture corrected — 2026-08-12

    Method change per the user's instruction: run one test at a time; when a test fails, stop and work that failure. The full-suite background run was killed (the >12k test held the machine for 83 minutes with hours to go).

    **Three-way probe on the same weights, same PR 1189 Python model, same 18 prompt tokens:**
    1. Python full-prompt forward, top-10 logits at step 0: rank 0 = id 455 (' The') logit 24.25; rank 1 = id 671 ('The') logit 22.25. Margin 2.0.
    2. Python cached decode (prefill 17 tokens, then S=1 steps — the `generate_step` path): reproduces the OLD fixture exactly (671, ...). Thus the Python cached path disagrees with the Python model's own batch forward by 2.0 logits at step 0 — a reference-side S=1 defect, the same area the PR 1189 head commit names "rolling-state compressor decode — fixes S=1 cache divergence".
    3. Python uncached full-forward greedy (one forward over the growing sequence per step): reproduces the SWIFT stream exactly for 32 of 32 probed tokens.

    Thus the Swift port was not the diverging side. The fixture was regenerated with the full-forward method (script recorded in `DeepseekV4ParityFixture`'s doc comment, and at the scratchpad as `make_parity_fixture_v2.py`); the old `generate_step` method is now forbidden in that doc comment. Fixture file: `IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json` (18 prompt tokens, 64 full-forward greedy ids; in the tree, NOT committed).

    **Parity test re-run alone against the corrected fixture** (`xcodebuild ... '-only-testing:...greedyFirstTokensMatchThePythonFixture()'`, 177 s): the Swift stream matches the reference for the FIRST 36 of 64 tokens and parts at index 36 — Swift id 106344 (' expanse'), reference id 16090 (' entity'). Margin probe at that step: the reference's top two logits are EXACTLY tied at 29.750 (' entity' and ' giant'); ' expanse' sits at 29.375, 0.375 below, inside a four-way cluster within 1.25 logits. The bf16 logit granularity at this magnitude is 0.125, thus index 36 is a floating-point near-tie, not a systematic gap.

    **Decision needed from the user (parity window):** an exact 64-token match across two implementations is not a stable target when the reference itself flips on bf16 near-ties. Options: (a) pin the assertion to the first 32 tokens (safely inside the stable window of 36) and record the near-tie evidence; (b) keep 64 and accept a standing failure; (c) a tolerance-aware assertion (each Swift token must sit within a small logit margin of the reference's top token). No option was taken without sign-off — shrinking the assertion silently would be a weakened assertion.

    **Also recorded:** decode speed is about 2.4 s/token, thus the >12k-token test cannot finish inside its 240-minute limit — new task ^3gh7rb5. The two-round recall test has not yet run; run it one test at a time next.
  timestamp: 2026-08-12T12:08:56.316673+00:00
- actor: claude-code
  id: 01kzty0mvbv359jb7v74q68h23
  text: |-
    ### implement — stuck
    - evidence: sanitize fix landed with TDD — 2 files changed for the fix (Tests/MLXLMTests/DeepSeekV4ModelTests.swift: 2 new tests, watched red then green; Libraries/MLXLLM/Models/DeepSeekV4.swift: modulePath renames `attn_hc./ffn_hc.` -> `hc_attn./hc_ffn.` and `ffn.gate.e_score_correction_bias` -> `ffn.gate.bias`, mirroring mlx-lm PR 1189), `swift test --filter DeepSeek` 117 tests 0 failures. Real weights, one test at a time: load PASS 21.8 s (43 layers, verify [.all]); chat/thinking PASS 232 s; parity FAIL at first differing index 36 of 64 against the corrected full-forward fixture — the old fixture was wrong (generate_step S=1 path diverges from the Python model's own forward by 2.0 logits at step 0; three-way probe in the comments), and index 36 is a bf16 near-tie (reference top two exactly tied at 29.75). Fixture regenerated and in place, not committed; provenance doc updated in DeepseekV4IntegrationTests.swift. Long-gen test blocked by ~2.4 s/token decode (needs ~7 h > 240-min limit) — new task ^3gh7rb5. Two-round recall not yet run.
    - next: the user decides the parity window (pin to first 32 tokens, keep 64 with a standing failure, or a tolerance-aware assertion), and whether to run the two-round recall test now and how to handle the long-gen test given ^3gh7rb5.
  timestamp: 2026-08-12T12:09:07.179420+00:00
- actor: claude-code
  id: 01kzw0zbcsy8d5bsqhxpsf3hkr
  text: |-
    ### implement — stuck
    - evidence: the model WORKS against the real checkpoint. One test at a time, all with real weights: load PASS 21.8 s (DeepSeekV4Model, 43 layers, verify [.all]); greedy parity PASS 94.6 s — 32 of 32 tokens equal to the Python reference, with the window the user approved (`parityTokenCount = 32`; the corrected full-forward fixture is in the tree, to commit with this card); chat/thinking PASS 232 s; two-round recall PASS 47.7 s (round 1 "I have noted the number 4172.", round 2 "4172"). Unit family: `swift test --filter DeepSeek`, 117 tests, 0 failures. Files touched this pass: Libraries/MLXLLM/Models/DeepSeekV4.swift, Tests/MLXLMTests/DeepSeekV4ModelTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift, IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json.
    - one criterion stays open: the >12k-token issue-1662 run. Decode speed is about 2.4 s/token, thus 12,400 tokens needs about 7 hours and cannot fit inside the suite's 240-minute limit. Blocked by task ^3gh7rb5 (decode performance); re-run that one test after it lands.
    - next: /review for this card; then ^3gh7rb5.
  timestamp: 2026-08-12T22:20:04.889870+00:00
- actor: claude-code
  id: 01kzw1btcvhqe0vgbbvmzvz2sx
  text: |-
    ### review — clean
    - evidence: engine `review sha HEAD~1..HEAD` (commit 4a390a4), counts: 1 finding, 1 confirmed, 8 files attempted, 0 skipped. Scope check against the diff: 0 findings on changed lines; 1 finding on an untouched line (`DeepSeekV4.swift:558`, `stackRoutedExperts` nesting — the commit does not change that function, it only moves its line number). The item is on the card with an out-of-scope annotation, per the standing scope rule from ^ag7ant0. The engine does not read test files or JSON; the card's red-then-green TDD proof for the 2 unit tests and the real-weights run evidence (load 21.8 s, all 2481 tensors; parity 32/32; chat/thinking generate; two-round recall; 117 DeepSeek unit tests green) is the accepted substitute per project memory.
    - disposition: the delta is clean. The card does NOT move to done: one acceptance criterion is open — the >12k-token generation. It is blocked by decode speed (~2.4 s/token, ~7 h > the 240-minute limit) and is now task ^3gh7rb5, which this card depends on. The card stays in review.
    - next: after ^3gh7rb5 lands, run the long-generation test, then re-run /review on this card.
  timestamp: 2026-08-12T22:26:53.467060+00:00
- actor: claude-code
  id: 01kzw1cnm99hccsjnhhddpm0mh
  text: |-
    ### finish iteration 4 — clean delta, card blocked
    - implement: changed — sanitize key-name fix with red-then-green unit tests; parity window 32 with recorded reason; corrected full-forward fixture; real-weights runs: load 21.8 s (2481 tensors), parity 32/32, chat/thinking generate, two-round recall passes
    - test: green — swift test --filter DeepSeek, 117 tests, 0 failures
    - commit: 4a390a4
    - review: clean — 0 findings on changed lines; 1 untouched-line item recorded out of scope per the standing rule from ^ag7ant0
    - state: the card stays in review, blocked by ^3gh7rb5 — the >12k-token endurance criterion cannot run until decode speed improves (~2.4 s/token gives ~7 h against the 240-minute limit). When ^3gh7rb5 closes and the endurance test passes, a fresh /review closes this card.
  timestamp: 2026-08-12T22:27:21.353673+00:00
- actor: claude-code
  id: 01kzwvzp8j3va40w86zz1y36pm
  text: |-
    ### The user skips the endurance test — 2026-08-13
    The user wrote: "ok, let's skip the endurance test. and that's slow enough that i think there is some tuning to do". Thus the >12k-token endurance criterion is skipped by the user's decision. The decode-speed tuning work stays in ^3gh7rb5. This card no longer waits for it.
  timestamp: 2026-08-13T06:12:07.570866+00:00
- actor: claude-code
  id: 01kzww3b8kj9zme74grkjdd26y
  text: |-
    ### review — clean
    - evidence: no re-run of the engine was necessary — no code changed since the clean engine pass on commit 4a390a4 (`git log 4a390a4..HEAD` excluding `.kanban` is empty; `git diff --stat 4a390a4..HEAD` excluding `.kanban` is empty; `git status` shows only this card's kanban files). That pass found 0 findings on changed lines; the one recorded item (`DeepSeekV4.swift:558`, `stackRoutedExperts`) sits on an untouched line and stays on record as out of scope per the standing scope rule from ^ag7ant0. The two open checklist items — the >12k-token acceptance criterion and the one-test-at-a-time run line — are now checked with the disposition: skipped by the user's decision of 2026-08-13 (comment 01kzwvzp8j3va40w86zz1y36pm); the decode-speed work is ^3gh7rb5. All other criteria were verified with real weights on 2026-08-12.
    - next: none for this card — moved to done. The decode-speed tuning continues in ^3gh7rb5.
  timestamp: 2026-08-13T06:14:07.379509+00:00
depends_on: []
position_column: done
position_ordinal: f080
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

- [x] New integration test loads the real repo id end to end with no missing/unexpected weight keys. (VERIFIED 2026-08-12 after the sanitize fix: `loadsTheRealCheckpointEndToEnd` passes in 21.8 s — model type `DeepSeekV4Model`, 43 layers, `verify [.all]`, thus no missing and no unexpected keys. The fix: `modulePath(of:)` now renames the checkpoint's `attn_hc.*`/`ffn_hc.*` to `hc_attn.*`/`hc_ffn.*` and `ffn.gate.e_score_correction_bias` to `ffn.gate.bias`, mirroring mlx-lm PR 1189; two unit tests pin the map.)
- [x] First N greedy token ids match a Python-generated fixture exactly. (VERIFIED 2026-08-12 with N = 32, the window the user approved: `greedyFirstTokensMatchThePythonFixture` passes in 94.6 s — 32 of 32 ids equal. The fixture was regenerated with the full-forward greedy method, because `generate_step`'s cached S=1 path diverges from the Python model's own forward by 2.0 logits at step 0. The measured stable window is 36 tokens; step 36 is an exact bf16 tie in the reference. The `parityTokenCount` constant in the test file records this reasoning.)
- [x] Both `chat` and `thinking` prompts generate successfully. (VERIFIED 2026-08-12: `chatAndThinkingModesBothGenerate` passes in 232 s. Thinking output reasons then answers; chat output: "The sea exhales a salty sigh against the shore.")
- [x] A >12k-token generation completes without a Metal buffer-count crash, or an equivalent buffer-stability assertion passes. (BLOCKED 2026-08-12 by decode speed: about 2.4 s/token, thus 12,400 tokens needs about 7 hours and exceeds the suite's 240-minute limit. A run was killed after 83 minutes in progress. See task ^3gh7rb5 for the performance work; re-run this test after that task lands.) (SKIPPED by the user's decision of 2026-08-13; the decode-speed work is ^3gh7rb5.)
- [x] The test skips cleanly (not fails) when the weights are absent or memory is insufficient, with a message saying why. (Verified: full suite run shows 5 clean skips with reason messages, 0 failures. Also verified 2026-08-12 with the weights present: a load failure gives a clean skip that carries the error text, not a test failure.)
- [x] Two-round prompt-cache behavior is asserted or explicitly documented as not-cacheable with the reason. (Verified at the encoder level: thinking mode is NOT prefix-cacheable across rounds, chat mode IS; mutation-proved. ALSO verified with real weights 2026-08-12: `twoRoundConversationRecallsTheFirstRound` passes in 47.7 s — round 1 "I have noted the number 4172.", round 2 "4172", thus the in-session KV cache holds across turns.)

## Tests

- [x] New `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift`. (The card's original `Tests/MLXLMIntegrationTests/` path is not buildable: that target does not exist and the root package deliberately has no swift-transformers dependency, thus it cannot load the real tokenizer. All real-weights suites live in the IntegrationTesting Xcode project; see the comment trail.)
- [x] Test: load plus assert model type and layer count (43). (Passes with real weights, 2026-08-12.)
- [x] Test: greedy first-N-token parity against the checked-in Python fixture. (Passes with real weights, 2026-08-12, N = 32 per the user's decision. The corrected full-forward fixture is in the tree at `IntegrationTesting/IntegrationTestingTests/Fixtures/deepseek-v4-flash-4bit-greedy-parity.json` and must be committed with this card.)
- [x] Test: chat vs thinking mode both produce non-empty output. (Passes with real weights, 2026-08-12.)
- [x] Test: long generation past 12k tokens completes (regression guard for the mlx-lm 1662 leak). (The test exists; it cannot finish inside the 240-minute limit at ~2.4 s/token — see ^3gh7rb5.)
- [x] Test: two-round conversation — assert cache hit, or assert-and-document no-hit. (The real-weights recall test also passes, 2026-08-12.)
- [x] Run: one test at a time via `xcodebuild test ... '-only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/<test>()'` (the user's instruction of 2026-08-12; a full-suite run wastes hours behind the long test). State 2026-08-12: load PASS 21.8 s, parity PASS 94.6 s (32/32), chat/thinking PASS 232 s, two-round recall PASS 47.7 s, encoder cache tests PASS. Only the long-generation test stays open, blocked by ^3gh7rb5. (The open long-generation run is skipped by the user's decision of 2026-08-13; the decode-speed work is ^3gh7rb5.)

## Workflow
- Use `/tdd` — write the load plus parity test first; it will fail until every preceding task is green, which is the point.
#deepseek-v4

## Review Findings (2026-08-12 17:22)

- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:558` — Function `stackRoutedExperts` has deeply nested loops (4 levels deep) with conditions at the nesting limit, making the control flow difficult to follow and reason about. Extract the innermost loops (projection+tensor loop at lines 563-575) into a separate helper function to reduce nesting to 2 levels, improving readability and testability.
  - Scope note: UNTOUCHED-LINE. Commit 4a390a4 changes `DeepSeekV4.swift` in two hunks only: the two path constants near line 455 and the rename block in `modulePath(of:)` at lines 531-549. The commit does not change `stackRoutedExperts`; it only moves its line number down. Thus this item is out of scope for this delta review, per the standing scope rule from ^ag7ant0. The item stays on record for a review that includes that function's changes.