---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kznzq1w1d3cc37ytbggywyn9
  text: |-
    Research notes for the next agent.

    Sources read (real paths, real SHAs — both public, MIT):
    - `osaurus-ai/vmlx-swift-lm` @ `b166896353b9c95d773de993990c20a0b5ba6905` — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift`, 393 lines. This is the attribution source that `CONTRIBUTING.md` names. Last commit for that path: "fix(dsv4): preserve jangtq-k routed bit plan", 2026-05-12.
    - `scouzi1966/mlx-swift-lm` @ `e1852869ce61ded0d23b76df3757e9b75c77c1f5` — same path, 437 lines. It is a transcription of the file above with the four `dspark_*` keys, an `activationQATEnabled` flag, and no `RuntimeMoETopKOverride`. Both files give the same `Copyright © 2026 Osaurus AI` line.

    Discoveries:
    - The `dspark_*` keys exist ONLY in the scouzi copy, not in the osaurus file. The new file thus names both sources in its header — osaurus in the block that `CONTRIBUTING.md` fixes, scouzi in a note below it. Both repositories are already in `THIRD-PARTY-NOTICES.md`, thus no notice entry was added.
    - `RuntimeMoETopKOverride` (used by the osaurus `init`) does not exist in this repository. The port reads `num_experts_per_tok` straight from the file.
    - Real `config.json` shape: `compress_ratios` holds 44 entries against 43 layers (the extra entry is the MTP layer). Layers 0 and 1 hold 0, layer 2 holds 4, layer 3 holds 128, layer 42 holds 4, layer 43 holds 0. `use_attn_sink` and every `dspark_*` key are absent, thus the tests examine the defaults for them.
    - The real file holds both `quantization` and `quantization_config`, each with 644 keys. Neither is decoded — the JANGTQ/MXTQ plans stay out of scope per this card.

    Not done, and why:
    - `activationQATEnabled` from the scouzi copy is not a `config.json` key (that copy hardcodes it to `true`). It belongs with the attention/indexer numerics, not with config decoding, thus it is left for the attention task.

    Deviation to note for review: the card names the helper `hasCompressor(_:)`. The implementation uses `hasCompressor(layer:)`, which is the reference's own signature, and which the Swift naming rule requires — a bare `Int` argument needs a noun that says what it is. `isHashLayer(_:)` and `ropeTheta(forLayer:)` match the card exactly.
  timestamp: 2026-08-10T14:02:37.825867+00:00
- actor: claude-code
  id: 01kznzqvyxccdr7s0etqxmdx7w
  text: |-
    ### implement — changed
    - evidence: 4 files — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (new), `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` (new, 9 tests), `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` (new fixture, a copy of the real `config.json`), `Package.swift` (the fixture added to the `MLXLMTests` resources).
    - TDD: the test came first and failed with "cannot find type 'DeepseekV4Configuration' in scope". After the struct landed, `swift test --filter DeepseekV4ConfigurationTests` gave 9 tests, 0 failures.
    - `swift build`: complete, no compiler warning. Full `swift test`: exit 0, 0 failures. The 2 skips are the pre-existing, documented skips in `CompiledDecodeCorrectnessTests`.
    - `swift-format lint --configuration .swift-format` is clean on the three changed Swift files, and a format run makes no further change.
    - `DeepseekV3.swift` is unmodified.
    - Note: a full-tree `swift-format` run reformatted 50 unrelated files (the installed swift-format is 603.0.0 and disagrees with the version that formatted the repository). Those files were reverted. Format only the files you change.
    - next: `/review`
  timestamp: 2026-08-10T14:03:04.541638+00:00
- actor: claude-code
  id: 01kzp0d2pg275cmw1hz37v3mzt
  text: |-
    ### review — findings
    - scope: `review sha 6abd9d5~1..6abd9d5` (round 1). Engine attempted 9 files, 0 failed, 0 skipped, 0 findings.
    - engine result: clean. The 5 findings come from the directed checks that the engine does not do (upstream comparison, live fixture fetch, test mutation).
    - field fidelity: PASS. 37 decoded keys, 0 mismatches. 33 keys match `osaurus-ai/vmlx-swift-lm` @ b1668963; the 4 `dspark_*` keys match `scouzi1966/mlx-swift-lm` @ e1852869. All 36 `Default` values match the reference. `rope_scaling` is the one optional, with no default, as in the reference.
    - cited SHAs: both are real commits. b1668963 = "fix(dsv4): preserve jangtq-k routed bit plan" (2026-05-12). e1852869 = "Optimize DeepSeek V4 MLX execution" (2026-08-03).
    - helpers: `isHashLayer(_:)`, `hasCompressor(layer:)`, `ropeTheta(forLayer:)` match the reference logic exactly, with the same out-of-bounds guard. The name `hasCompressor(layer:)` is correct and better than the `hasCompressor(_:)` on the card.
    - fixture: real. Live fetch from HuggingFace gave HTTP 200. All 48 top-level keys match with 0 differences.
    - attribution header: matches the block that `CONTRIBUTING.md` requires, word for word.
    - scope hygiene: exactly the 4 expected files. No unrelated reformatted file.
    - swift-format 603.0.0 gives a 0-byte diff on all 4 files of this commit.
    - evidence: Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift:38, :104, :114, Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:23, :236
    - next: strengthen the 9 new tests so a wrong default or a wrong key string makes them fail. Task stays in `review`.
  timestamp: 2026-08-10T14:14:39.568125+00:00
- actor: claude-code
  id: 01kzp0ee3nn3bqhrxfd5sth9bh
  text: |
    ### finish iteration 1 — findings
    - implement: changed — DeepseekV4Configuration.swift (37 keys, `Default` enum, 3 helpers), DeepseekV4ConfigurationTests.swift, DeepSeek-V4-Flash-4bit-config.json, Package.swift
    - test: green — `swift build` clean; `swift test --filter DeepseekV4ConfigurationTests` 9/9; full `swift test` exit 0
    - commit: 6abd9d5 feat(mlx-lm): add the DeepSeek-V4 configuration type; b6f585a (kanban chore, out of scope)
    - review: findings — DeepseekV4ConfigurationTests.swift:38, :104, :114; DeepseekV4Configuration.swift:23, :236 (5 findings)
    - PASSED, verified exhaustively: all 37 decoded keys and all 36 `Default` values match the references, 0 mismatches. The fixture is the real published config (live fetch, 48 keys, 0 differences). Both cited SHAs are real commits. The attribution header matches CONTRIBUTING.md.
    - MAIN FINDING: the 9 tests are vacuous. The `Default` table holds DeepSeek-V4-Flash values and the fixture is DeepSeek-V4-Flash, thus a broken decoder falls back to the same number the test asserts. Proof by mutation: 3 misspelled JSON keys gave 9 tests 0 failures; 3 corrupted defaults gave 9 tests 0 failures. 32 of 37 keys and 29 of 36 defaults have no protection. Only `compress_ratios` is truly covered, because its default `[]` differs from the fixture.
    - note: the review engine attempted 9 files here (0 failed, 0 skipped). The 0-file gap on the earlier task was specific to Markdown-only commits.
    - next: implement — make the tests discriminate
  timestamp: 2026-08-10T14:15:24.021890+00:00
- actor: claude-code
  id: 01kzp164n42228pdycct15fq0h
  text: |-
    Review round 1 findings closed. Notes for the next agent.

    What changed: only `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift`. No source file changed. `git diff --quiet Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` passes, thus all 37 decoded keys and all 36 `Default` values keep the values the reviewer verified against the references. `hasCompressor(layer:)` keeps its spelling.

    Shape of the new test file:
    - `ExpectedValues` — a private struct with one property for each of the 37 decoded keys, and one `assertMatches(_:)` that makes all 37 comparisons. No property has a default value, thus the memberwise initializer asks for all 37 and a table that forgets a key does not compile.
    - Three tables read that one method: `flashDefaults` (the 36 defaults), `distinctValues` (a value for each key that is different from the default of that key), and `publishedFixture` (`flashDefaults` plus the 44 compress ratios and the YaRN rope scaling, which are the only two keys where the real file is different from the defaults).
    - Tests: `testDistinctJSONDecodesEveryKey`, `testDistinctJSONSurvivesEncodeAndDecode`, `testMinimalJSONDecodesToFlashDefaults`, `testFixtureDecodesEveryKey`, plus the four helper tests. 8 tests in place of 9. `testDsparkKeysDecode` went out because `distinctJSON` gives all four `dspark_*` keys and reads them back.

    Why one `assertMatches` and three data tables, and not three blocks of assertions: three parallel blocks of 37 `XCTAssertEqual` lines that are different in their literals only are a near-verbatim duplicate, which the duplication rule calls a blocker. One method plus three tables is the data-driven shape.

    Mutation proof, run before this report. Each mutation was reverted and `git diff --quiet` confirmed clean after each one.
    1. The reviewer's three misspelled keys (`hidden_size_TYPO`, `num_attention_heads_TYPO`, `rope_theta_TYPO`): 8 tests, 6 failures (3 in the distinct test, 3 in the round-trip test). Round 1 gave 0.
    2. All 37 key strings misspelled at one time: 37 distinct failed assertion lines in `testDistinctJSONDecodesEveryKey` — one for each key. No key string is unprotected.
    3. The reviewer's three corrupted defaults (`numAttentionHeads = 999`, `ropeTheta = 777.0`, `indexTopk = 4242`): 8 tests, 3 failures, all in `testMinimalJSONDecodesToFlashDefaults`. Round 1 gave 0.
    4. All 36 defaults corrupted at one time: 36 distinct failed assertion lines in that same test. No default is unprotected.
    5. An `encode(to:)` that writes 2 of the 37 keys: 35 failures in `testDistinctJSONSurvivesEncodeAndDecode`. The encode path is real.

    Discovery worth keeping: exact `Float` equality is sufficient for every number in the tables. `1e-6` from the file and `1e-6` as a Swift `Float` literal compare equal, thus the earlier `accuracy: 1e-12` is not needed. The distinct floats (0.125, 0.25, 0.5, 0.75, 3.5, 4.5) are all exact in binary.

    Verification: `swift build` complete, no compiler warning. `swift test --filter DeepseekV4ConfigurationTests` gives 8 tests, 0 failures. Full `swift test` exits 0 with 0 failures; the 2 skips are the pre-existing documented skips in `CompiledDecodeCorrectnessTests`. `swift-format lint --configuration .swift-format` is clean on the one changed file and a format run makes no further change. No tree-wide format run was made.
  timestamp: 2026-08-10T14:28:20.772555+00:00
- actor: claude-code
  id: 01kzp16akdqavzxtat3ve7bdvx
  text: |-
    ### implement — changed
    - evidence: 1 file — `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift`. 5 of 5 review findings closed. Mutation proof: 37 of 37 key strings and 36 of 36 defaults now make a test fail; an encoder that drops keys makes the round-trip test fail. `swift build` clean; `swift test --filter DeepseekV4ConfigurationTests` 8 tests 0 failures; full `swift test` exit 0, 0 failures, 2 pre-existing documented skips.
    - next: `/review`
  timestamp: 2026-08-10T14:28:26.861632+00:00
- actor: claude-code
  id: 01kzp1zv8fc07n62bgeb44akc7
  text: |-
    ### review — clean (round 2)

    - scope: `review sha 67f02f3~1..67f02f3`. Commit `67f02f3` changes one file only: `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` (312 added, 61 deleted).
    - engine result: 0 findings, **0 attempted files**. The engine does not read test files. A direct `review file` on the test path also gives 0 attempted, while `review file` on `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` gives 9 attempted. Thus the engine gives no signal on a test-only commit, and the checks below are the evidence.
    - source untouched: `git diff --quiet 67f02f3~1 67f02f3 -- Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` exits 0. The 37 keys and 36 defaults from round 1 stay valid.
    - mutation tests, run again by hand, each one reverted:
      - 3 misspelled `CodingKeys` strings (`qk_rope_head_dim`, `norm_topk_prob`, `routed_scaling_factor`): each gives 2 failures in `testDistinctJSONDecodesEveryKey` and `testDistinctJSONSurvivesEncodeAndDecode`.
      - 3 corrupted defaults (`oGroups`, `indexTopk`, `dsparkMarkovRank`): each gives a failure in `testMinimalJSONDecodesToFlashDefaults`.
      - an `encode(to:)` that writes 2 of 37 keys gives 35 failures, all in `testDistinctJSONSurvivesEncodeAndDecode`. The shipped encoder is synthesized; the file has no hand-written `encode(to:)`.
    - `ExpectedValues`: 37 stored properties, and `assertMatches(_:)` compares all 37. No property is declared and left out of the comparison. The memberwise initializer has no default values; removal of one field from a table gives `error: missing argument for parameter 'swigluLimit'`.
    - `testFixtureDecodesEveryKey` compares `ropeScaling` as a whole dictionary (5 entries) and `compressRatios` as a whole 44-element array. Neither is a count or a prefix.
    - exact `Float` equality is safe. All 6 float properties are `Float`. The distinct values are dyadic (0.25, 0.5, 0.75, 0.125, 3.5, 4.5). `1e-6` is not exact in binary32, but the JSON text and the Swift literal give the same bits, and the narrowing from `Double` adds no second rounding.
    - test count 9 to 8 is a deliberate merge. 4 tests removed, 3 added. All 20 assertions of the 3 removed `testFixtureDecodes*` tests are inside the 37 comparisons of `testFixtureDecodesEveryKey`; `testDsparkKeysDecode` is replaced by `distinctJSON`. No test loses its checks.
    - `swift test --filter DeepseekV4ConfigurationTests`: 8 tests, 0 failures. Working tree clean after all mutation tests.
    - next: none for this task. See the note on out-of-scope naming findings in the review reply.
  timestamp: 2026-08-10T14:42:23.119749+00:00
- actor: claude-code
  id: 01kzp21kxjnp9cre2x10a318v8
  text: |
    ### finish iteration 2 — clean
    - implement: changed — DeepseekV4ConfigurationTests.swift only. No source file changed.
    - test: green — `swift build` clean; filter 8 tests 0 failures; full `swift test` exit 0
    - commit: 67f02f3 test(mlx-lm): fix vacuous DeepseekV4Configuration tests; 8118e0b (kanban chore)
    - review: clean — zero new findings. All 5 findings closed. Task moved to `done`.
    - verification: the reviewer re-ran the mutations independently. 3 misspelled keys give 2 failures each; 3 corrupted defaults each fail `testMinimalJSONDecodesToFlashDefaults`; a broken `encode(to:)` gives exactly 35 failures. `ExpectedValues` has 37 properties and 37 comparisons, with a compile-time guard that a missing field breaks the build. `rope_scaling` and the 44 compress ratios compare deeply. Exact `Float` equality is safe, because all values are dyadic and the type has no `Double`.
    - note: 9 tests became 8 by a deliberate merge. 4 removed, 3 added, and no check was lost.
    - note: the review engine excludes test files. A commit that changes only test files always gives 0 attempted files. This is not a pass.
  timestamp: 2026-08-10T14:43:21.138930+00:00
depends_on:
- 01KZGMN4FQKCVPAYJDJJHK0APK
position_column: done
position_ordinal: dd80
title: Port DeepseekV4Configuration (deepseek_v4 config decoding)
---
## What

Create `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` — the `Codable` config struct for `model_type == "deepseek_v4"`. Foundation for every other DSV4 task; no model code yet.

Port from `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (437 lines). Cross-check key names against the real checkpoint: `https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/raw/main/config.json`.

Verified target values from that config.json: `hidden_size=4096`, 43 layers, 64 attention heads, `num_key_value_heads=1`, `head_dim=512`, `max_position_embeddings=1048576`, 256 routed experts, 6 experts per token, 1 shared expert, q/o LoRA rank 1024.

Config keys the reference decodes (from its `CodingKeys`): `hc_sinkhorn_iters`, `hc_eps`, `rope_theta`, `compress_rope_theta`, `rope_scaling`, `sliding_window`, `compress_ratios`, `index_n_heads`, `index_head_dim`, `index_topk`, `use_attn_sink`, `dspark_block_size`, `dspark_noise_token_id`, `dspark_target_layer_ids`, `dspark_markov_rank`, plus nested quant plans (`routed_expert`, `default_bits`, `routed_layer_bits`, `routed_expert_bits`, `mxtq_bits`, `routed_expert_bit_plan`, `routed_experts`, `bit_plan`).

Scope decisions for this task:
- **Include** the `dspark_*` keys as decoded-but-unused stored properties so unknown-key handling never trips; DSpark behavior itself is out of scope (see the out-of-scope task).
- **Skip** the `mxtq_bits` / JANGTQ nested plans entirely — that is a separate quant format we do not support (out of scope).
- Provide the reference's derived helpers: `isHashLayer(_:)`, `hasCompressor(_:)`, `ropeTheta(forLayer:)`. Per-layer theta selects `rope_theta` when `compress_ratio == 0` and `compress_rope_theta` otherwise.
- Follow this repo's existing config idiom — see `Libraries/MLXLLM/Models/DeepseekV3.swift` for the sibling `DeepseekV3Configuration`. Do **not** modify `DeepseekV3.swift`; it still serves DSV3/DSV3.2/Kimi/GLM-5.1/Nemotron bundles.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (MIT).
- Upstream copyright per file header: Osaurus AI, `SPDX-License-Identifier: MIT`. Apply the header decided in task `jhk0apk`.
- Gap tracker cross-reference: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md`.

## Acceptance Criteria

- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` exists with `public struct DeepseekV4Configuration: Codable, Sendable`.
- [x] Decoding the real `config.json` from `mlx-community/DeepSeek-V4-Flash-4bit` succeeds and yields the verified values above.
- [x] `isHashLayer`, `hasCompressor`, and `ropeTheta(forLayer:)` return correct values for layers 0, 2, 3, and 42.
- [x] Absent/optional keys fall back to documented defaults rather than throwing.
- [x] `DeepseekV3.swift` is unmodified (`git diff --stat` shows no change to it).

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` with a checked-in fixture copy of the model's `config.json` (weights not needed).
- [x] Test: decodes the fixture; asserts hiddenSize 4096, 43 layers, 64 heads, kvHeads 1, headDim 512, maxPositionEmbeddings 1048576, 256 routed experts, 6 per token, 1 shared.
- [x] Test: `isHashLayer(0..2) == true`, `isHashLayer(3) == false`.
- [x] Test: `ropeTheta(forLayer:)` picks `compress_rope_theta` where `compress_ratios[i] != 0` and `rope_theta` where it is 0.
- [x] Test: a minimal JSON with only required keys decodes without throwing.
- [x] Run: `swift test --filter DeepseekV4ConfigurationTests` — all pass.

## Workflow
- Use `/tdd` — write the failing decode tests against the real fixture first, then implement the struct.
#deepseek-v4

## Review Findings (2026-08-10 09:14)

- [x] `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift:38` — The 9 new tests do not fail when a JSON key string in `CodingKeys` is wrong. Proof by mutation: a change of `hidden_size` to `hidden_size_TYPO`, `num_attention_heads` to `num_attention_heads_TYPO`, and `rope_theta` to `rope_theta_TYPO` kept all 9 tests green. The cause is that each `Default` constant holds the same number as the fixture, thus `decodeIfPresent` gives nil, the default applies, and the default equals the number the test asserts. Remove this cause for the whole file: assert each value against the number read from the fixture JSON, not against a number that the `Default` table also holds.
  - Closed: the file now holds `distinctJSON`, which gives each of the 37 keys a value that is different from the default of that key. `testDistinctJSONDecodesEveryKey` reads all 37 values back, thus no assertion can pass through a default. Proof by mutation: the reviewer's three misspelled keys give 6 failures. A run with all 37 key strings misspelled gives 37 distinct failed assertion lines in that one test — one for each key.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:236` — The 9 new tests do not fail when a `Default` constant is wrong. Proof by mutation: `numAttentionHeads = 999`, `ropeTheta = 777.0`, and `indexTopk = 4242` kept all 9 tests green. 29 of the 36 `Default` constants have no test that pins them. Only `testMinimalJSONDecodesToFlashDefaults` pins a default, and it pins 7.
  - Closed: `testMinimalJSONDecodesToFlashDefaults` now reads the `ExpectedValues.flashDefaults` table, which holds all 36 defaults, and asserts `nil` for `ropeScaling`, which has no default. Proof by mutation: the reviewer's three corrupted defaults give 3 failures. A run with all 36 defaults corrupted gives 36 distinct failed assertion lines in that one test.
- [x] `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift:104` — 32 of the 37 decoded keys have no assertion that can fail on a wrong key string. Only `compress_ratios` is truly protected, because it is the one key whose `Default` (`[]`) differs from the fixture value. These 5 keys have no reference in any test: `qk_rope_head_dim`, `rms_norm_eps`, `o_groups`, `norm_topk_prob`, `routed_scaling_factor`. Add an assertion for each decoded key.
  - Closed: `ExpectedValues` holds one property for each of the 37 decoded keys, and `assertMatches(_:)` compares all 37. Three tables read it: the distinct values, the 36 defaults, and the published fixture. The memberwise initializer takes no default value, thus a table that forgets a key does not compile.
- [x] `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift:114` — `rope_scaling` is present in the fixture, but the only assertion on it is `XCTAssertNil` from the minimal JSON. No test asserts that it decodes to a non-nil value with the expected content from the fixture. Add that assertion.
  - Closed: `ExpectedValues.publishedRopeScaling` holds the five entries of the fixture (`beta_fast` 32, `beta_slow` 1, `factor` 16, `original_max_position_embeddings` 65536, `type` "yarn"), and `testFixtureDecodesEveryKey` compares the whole dictionary. `distinctValues` also gives `rope_scaling` a non-nil value of its own.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:23` — The type declares `Codable`, but no test runs `encode(to:)`. The synthesized encoder is never exercised. Add an encode-and-decode test that shows a round trip keeps every value.
  - Closed: `testDistinctJSONSurvivesEncodeAndDecode` encodes the distinct configuration, decodes the result, and compares all 37 values. Proof by mutation: an `encode(to:)` that writes 2 of the 37 keys gives 35 failures in that test.