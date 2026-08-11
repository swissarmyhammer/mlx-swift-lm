---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp968sph15apw2exds16qh3
  text: |-
    ### Research

    Read the fixture `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` and the plumbing. Facts:

    - The `quantization` block holds 3 scalar keys (`group_size: 64`, `bits: 4`, `mode: "affine"`) and 641 per-layer keys. No `false` (skip) entry.
    - 129 keys are `{group_size: 32, bits: 4, mode: "mxfp4"}` = 43 layers x 3 projections, all named `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj`.
    - 512 keys are `{group_size: 64, bits: 4, mode: "affine"}`: `model.embed_tokens`, `lm_head`, `model.layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}`, `model.layers.N.attn.compressor.{wgate,wkv}`, `model.layers.N.attn.indexer.{wq_b,weights_proj}`, `model.layers.N.attn.indexer.compressor.{wgate,wkv}`, `model.layers.N.ffn.shared_experts.{gate,up,down}_proj`.
    - The router (`model.layers.N.ffn.gate`) is absent from the block, thus it stays in high precision.
    - `QuantizationMode` is `String, Codable` with raw values `affine`/`mxfp4`/`mxfp8`/`nvfp4`, so `mode` decodes with no new code.
    - `MLX.quantized(_:groupSize:bits:mode:)` returns `biases` as `MLXArray?` and gives `nil` for mxfp4 (the C vector holds 2 arrays, not 3). `QuantizedSwitchLinear.biases` is already `MLXArray?`, thus it stores `nil`.
    - `Load.swift` `loadWeights` gates on `weights["<path>.scales"] != nil` and then hands `perLayerQuantization.quantization(layer: path)?.asTuple` to `quantize(model:)`, which threads `mode` through.

    The path strings in the block are checkpoint key paths. The Swift module tree must give the same flattened paths (`ffn`, `switch_mlp`, `attn`, `wq_a`, ...) for the plan to resolve. This is a constraint the MoE/model task must meet; the tests here examine it with a probe module tree that uses those `@ModuleInfo` keys.
  timestamp: 2026-08-10T16:48:13.622503+00:00
- actor: claude-code
  id: 01kzp9peaa7ep0ymrt60sb5efx
  text: |-
    ### Result: the generic path already resolves the mixed plan -- no production change

    (Key counts corrected in review round 1: the two numbers below were layer counts with a key-count label.)

    Eight new tests in `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift`. Seven passed on the first run against unchanged production code, which is the finding the card asked for. `Libraries/` held no diff at this point.

    **One test went red first, and it was a fixture fact, not a code gap.** The first draft examined `model.layers.0.attn.compressor.wkv`. That key is absent: layers 0 and 1 hold a compress ratio of 0 and thus hold no compressor, and layer 2 is the first that does. The block names a compressor for the 41 layers 2 to 42 -- **82 keys**, `wgate` and `wkv` on each -- and an indexer for the 21 even layers 2, 4, ... 42 -- **84 keys**, `wq_b`, `weights_proj`, `compressor.wgate` and `compressor.wkv` on each -- which is where `compress_ratios` holds 4. The test now names layer 2 and passes. The MoE and attention tasks must build the compressor and the indexer conditionally on the same rule; the fact now sits on `^tty95f4`, `^r92pjcr` and `^ag7ant0` as well.

    **No reference conflict.** The card, the fixture and the code agree on every point examined. Nothing needed adjudication against the Python.

    **Deferred: fused mxfp4 Metal kernels.** `scouzi1966/mlx-swift-lm`'s `deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and `deepseek_v4_native_mxfp4_down_sum6` are a throughput optimization, not a correctness requirement. The generic `gatherQuantizedMM` path gives correct mxfp4 output, as `mxfp4SwitchLinearMatchesItsOwnDequantizedWeights` shows. `Libraries/MLXLMCommon/SwitchLayers.swift` is unmodified (0 lines, against the ~40 line limit), and no Metal source was added. File a performance task if profiling later justifies the kernels.

    **Access level.** `QuantizedSwitchLinear` keeps `weight`, `scales` and `biases` internal, thus the test file uses `@testable import MLXLMCommon`, which is the pattern `MiniMaxM3Tests` already uses. No production access level was widened.
  timestamp: 2026-08-10T16:57:03.562907+00:00
- actor: claude-code
  id: 01kzp9pwdwpx54h0454ezh961t
  text: |-
    ### Mutation proof (the tests are not vacuous)

    The review engine leaves test files out, thus each new test was examined against a broken production file. Two mutations, each reverted after the run.

    **Mutation 1** -- `Libraries/MLXLMCommon/BaseConfiguration.swift`: `public var mode: QuantizationMode { _mode ?? .affine }` changed to `{ .affine }`, which drops the decoded `mode`.

    Result: 2 of 8 tests failed with 136 issues.
    - `fixtureGivesEveryExpertProjectionMxfp4` -- 130 issues, one for each of the 129 `resolved.mode == .mxfp4` expectations plus the count.
    - `planAppliesMxfp4ToExpertsAndAffineToEverythingElse` -- 6 issues: `layer.mode == .mxfp4` and `layer.biases == nil` on all three expert projections.

    **Mutation 2** -- `Libraries/MLXLMCommon/SwitchLayers.swift`: `MLX.quantized(other.weight, groupSize: groupSize, bits: bits, mode: mode)` changed to `mode: .affine`, which packs the weights the wrong way while the layer keeps reporting `.mxfp4`.

    Result: 3 tests failed.
    - `planAppliesMxfp4ToExpertsAndAffineToEverythingElse` -- 3 issues, `layer.biases == nil` on each expert projection.
    - `mxfp4SwitchLinearRunsWithScalesAndNoBiases` -- `layer.biases == nil` failed, then mlx aborted the run: `[gather_qmm] Scale type must be uint8 but received type float32`.
    - `mxfp4SwitchLinearMatchesItsOwnDequantizedWeights` -- reached by the same abort.

    Both mutations reverted. `git status --porcelain -- Libraries/ Tests/` then gave one line, the new untracked test file, and `git diff --stat -- Libraries/` gave nothing.

    ### Verification

    - `swift build` -- exit 0.
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 8 tests in 1 suite passed.
    - `swift test --filter Quantiz` -- 13 tests in 2 suites passed (this suite plus the pre-existing quantization tests), no regression.
    - `swift test` -- exit 0. Swift Testing: 378 + 80 + 282 + 7 tests passed. XCTest: 0 failures across every suite. The only 2 skips are the pre-existing `CompiledDecodeCorrectnessTests` skips (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ), untouched.
    - `swift format` run on the one touched file only, with the repo `.swift-format`. It made no change.
  timestamp: 2026-08-10T16:57:18.012820+00:00
- actor: claude-code
  id: 01kzp9qv8x696hp0ztx322fh05
  text: |-
    ### implement — changed
    - evidence: 1 file — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift (new, 8 tests). No production change; `git diff --stat -- Libraries/` is empty. Mutation proof: BaseConfiguration `mode` forced to `.affine` killed 2 tests with 136 issues; SwitchLayers `MLX.quantized(mode:)` forced to `.affine` killed 3 tests (one through an mlx `[gather_qmm] Scale type must be uint8` abort). Both reverted, tree clean. `swift build` exit 0; `swift test --filter DeepseekV4QuantizationPlanTests` 8/8 pass; `swift test --filter Quantiz` 13/13 pass; full `swift test` exit 0, zero failures.
    - next: /review
  timestamp: 2026-08-10T16:57:49.597527+00:00
- actor: claude-code
  id: 01kzpa85tz5v5sn377f8qnmwkc
  text: |-
    ### review — findings

    - evidence: 6 findings — Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift:45, :261, :364; Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json; Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:297, :298. Engine `review sha 8bb4936~1..8bb4936` gave 0 findings on 0 attempted files, because the commit adds a test file only and the engine leaves test files out. Thus every point below was examined by hand.

    **Round 1, hand examination.**

    1. The central claim holds. The existing `BaseConfiguration` and `SwitchLayers` machinery does resolve the mixed plan, and the two mutations reproduce exactly. Mutation A, `mode` forced to `.affine` in `BaseConfiguration.swift`: 2 tests failed with 136 issues, 130 in `fixtureGivesEveryExpertProjectionMxfp4` and 6 in `planAppliesMxfp4ToExpertsAndAffineToEverythingElse`. Mutation B, `MLX.quantized(mode:)` forced to `.affine` in `SwitchLayers.swift`: 3 issues in `planApplies...`, then `layer.biases == nil` failed and mlx aborted the run with `[gather_qmm] Scale type must be uint8 but received type float32`. Both reverted. `git status --porcelain -- Libraries/` and `git diff --stat -- Libraries/` are both empty, and the suite is 8/8 green again.

    2. A third mutation shows one gap. The whole `else` branch of `PerLayerQuantization.quantization(layer:)` changed to `return nil` keeps all 8 tests green. The suite thus pins no part of the production filter in `Load.swift`, and the "router stays in high precision" result comes from the test's own helper, not from the code that runs at load time.

    3. Every fixture fact is correct. Counted directly: default `(64, 4, affine)`; 641 per-layer keys; exactly 2 distinct tuples, `(32, 4, mxfp4)` on 129 keys and `(64, 4, affine)` on 512; the 129 mxfp4 keys equal the set `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj` for N in 0 to 42, with none missing and none extra; no `model.layers.N.ffn.gate` key.

    4. The layers 0 and 1 result is correct and is under-recorded. `compress_ratios` gives 0 to layers 0, 1 and to entry 43, and the compressor keys equal the nonzero-ratio layers exactly, and the indexer keys equal the ratio-4 layers exactly. `hasCompressor(layer:)` already holds the rule. The recorded key counts are wrong, and the downstream cards do not hold the fact. See the findings.

    5. The probe `@ModuleInfo` keys are real checkpoint key paths, with one exception: `model.layers.0.attn.compressor.*`, which the fixture does not name. See the findings.

    6. The deferral of the fused mxfp4 Metal kernels is genuine. The card states it under "Explicitly out of scope", it names both kernels and the three env knobs, and acceptance criterion 6 asks only for a note. This is not a silent narrowing.

    - next: /implement — resolve the 6 findings, then /review round 2.
  timestamp: 2026-08-10T17:06:44.703494+00:00
- actor: claude-code
  id: 01kzpa9egeaf5jv9h3r3fj4hn5
  text: |
    ### finish iteration 1 — findings
    - implement: changed — DeepseekV4QuantizationPlanTests.swift (8 tests). No production change.
    - test: green — `swift build` exit 0; filter 8/8; `--filter Quantiz` 13/13; full `swift test` exit 0
    - commit: 8bb4936 test(mlx-lm): lock in mixed affine+mxfp4 quantization plan for deepseek_v4; b62711c (kanban chore)
    - review: findings — DeepseekV4QuantizationPlanTests.swift:45, :261, :364; DeepSeek-V4-Flash-4bit-config.json; DeepseekV4Configuration.swift:297, :298 (6 findings)
    - CONFIRMED GOOD: the central claim holds. The existing machinery does resolve the mixed plan. The reviewer reproduced both mutations exactly (136 issues, and the `[gather_qmm] Scale type must be uint8` abort), and the tree is clean after the revert. Every fixture fact is exact: 641 per-layer keys, 129 mxfp4, 512 affine, no router key. The kernel deferral is a real scope-out on the card.
    - MAIN FINDING: the test helper puts plan membership in place of the production gate. Production gates on `weights["<path>.scales"] != nil` in `Load.swift:38`, then falls back to the default in `BaseConfiguration.swift:97`. A third mutation, which changed the whole `else` branch to `return nil`, keeps all 8 tests green. The suite pins no part of the production filter.
    - note: the engine attempted 0 files, because the commit adds only a test file. This is not a pass.
    - next: implement — fix all 6 findings
  timestamp: 2026-08-10T17:07:26.350192+00:00
- actor: claude-code
  id: 01kzpb6mxbpgbqf8c7fweqj9s8
  text: |-
    ### Review round 1 -- all 6 findings closed, and one production change was genuinely needed

    **Finding 1 needed production code, and this is stated openly rather than hidden in a test.** The gate the review asked the tests to run lived inside the body of `loadWeights`, thus no test could reach it. It moved out, unchanged, into a package function:

    `Libraries/MLXLMCommon/Load.swift` -- `package func quantizationParameters(forPath:weights:quantization:perLayerQuantization:) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?`

    `loadWeights` now hands that function to `quantize(model:filter:)`, thus the tests and the load path run one body and cannot drift. The behaviour is the same, term for term: no `<path>.scales` array gives `nil`; a plan gives `plan.quantization(layer: path)?.asTuple`; no plan gives `quantization?.asTuple`. `package` matches `safetensorWeightURLs` beside it, which `LoadWeightsTests` already reads.

    **Test changes.** The helper `quantizationTuple(for:plan:)` is deleted. `stubScales(for:)` builds the stand-in weight dictionary, and `leavesAfterLoadFilter(plan:scalePaths:)` is the one place that calls the production filter. Five tests are new or reworked:

    - `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` -- expert projections mxfp4 at group size 32 with `biases == nil`; `wq_a`, `shared_experts.gate_proj` and `layers.2.attn.compressor.wkv` affine at group size 64; embeddings affine.
    - `loadFilterLeavesAPathWhoseWeightsHoldNoScalesAlone` -- the router holds no `scales` array, thus the first part of the gate leaves it a plain `Linear`.
    - `loadFilterGivesTheDefaultToAPathThePlanDoesNotName` -- give the router a `scales` array and it quantizes at the plan's own `(64, 4, affine)` default. This test is what kills the reviewer's third mutation.
    - `loadFilterQuantizesNothingWhenNoWeightsHoldScales` -- an empty weight dictionary leaves every leaf in high precision.
    - `fixtureNamesCompressorKeysFromLayerTwoUp` and `fixtureNamesIndexerKeysOnTheEvenLayersFromTwoUp` -- measure the 82 and 84 key counts and the layer sets, thus the corrected numbers are checked by a run and not by a reader.

    **Probe tree.** Three layers now, with `ProbeAttention.compressor` optional and present only from layer 2 up, which is what the fixture names. An assertion reads it.

    **Discovery worth keeping.** The mxfp4 forward tests and the plan tests were already sharp, but nothing in the suite reached `Load.swift` at all. A filter written inline in a function body is unreachable by a test, and that is the whole cause behind findings 1 and 2. The extraction is the fix for the cause, not for the one line.
  timestamp: 2026-08-10T17:23:23.179529+00:00
- actor: claude-code
  id: 01kzpb76fk6a4mvnqqnzz7km8c
  text: |-
    ### Mutation proof, round 2 -- all three mutations now FAIL

    Each mutation was applied alone, run, and reverted.

    1. `Libraries/MLXLMCommon/BaseConfiguration.swift` -- `public var mode: QuantizationMode { _mode ?? .affine }` changed to `{ .affine }`. **FAILS**: 2 tests, 136 issues -- `fixtureGivesEveryExpertProjectionMxfp4` 130 issues, `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` 6 issues.
    2. `Libraries/MLXLMCommon/SwitchLayers.swift` -- `MLX.quantized(other.weight, groupSize: groupSize, bits: bits, mode: mode)` changed to `mode: .affine`. **FAILS**: `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` 3 issues on `layer.biases == nil`, then `mxfp4SwitchLinearRunsWithScalesAndNoBiases` failed the same expectation and mlx aborted the run with `[gather_qmm] Scale type must be uint8 but received type float32`.
    3. `Libraries/MLXLMCommon/BaseConfiguration.swift` -- the whole `else` branch of `PerLayerQuantization.quantization(layer:)` changed to `return nil`. **FAILS**: `loadFilterGivesTheDefaultToAPathThePlanDoesNotName`, 1 issue -- `leaves[routerPath] as? QuantizedLinear` gave nil. This mutation kept all 8 of the round-1 tests green.

    After the three reverts, `git status --porcelain -- Libraries/` gives one line, ` M Libraries/MLXLMCommon/Load.swift`, which is the intended extraction. `BaseConfiguration.swift` and `SwitchLayers.swift` are clean.

    ### Verification

    - `swift build` -- exit 0. The only warnings are the pre-existing ones in `Qwen2VL.swift`, `Qwen25VL.swift` and `Gemma4.swift`, untouched.
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 13 tests in 1 suite passed.
    - `swift test` -- exit 0, zero failures. Swift Testing: 383 + 282 + 80 + 7 tests passed. XCTest: 0 failures in every suite. The only 2 skips are the pre-existing `CompiledDecodeCorrectnessTests` skips (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ), untouched.
    - `swift format --in-place --configuration .swift-format` on the two touched files only. No tree-wide format run.
  timestamp: 2026-08-10T17:23:41.171284+00:00
- actor: claude-code
  id: 01kzpb7f356g3dgex5kvrd0vx6
  text: |-
    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift (load-time filter extracted to `package func quantizationParameters(forPath:weights:quantization:perLayerQuantization:)`, +37/-10), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift (13 tests, +230/-52 across both files). All 6 review findings closed. 3-mutation proof: BaseConfiguration `mode` -> `.affine` FAILS (2 tests, 136 issues); SwitchLayers `MLX.quantized(mode:)` -> `.affine` FAILS (3 issues plus the `[gather_qmm] Scale type must be uint8` abort); BaseConfiguration `quantization(layer:)` `else` -> `return nil` FAILS (1 issue). All three reverted; `git status --porcelain -- Libraries/` then shows only the intended `Load.swift` change. `swift build` exit 0; `swift test --filter DeepseekV4QuantizationPlanTests` 13/13; full `swift test` exit 0, zero failures, 2 pre-existing skips.
    - next: /review round 2
  timestamp: 2026-08-10T17:23:49.989437+00:00
- actor: claude-code
  id: 01kzpbzwczga9crhanb0e6aybr
  text: |
    ### review — findings

    - scope: `review sha fa51cba~1..fa51cba` (round 2). The engine attempted 9 files, skipped 0, failed 0. Round 1 could not reach production code; this round did.
    - counts: 3 findings, 3 confirmed, 1 refuted.
    - evidence: `Libraries/MLXLMCommon/Load.swift:15`, `Libraries/MLXLMCommon/Load.swift:26`, `Libraries/MLXLMCommon/Load.swift:81`.

    #### Round 1 findings: all six confirmed closed

    Read each of the three downstream cards. `^tty95f4` holds the layers-0-and-1 rule, the "submodule must be absent" wording, the 82 keys across 41 layers, and the 44-against-43 fact. `^r92pjcr` holds the 84 keys across the 21 even layers, the missing `hasIndexer(layer:)` gap, and the 44-against-43 fact. `^ag7ant0` holds both facts. Findings 2, 3 and 4 read correct in the test file: the `stubScales(for:)` document comment states the scales assumption in as many words, `ProbeAttention` holds `ProbeCompressor?` and builds it only from layer 2 up with `model.layers.2.attn.compressor.wkv` asserted, and the counts read 82 and 84.

    #### The extraction is behaviour preserving

    Compared the new `quantizationParameters(forPath:weights:quantization:perLayerQuantization:)` body against the closure it came out of. The two-part gate keeps its order: the `weights["\(path).scales"] != nil` test first, the plan second. `guard ... else { return nil }` gives the same result as the old `else { return nil }`, and the `if let perLayerQuantization { return ... }` with a fall-through `return quantization?.asTuple` gives the same result as the old `if/else`. `loadWeights` keeps its `quantization != nil || perLayerQuantization != nil` guard and calls the new function for each path. `weights` moved from a captured value to a parameter; a Swift dictionary is a value type, thus this is the same read.

    `package` is the correct access level and matches `safetensorWeightURLs` in the same file. The test target sits in the same package, thus it can call it.

    #### The three mutations were run again

    - `BaseConfiguration` `mode` forced to `.affine`: 136 issues. Fails.
    - `SwitchLayers` `MLX.quantized(mode:)` forced to `.affine`: MLX aborts with "[gather_qmm] Scale type must be uint8 but received type float32". Fails.
    - `quantization(layer:)` `else` changed to `return nil`: `loadFilterGivesTheDefaultToAPathThePlanDoesNotName` fails at `DeepseekV4QuantizationPlanTests.swift:386`. This mutation kept all 8 round-1 tests green; it is now caught.

    Each mutation was reverted. `git status` and `git diff fa51cba` over `Libraries` and `Tests` are both empty.

    #### No other model changed

    `loadWeights` has five production call sites -- `Libraries/MLXVLM/VLMModelFactory.swift:434`, `Libraries/MLXEmbedders/ModelFactory.swift:220`, `Libraries/MLXLLM/LLMModelFactory.swift:736`, `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift:95` and `Libraries/MLXLMCommon/ModelConversion.swift:209`. All five reach the filter through the same call, thus a behaviour-identical extraction leaves them alone. `Tests/MLXLMTests/Gemma4KVSharedLoadTests.swift`, `Tests/MLXLMTests/GLM4LmHeadTiedLoadTests.swift`, `Tests/MLXLMTests/BaseConfigurationTests.swift` and `Tests/MLXLMTests/MiniMaxM3Tests.swift` read this path and all pass. The full `swift test` run passes 282 tests in 44 suites and 7 tests in 5 suites, with no failure and no warning.

    - next: correct the three `Load.swift` findings, then run `/review` again.
  timestamp: 2026-08-10T17:37:10.047683+00:00
- actor: claude-code
  id: 01kzpcq7jr493sy85jx6038aq8
  text: |
    ### Review round 2 -- all 3 findings closed in `Load.swift`

    All three are pre-existing code that the engine reached because round 1 touched production code for the first time. Each cause was removed for the whole file, not only for the flagged line.

    **Finding 1, missing documentation.** `safetensorWeightURLs` now holds a `///` comment in the file's own style. The same cause elsewhere in the file: `SafetensorsIndex` was undocumented and now is documented, and the new error type and its case are documented. `quantizationParameters` and `loadWeights` already were.

    **Finding 2, the force unwrap.** The two options the finding gives are a throw or an empty array. The empty array swallows the failure, which the card forbids, thus the throw. The file reports every other failure by `throws` and held no error type of its own, so a file-local one was added:

    `package enum WeightLoadingError: LocalizedError, Equatable { case unreadableModelDirectory(URL) }`

    That matches the pattern already in this module -- `KVCacheError` in `KVCache.swift`, `UserInputError` in `UserInput.swift`, `ModelConversionError` in `ModelConversion.swift`. `package` matches `safetensorWeightURLs` beside it.

    **The `nil` branch cannot be reached from a real call.** A probe program gives a non-nil enumerator for an absent directory, a plain file and an `https:` URL alike:

    ```
    missing dir non-nil / a plain file non-nil / http url non-nil / real dir non-nil
    ```

    `FileManager.enumerator(at:)` declares an Optional result, thus the guard is correct and necessary, but no test can drive it without a `FileManager` parameter on the production signature. That would put test-only surface on the load path of every model in the repository, so it was not added. **No test for the throw. This is stated openly rather than hidden.**

    **What is tested instead is the behaviour the card asks to preserve.** Three new tests in `Tests/MLXLMTests/LoadWeightsTests.swift`:

    - `testSafetensorWeightURLsFindsEverySafetensorsFileWhenNoIndexIsPresent` -- the branch that held the force unwrap, with a subdirectory, thus the recursive walk is held still.
    - `testSafetensorWeightURLsGivesAnEmptyListForADirectoryWithNoWeightFiles` -- an empty result is not an error.
    - `testSafetensorWeightURLsGivesAnEmptyListForADirectoryThatIsAbsent` -- an absent directory is still an empty list, thus the guard did not turn a quiet case into a throw.

    All three were written first and run against the file **before** the edit: 4 of 4 passed. They passed again after it. That is the evidence that the change is behaviour preserving, which is what a refactor of a shared path needs; a red-first test was not available, because the change adds no reachable behaviour.

    **Finding 3, the empty-collection form.** `var weights = [String: MLXArray]()` and the line below it, `var metadata = [String: String]()`, both take the annotated literal form. No other call form is left in the file.

    ### Discovery worth keeping

    `FileManager.enumerator(at:)` does not fail for an absent directory. Anyone who later reads the new guard and expects a missing model directory to throw will be wrong: that case still gives an empty weight list, and the load fails later at `model.update(parameters:verify:)` instead. The third test records this so the next reader does not have to run the probe again.
  timestamp: 2026-08-10T17:49:55.160790+00:00
- actor: claude-code
  id: 01kzpcqmgp4yzgm3nr8736ee4d
  text: |
    ### Mutation proof, round 3 -- all three mutations still FAIL

    Each was applied alone against the corrected tree, run, and reverted.

    1. `BaseConfiguration.swift` -- `public var mode: QuantizationMode { _mode ?? .affine }` to `{ .affine }`. **FAILS**: 136 issues, 13 tests in 1 suite.
    2. `SwitchLayers.swift` -- `MLX.quantized(other.weight, groupSize:bits:mode: mode)` to `mode: .affine`. **FAILS**: `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` 3 issues on `layer.biases == nil`, then MLX aborts with `[gather_qmm] Scale type must be uint8 but received type float32`.
    3. `BaseConfiguration.swift` -- the whole `else` branch of `quantization(layer:)` to `return nil`. **FAILS**: `loadFilterGivesTheDefaultToAPathThePlanDoesNotName`, 1 issue at `DeepseekV4QuantizationPlanTests.swift:386`.

    After the three reverts, `git status --porcelain -- Libraries/ Tests/` gives exactly two lines, ` M Libraries/MLXLMCommon/Load.swift` and ` M Tests/MLXLMTests/LoadWeightsTests.swift`. `BaseConfiguration.swift` and `SwitchLayers.swift` are clean.

    ### Verification

    - `swift build` -- exit 0. The only warnings are the pre-existing ones in `Qwen2VL.swift`, `Qwen25VL.swift` and `Gemma4.swift`, untouched.
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 13 tests in 1 suite passed.
    - `swift test --filter LoadWeightsTests` -- 4 tests passed, 0 failures.
    - `swift test --filter Quantiz` -- 18 tests in 2 suites passed.
    - `swift test` -- exit 0, zero failures. Swift Testing: 383 + 282 + 80 + 7 tests passed. XCTest: 210 executed, 0 failures, the 2 pre-existing `CompiledDecodeCorrectnessTests` skips only (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ). The XCTest count went from 207 to 210, which is the three new `LoadWeightsTests` cases.
    - `swift format --in-place --configuration .swift-format` on the two touched files only. No tree-wide format run.
  timestamp: 2026-08-10T17:50:08.406818+00:00
- actor: claude-code
  id: 01kzpcqy634cf701jjd3r66xhh
  text: |
    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift (+38/-4: doc comments on `safetensorWeightURLs` and `SafetensorsIndex`, new `package enum WeightLoadingError`, the `enumerator(...)!` force unwrap replaced by `guard ... else { throw }`, both empty-dictionary call forms changed to the annotated literal), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/LoadWeightsTests.swift (+43: three behaviour-preservation tests). All 3 round-2 findings closed and flipped to `- [x]`. No test drives the new throw, because `FileManager.enumerator(at:)` gives a non-nil enumerator for an absent directory, a plain file and a non-file URL alike; a seam for it would put test-only surface on the load path of every model. Mutation proof: all three round-2 mutations still fail (136 issues; the `[gather_qmm] Scale type must be uint8` abort; 1 issue at DeepseekV4QuantizationPlanTests.swift:386), each reverted, tree clean. `swift build` exit 0; `swift test --filter DeepseekV4QuantizationPlanTests` 13/13; `--filter LoadWeightsTests` 4/4; `--filter Quantiz` 18/18; full `swift test` exit 0, zero failures, 2 pre-existing skips.
    - next: /review round 3
  timestamp: 2026-08-10T17:50:18.307237+00:00
- actor: claude-code
  id: 01kzpdvm426wx7ysnzsfnra2ck
  text: |
    ### review — findings

    - scope: `review sha b93de22~1..b93de22` (round 3). Commit `7503469` was left out, because it is a kanban chore.
    - engine: 9 attempted files, 0 skipped, 0 failed. Counts: 1 finding, 1 confirmed, 0 refuted.
    - evidence: `Libraries/MLXLMCommon/Load.swift:49`.

    #### The one finding is new, not a repeat

    It is not substantively the same as any finding of round 1 or round 2. Round 2 named `Load.swift:15` (missing document comment), `:26` (the force unwrap) and `:81` (the empty-collection call form). This one names the index branch at `:49`, which the commit did not touch. No stop-and-escalate is armed.

    Like the three round-2 findings, it sits on code that already existed. The engine reached it because the commit touches this file.

    #### Round-2 findings: all three confirmed closed

    Read the file at HEAD.

    - Document comments. `safetensorWeightURLs` holds a summary, both branches, and `- Parameter`, `- Returns` and `- Throws`. `SafetensorsIndex`, `WeightLoadingError` and its case are documented as well. One `package` declaration holds no `///`: `package var errorDescription` at line 22. This matches the module: `ModelConversionError.errorDescription` and `UserInputError.errorDescription` hold none either.
    - Force unwrap. `grep '!'` over the file gives two hits, lines 95 and 131, and both are the `!=` operator. No force unwrap is left in the file.
    - Empty collections. Lines 115 and 116 both read `= [:]`. No `Type()` call form is left in the file.

    #### Behaviour preservation on the shared load path: the claim holds

    `git checkout b93de22~1 -- Libraries/MLXLMCommon/Load.swift` compiles, and `swift test --filter LoadWeightsTests` gives 4 tests, 0 failures. Restoring the `b93de22` file gives 4 tests, 0 failures again. Nothing outside `Load.swift` names `WeightLoadingError` -- its only three references are its declaration, its document link and its throw -- so the revert breaks no other file.

    One correction to the wording: the suite is 4 tests, not 3, because `testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent` already existed. The tests do pass on both sides, and that is exactly because they reach only paths the edit did not change. Not one of the four reaches the new `guard` `else` branch.

    #### The untested throw: the claim is TRUE

    A probe program on this machine gives a NON-NIL enumerator for all three inputs:

    ```
    (a) absent directory   NON-NIL  items=0
    (b) plain file         NON-NIL  items=0
    (c) https URL          NON-NIL  items=0
    ```

    The `else` branch is unreachable from these inputs, thus no test can drive the throw. The commit message states this openly.

    The guard is still the right call. `FileManager.enumerator(at:includingPropertiesForKeys:)` declares an Optional result, thus the force unwrap was a crash the type system already warned about, and a `guard` that throws costs nothing on the success path.

    New fact the probe found, beyond the claim: an `https:` URL whose path is `/` gives an enumerator that walks the LOCAL filesystem root. `https://example.com/` gave `file:///home`, `file:///usr/` and `file:///usr/bin/`. `FileManager` reads the URL path and ignores the scheme and the host. No caller does this today. It is the same cause as the finding above -- an unvalidated path reaching `FileManager`.

    #### The new error type: the scope is consistent, the named precedent is wrong

    `package` agrees with `safetensorWeightURLs` beside it, which is the only function that throws it, thus the level is internally consistent. `safetensorWeightURLs` already declared `throws` at `b93de22~1`, and the commit added a `throw` inside it without touching the signature. Every call site is `Libraries/MLXLMCommon/Load.swift:117` and four cases in `Tests/MLXLMTests/LoadWeightsTests.swift`; all already hold `try`. No call site newly needs a `try`.

    The commit message names the wrong precedent. It says the new error is "in the same style as `KVCacheError` and `UserInputError`". Read at HEAD:

    - `KVCacheError` (`KVCache.swift:1574`) -- internal, a `struct`, plain `Error`, not `Equatable`.
    - `UserInputError` (`UserInput.swift:454`) -- internal, `LocalizedError`, not `Equatable`.
    - `ModelConversionError` (`ModelConversion.swift:148`) -- public, `LocalizedError`, `Equatable`.

    The shape `LocalizedError, Equatable` matches `ModelConversionError`, not the two named. `grep '^package \(enum\|struct\) .*Error' Libraries/` gives one hit in the whole tree, `WeightLoadingError` itself. The file-local placement is right; every error type in the module sits beside its user and no dedicated errors file exists.

    `ModelConversionError` earns its `Equatable` -- eight test assertions read it, for example `Tests/MLXLMTests/ModelConversionTests.swift:178`. `WeightLoadingError` has no such user, and per the probe it can have none.

    #### Standing mutation re-run

    `BaseConfiguration.swift`, the whole `else` branch of `quantization(layer:)` changed to `return nil`. Clean run: 13 tests, 0 failures. Mutated run: 13 tests, 1 failure -- `loadFilterGivesTheDefaultToAPathThePlanDoesNotName` at `DeepseekV4QuantizationPlanTests.swift:386:26`, `leaves[routerPath]` gave `Linear(inputDimensions=64, outputDimensions=4, bias=false)`. The round-1 gap stays closed. Reverted; `shasum` before and after are both `7a745b5b00e72faec87a85bbbf4e2b220a3a7aa5`, and the clean re-run is 13 tests, 0 failures.

    #### No other model's loading changed

    - `Gemma4KVSharedLoadTests` -- 2 tests, 0 failures.
    - `GLM4LmHeadTiedLoadTests` -- 6 tests, 0 failures.
    - `BaseConfigurationTests` -- 2 tests, 0 failures.
    - `MiniMaxM3Tests` -- 47 tests, 0 failures.

    - next: /implement -- validate the index path components in `safetensorWeightURLs`, then /review round 4.
  timestamp: 2026-08-10T18:09:47.650060+00:00
- actor: claude-code
  id: 01kzpdy6yrdhn4wxsvn5ptbqjw
  text: |
    ### finish iteration 3 — findings, loop stopped by the operator

    - implement: changed — Load.swift doc comments, force unwrap removed, empty-dictionary form; LoadWeightsTests.swift +3 behaviour-preservation tests
    - test: green — `swift build` exit 0; DeepseekV4QuantizationPlanTests 13/13; LoadWeightsTests 4/4; Quantiz 18/18; full `swift test` exit 0
    - commit: b93de22 fix(mlx-lm): remove force-unwrap in safetensorWeightURLs; 7503469 (kanban chore)
    - review: findings — Load.swift:49, a path traversal. 9 files attempted this round.
    - CONFIRMED GOOD: the round-2 fixes are real. The shared load path did not change: `Load.swift` reverted to `b93de22~1` still compiles and `LoadWeightsTests` gives 4/0 on both sides. No caller newly needs `try`. The standing mutation still bites. Gemma4, GLM4, BaseConfiguration and MiniMaxM3 load tests all pass.
    - correction from the reviewer: the commit message names `KVCacheError` and `UserInputError` as the precedent for `WeightLoadingError`. Neither matches. The `LocalizedError, Equatable` shape is `ModelConversionError`'s.

    ### The operator stopped the loop here

    The new finding is real, and the reviewer is right that it is not a repeat. It
    is also outside what this card is about. This card asks whether the mixed
    affine and mxfp4 plan resolves for `deepseek_v4`. That question is answered, and
    the answer held through three rounds of review.

    Rounds 2 and 3 each found a defect that was already in `Load.swift` before this
    work started. `Load.swift` is on the load path of every model in the repository.
    A security change there needs its own card, its own tests, and its own review.

    The path traversal is now task ^gnrcb04. This card does not hold that work.
  timestamp: 2026-08-10T18:11:12.472180+00:00
- actor: claude-code
  id: 01kzq72mnqg33d9zy1vjz12ek3
  text: |-
    ### Review round 3 -- the one finding is closed, and the cause is removed from the whole function

    **Both branches of `safetensorWeightURLs` took an unchecked path into `FileManager`, thus both are closed.**

    1. `weightFileURL(forIndexEntry:in:)`, a new private function, maps one index entry onto the model directory. It rejects an empty entry, an entry that starts with `/`, and an entry that holds a `..` component. The examination reads the text of the entry alone: no file is opened, thus no good entry can change. The index branch calls it in place of the bare `appendingPathComponent`.
    2. `guard modelDirectory.isFileURL` now sits at the top of the function, before both branches. This closes the second face the reviewer found. `FileManager` reads the path of a URL and gives no attention to the scheme or the host, thus `https://example.com/` gave an enumerator that walked the local file system root.

    **Why the components and not a prefix.** A check on the prefix alone lets `shards/../../outside.safetensors` through. The split on `/` finds `..` wherever it sits. `testSafetensorWeightURLsRejectsAnIndexEntryThatClimbsOutFromASubdirectory` is the test that holds this still.

    **Why two new error cases and not the one the finding named.** The finding's example threw `unreadableModelDirectory`, which names a different failure -- the directory cannot be listed. `WeightLoadingError` is thus extended, which is what the card asked for: `modelDirectoryIsNotAFileURL(URL)` and `weightFileOutsideModelDirectory(entry:modelDirectory:)`, each with its own `errorDescription`. A bad entry is reported and never dropped without a word.

    **Why an absolute entry is rejected although it cannot escape.** `appendingPathComponent("/etc/passwd")` nests rather than replaces, thus an absolute entry gives a path inside the model directory today. It is still not a name the index may hold, and the card asks for it to be reported. The rejection is deliberate, not incidental.

    **TDD.** The five new tests were written first. The four rejection tests each failed with "did not throw an error" against the file before the fix, which is the correct red. `testSafetensorWeightURLsKeepsAnIndexEntryInASubdirectory` passed on both sides -- that is a preservation test, and its passing before the fix is the evidence that the good path did not change.

    **Correction carried from the reviewer.** The round-2 commit message named `KVCacheError` and `UserInputError` as the precedent for the shape of `WeightLoadingError`. Neither matches. `ModelConversionError` is the `LocalizedError, Equatable` precedent, and this record does not repeat the wrong claim.

    ### A note on scope

    The comment of 2026-08-10 18:11 states that the operator moved this work to task `^gnrcb04`. This run was dispatched to close the finding on this card, thus the work landed here. `^gnrcb04` now holds the same requirements and needs a decision from the user: close it as done by this change, or keep it for a wider examination of `Load.swift`.
  timestamp: 2026-08-11T01:30:31.991295+00:00
- actor: claude-code
  id: 01kzq7312hpzkr1z4rfmy1ww67
  text: |-
    ### Verification, round 3

    - `swift build` -- exit 0. The only warnings are the pre-existing ones, untouched.
    - `swift test --filter LoadWeightsTests` -- 9 tests, 0 failures (4 pre-existing, 5 new).
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 13 tests in 1 suite passed.
    - `swift test --filter Gemma4KVSharedLoadTests` 2/2, `GLM4LmHeadTiedLoadTests` 6/6, `BaseConfigurationTests` 2/2, `MiniMaxM3Tests` 47/47.
    - `swift test` -- exit 0, zero failures. Swift Testing: 383 + 80 + 282 + 7. XCTest: 215 executed, 0 failures, the 2 pre-existing `CompiledDecodeCorrectnessTests` skips only (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ). The XCTest count went from 210 to 215, which is the five new `LoadWeightsTests` cases.
    - Standing mutation re-run: `BaseConfiguration.quantization(layer:)`, the whole `else` branch changed to `return nil`. **FAILS** -- `loadFilterGivesTheDefaultToAPathThePlanDoesNotName`, 1 issue. The DeepSeek-V4 coverage still bites. Reverted; `shasum` of `BaseConfiguration.swift` is `7a745b5b00e72faec87a85bbbf4e2b220a3a7aa5` before and after, and the clean re-run is 13 tests, 0 failures.
    - `git status --porcelain -- Libraries/ Tests/` gives exactly two lines, ` M Libraries/MLXLMCommon/Load.swift` and ` M Tests/MLXLMTests/LoadWeightsTests.swift`.
    - `swift format --in-place --configuration .swift-format` on the two touched files only. No tree-wide format run.
  timestamp: 2026-08-11T01:30:44.689335+00:00
- actor: claude-code
  id: 01kzq73hv229jkd4v249bxfxkf
  text: |-
    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift (+56/-3: new private `weightFileURL(forIndexEntry:in:)` that rejects an empty, absolute or `..`-holding index entry; `guard modelDirectory.isFileURL` before both branches; two new `WeightLoadingError` cases with their `errorDescription`), /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/LoadWeightsTests.swift (+77: five tests). The one round-3 finding, `Load.swift:49`, is closed and flipped to `- [x]`; 0 open findings on the card. The four rejection tests each failed with "did not throw an error" before the fix; the subdirectory preservation test passed on both sides. `swift build` exit 0; `swift test --filter LoadWeightsTests` 9/9; `--filter DeepseekV4QuantizationPlanTests` 13/13; Gemma4 2/2, GLM4 6/6, BaseConfiguration 2/2, MiniMaxM3 47/47; full `swift test` exit 0, zero failures, 2 pre-existing skips. Standing mutation (`quantization(layer:)` `else` -> `return nil`) still FAILS 1 test, reverted, shasum unchanged, tree holds only the two intended files.
    - next: /review round 4
  timestamp: 2026-08-11T01:31:01.858048+00:00
- actor: claude-code
  id: 01kzq76eggpd384v7g4y8nn38n
  text: |-
    ### commit — changed
    - evidence: 638dde5 fix(mlx-lm): reject unsafe safetensors index paths and non-file URLs; 5cfd48e chore(kanban): record round-3 fix and close ^wkv5j6f review findings
    - next: none, all 10 review findings across three rounds are closed
  timestamp: 2026-08-11T01:32:36.752585+00:00
- actor: claude-code
  id: 01kzq7z4php57rxntb4m56m1mc
  text: |
    ### review — findings

    - scope: `review sha 638dde5~1..638dde5` (round 4). 9 files attempted, 0 failed, 0 skipped. 1 finding, 1 confirmed, 0 refuted.
    - evidence: `Libraries/MLXLMCommon/Load.swift:29`

    #### Round-3 path-traversal fix: examined and sound

    The reviewer tried to get past `weightFileURL(forIndexEntry:in:)` with 28 entry strings. Every one was rejected, or it stayed under the model directory. No bypass survived.

    - Rejected: `../outside.safetensors`, `shards/../../outside.safetensors`, `..`, `./../x`, `shards/./../../x`, `a/b/../../../x`, `x.safetensors/../../../etc/passwd`, `/etc/passwd`, the empty entry.
    - Allowed and contained: `%2e%2e/x`, `%2E%2E%2Fx`, `..%2Fx`, `%252e%252e/x`. `URL.appendingPathComponent` percent-**encodes**; it does not decode. A percent-encoded entry is thus the literal name of a directory, and it names no parent.
    - Allowed and contained: `..\..\x`, `C:\Windows\x`, `\\server\share\x`. On Darwin the backslash is an ordinary filename character, not a separator.
    - Allowed and contained: `․․/x` (one dot leader), `．．/x` (fullwidth full stop), `.̣./x` (combining mark), `..\0/x`, `....//x`, `..;/x`, ` ../x`. Each is a different filename from `..`. Unicode normalization does not turn any of them into `..`.
    - Allowed and contained: `https://example.com/x`, `file:///etc/passwd`, `~/x`, `~root/x`. `appendingPathComponent` does no tilde expansion and reads no scheme.
    - A symlink inside the model directory that points outside is **not** closed by this check, and closing it would be wrong. The check reads the text of the entry alone. `Libraries/` and the `swift-transformers` Hub hold no `createSymbolicLink` call, thus the load path never makes one, and a check that resolved symlinks would break any download layout that does.

    `isFileURL` rejects no legitimate caller input. Every `URL(fileURLWithPath:)` form gives true, absolute or relative, with or without `isDirectory:`; so do `URL(string: "file:///...")`, `homeDirectoryForCurrentUser` and `temporaryDirectory`. The two callers pass either a downloader result or a `.directory(URL)` from `ModelConfiguration` (`Libraries/MLXLMCommon/ModelFactory.swift`).

    #### The success path is unchanged — verified by mutation, not by assertion

    A straight revert of `Load.swift` to `638dde5~1` **does not compile**: `LoadWeightsTests` names `WeightLoadingError.weightFileOutsideModelDirectory` and `.modelDirectoryIsNotAFileURL`, which the parent does not hold. The round-3 record's wording ("failed with 'did not throw an error' against the code before this fix") is thus not reproducible by a straight revert. It is reproducible when the two error cases stay and only the two guards are neutered, and the reviewer ran that:

    - Guards disabled (never reject): 9 tests, 4 failures — exactly the four rejection tests, each "did not throw an error". The five success tests stayed green. The good path is thus unchanged.
    - Guard inverted (always reject): 9 tests, 3 failures — `testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent` and `testSafetensorWeightURLsKeepsAnIndexEntryInASubdirectory` went red. The success tests do reach the new function; they are not vacuous.

    #### Standing mutation re-run

    `BaseConfiguration.quantization(layer:)` `else` branch → `return nil`: `DeepseekV4QuantizationPlanTests` goes red at `loadFilterGivesTheDefaultToAPathThePlanDoesNotName()`. The round-1 coverage holds. Reverted; `git diff HEAD -- Libraries/ Tests/ IntegrationTesting/` is empty.

    #### Suites, on the restored tree

    `LoadWeightsTests` 9/9. `DeepseekV4QuantizationPlanTests` 13/13. `Gemma4KVSharedLoadTests` 2/2. `GLM4LmHeadTiedLoadTests` 6/6. `BaseConfigurationTests` 2/2. `MiniMaxM3Tests` green. No other model's loading changed.

    #### All 10 prior findings

    Checked one by one, not taken on trust. Round 1: the mutation is now caught; the probe compressor sits on layer 2 and `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` reads `model.layers.2.attn.compressor.wkv`; the 82/84 counts sit in the test header with two tests that measure them; and the three downstream cards `^tty95f4`, `^r92pjcr` and `^ag7ant0` each hold the comment with the layers 0/1 rule and the 44-against-43 fact. Round 2: the doc comment, the `guard let enumerator`, and both `[:]` literals are in the file. Round 3: verified above. All 10 are genuinely closed.

    - next: add the `///` comment on `errorDescription` in `Libraries/MLXLMCommon/Load.swift`, then re-review.
  timestamp: 2026-08-11T01:46:05.905613+00:00
- actor: claude-code
  id: 01kzq8157ptjm463g03h1dnq06
  text: |
    ### finish iteration 4 — findings
    - implement: changed — Load.swift path validation (`weightFileURL(forIndexEntry:in:)`, `isFileURL` guard, two new error cases); LoadWeightsTests.swift +5 tests
    - test: green — `swift build` exit 0; LoadWeightsTests 9/9; DeepseekV4QuantizationPlanTests 13/13; Gemma4 2/2; GLM4 6/6; BaseConfiguration 2/2; MiniMaxM3 47/47; full `swift test` exit 0
    - commit: 638dde5 fix(mlx-lm): reject unsafe safetensors index paths and non-file URLs; 5cfd48e and cc00c0f (kanban chores)
    - review: findings — Load.swift:29 (1 finding). 9 files attempted.
    - SECURITY VERIFIED: the reviewer tried 28 attack strings and none went out of the model directory. Percent-encoding does not work, because `appendingPathComponent` encodes and never decodes. Backslash, UNC and `C:\` do not work on Darwin. Unicode near-copies of `..` do not work, because each is a different file name. `~` does not expand. The symlink case is correctly left open: the check reads the text of the entry alone, no code on the load path makes a symlink, and a check that followed symlinks would break a download layout that uses them.
    - CORRECTION to the round-3 evidence: a straight revert of `Load.swift` to the parent commit does not compile, because the test file names the two new error cases. The claim "the four rejection tests failed against the code before the fix" thus cannot come from a revert. The reviewer produced it another way, by keeping the error cases and turning off the guards alone: 4 failures, and the 5 success tests stayed green. An inverted guard turns 3 tests red, thus the success tests do reach the new function and are not vacuous.
    - next: implement — add the missing doc comment at Load.swift:29
  timestamp: 2026-08-11T01:47:11.990187+00:00
- actor: claude-code
  id: 01kzq8chbh3w5m239j4yz74226
  text: |-
    ### Review round 4 -- the one finding is closed, documentation only

    **The change touches no logic.** `git diff -- Libraries/` is 8 lines and every one of them starts with `///`. The round-3 path validation, which the reviewer examined with 28 attack strings, is untouched.

    **The cause was removed for the whole file, not for the one line.** The finding names `errorDescription`. Three more declarations in `Load.swift` held no doc comment either, and each now holds one:

    - `SafetensorsIndex.weightMap`
    - the `CodingKeys` enum of that struct
    - the `weightMap` case of that enum

    Every declaration in `Load.swift` -- private, package and public alike -- is now documented.

    **The comment on `errorDescription`** follows the style of the rest of the file: a one-sentence summary that names the `LocalizedError` conformance, a blank `///` line, then a paragraph that states what each case gives.

    **No test, and this is stated openly.** The change adds no behaviour, thus no test can go red first. The suites that read this file all stay green, which is the evidence that nothing moved.

    **One note for the reader.** `CodingKeys` enums are not documented anywhere else in `MLXLMCommon` -- six other files hold an undocumented one. The comment here is thus stricter than the prevailing pattern of the module. It sits inside the one file the finding names, and no other file was touched.

    ### Verification

    - `swift build` -- exit 0.
    - `swift test --filter LoadWeightsTests` -- 9 tests, 0 failures.
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 13 tests in 1 suite passed.
    - `swift test` -- exit 0. Swift Testing: 383 + 80 + 282 + 7 tests passed. XCTest: 215 executed, 0 failures, and only the 2 pre-existing `CompiledDecodeCorrectnessTests` skips (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ).
    - `git status --porcelain -- Libraries/ Tests/ IntegrationTesting/` gives one line, ` M Libraries/MLXLMCommon/Load.swift`.
    - `swift format --in-place --configuration .swift-format` on that one file only. It made no change. No tree-wide format run.
  timestamp: 2026-08-11T01:53:24.849424+00:00
- actor: claude-code
  id: 01kzq8cyjzt7x1s1sz6vne5g9n
  text: |-
    ### implement — changed
    - evidence: 1 file — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLMCommon/Load.swift (+8, every line a `///` comment: the doc comment on `errorDescription` that the finding asks for, plus `SafetensorsIndex.weightMap`, its `CodingKeys` enum and that enum's `weightMap` case, which held the same cause). The round-4 finding `Load.swift:29` is closed and flipped to `- [x]`; 0 open findings on the card. No logic changed, thus the round-3 path validation stays as the reviewer examined it. `swift build` exit 0; `swift test --filter LoadWeightsTests` 9/9; `--filter DeepseekV4QuantizationPlanTests` 13/13; full `swift test` exit 0, zero failures, 2 pre-existing skips. `git status --porcelain -- Libraries/ Tests/ IntegrationTesting/` gives one line, ` M Libraries/MLXLMCommon/Load.swift`.
    - next: /review round 5
  timestamp: 2026-08-11T01:53:38.399120+00:00
- actor: claude-code
  id: 01kzq8e99ws7f66j6tf04myk7r
  text: |-
    ### commit — changed
    - evidence: bc561fc docs(mlx-lm): document WeightLoadingError.errorDescription and SafetensorsIndex; 74e50d0 chore(kanban): record round-4 doc fix on task wkv5j6f
    - next: none. Local commits only, not pushed.
  timestamp: 2026-08-11T01:54:22.140306+00:00
- actor: claude-code
  id: 01kzq8zp3pm40aw2fzz4w1qh7k
  text: |-
    ### review — clean

    - scope: `review sha bc561fc~1..bc561fc` (round 5). Commit `74e50d0` is a kanban chore and was not read.
    - evidence: zero new findings in scope. `git show bc561fc --numstat` gives `8 0 Libraries/MLXLMCommon/Load.swift`. A grep of the added lines for a line that is not a `///` comment gives nothing, and a grep for a removed line gives nothing, thus the commit changes no executable line.
    - the four new comments read true against the code they sit on. `errorDescription` names the `LocalizedError` conformance and states that each case gives one sentence that names the directory or the index entry; the three case bodies do exactly that. `SafetensorsIndex.weightMap` reads "the name of the weight file that holds each checkpoint key", which matches the `[String: String]` shape and the header comment of the struct. The `CodingKeys` comment and the `weightMap` case comment name the `weight_map` object, which is the raw value.
    - all 11 findings of rounds 1 to 4 stay `- [x]`, each with its resolution note. 6 acceptance items and 6 test items stay `- [x]`.
    - tests: `swift test` exit code 0. Zero failures, zero errors, zero warnings.
    - the implementer's question on `CodingKeys`: not a finding here. A documented declaration is never a defect, and a file that holds more documentation than its neighbours does no harm. Documentation for the `CodingKeys` enums of the other files of `MLXLMCommon` belongs to a task of its own, and the comments this commit added stay.
    - out of this commit's scope, for a separate task, and **not an open finding on this card**: the engine reports that `loadWeights` throws and its document comment holds no `- Throws:` entry (`Libraries/MLXLMCommon/Load.swift`). That comment is upstream text, last changed by `f5f18ed` and `826fbc9`. No commit of this card touched it, and `bc561fc` added no line near it.
    - next: none. The card is done.
  timestamp: 2026-08-11T02:03:52.310308+00:00
- actor: claude-code
  id: 01kzq90nw383c7730v12tzmbg1
  text: |
    ### finish iteration 5 — clean
    - implement: changed — Load.swift, 8 lines, each one a `///` comment. No executable line changed.
    - test: green — `swift build` exit 0; LoadWeightsTests 9/9; DeepseekV4QuantizationPlanTests 13/13; full `swift test` exit 0, zero failures, 2 pre-existing skips
    - commit: bc561fc docs(mlx-lm): document WeightLoadingError.errorDescription and SafetensorsIndex; 74e50d0 (kanban chore)
    - review: clean — zero new findings. All 11 findings of rounds 1 to 4 stay closed. Task moved to `done`.
    - verification: `git show bc561fc --numstat` gives `8 0`, and a search of the added lines finds no line that is not a `///` comment. The `errorDescription` comment agrees with what each of the three cases truly gives.
    - note: the engine reported that `loadWeights` has no `- Throws:` entry in its doc comment. That comment is upstream text that no commit of this card touched, thus the reviewer recorded it for a separate task and did not open it here.
    - note: `CodingKeys` enums have no comment in six other files of `MLXLMCommon`. This file is now stricter than the module. The reviewer judged that this is not a defect.
  timestamp: 2026-08-11T02:04:24.835024+00:00
depends_on:
- 01KZGMPECN4FA7T3BFX6F6QMF7
position_column: done
position_ordinal: df80
title: Verify mixed affine+mxfp4 per-layer quantization resolves for deepseek_v4
---
## What

DeepSeek-V4-Flash-4bit is quantized as **4-bit affine, group size 64, plus `mxfp4` on some FFN layers**. Establish — with tests — that our existing quantization plumbing already resolves that mixed plan, and fix whatever gaps the tests expose.

**Scope note (revised after research):** an earlier read suggested `mxfp4` was a missing loader feature. It is not. Verified facts:

- `QuantizationMode` in mlx-swift already has `.mxfp4` (and `.mxfp8`, `.nvfp4`) — `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:1097`.
- `BaseConfiguration.Quantization` already decodes `mode` and exposes `asTuple = (groupSize, bits, mode)` — `Libraries/MLXLMCommon/BaseConfiguration.swift:22-54`.
- `PerLayerQuantization.quantization(layer:)` already resolves per-layer overrides and `.skip` — `Libraries/MLXLMCommon/BaseConfiguration.swift:71-118`.
- `Load.swift:64-69` already threads the resolved per-layer tuple (including `mode`) into `quantize(model:)`.
- `QuantizedSwitchLinear` already forwards `mode:` to `MLX.gatherQuantizedMM`, and its `biases` is already `MLXArray?` — correct for mxfp4, which has scales but no biases (`Libraries/MLXLMCommon/SwitchLayers.swift:379-445`).
- Our `mlx-swift` resolves to exactly `0.31.6` (`Package.resolved`), matching what `scouzi1966/maclocal-api` pins. No `Package.swift` change needed.

So the generic path should already work. What this task must prove:

1. The real `config.json` mixed plan decodes into a `PerLayerQuantization` whose mxfp4-designated FFN layers resolve to `mode == .mxfp4` while everything else resolves to `(64, 4, .affine)`.
2. A `QuantizedSwitchLinear` built with `mode: .mxfp4` produces finite, correctly-shaped output through `gatherQuantizedMM` with `biases == nil`.
3. Any layer-path mismatch between DSV4's checkpoint key naming and our `quantize(model:)` path strings is identified and handled.

**Explicitly out of scope:** the reference's custom fused Metal kernels. `scouzi1966/mlx-swift-lm`'s `Libraries/MLXLMCommon/SwitchLayers.swift` is 1025 lines vs our 444 (1105 changed diff lines), and that delta is mostly two hand-written Metal kernels — `deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and `deepseek_v4_native_mxfp4_down_sum6` — gated behind env knobs `VMLX_DSV4_NATIVE_MXFP4`, `VMLX_DSV4_MXFP4_ROWS_PER_SIMD`, `VMLX_DSV4_MXFP4_SIMD_GROUPS`. Those are a **throughput optimization, not a correctness requirement**. The slow-but-correct generic `gatherQuantizedMM` path is the target here; file a separate performance task if profiling later justifies the kernels. Do **not** wholesale-copy their `SwitchLayers.swift` — ours has diverged and the copy would regress other models.

## Provenance
- Reference for the kernel approach we are deliberately *not* porting: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLMCommon/SwitchLayers.swift` (MIT).
- Load-side gap list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md` (names FP4 e2m1fn routed-expert dequant and FP8 e4m3fn + UE8M0 128x128 dequant).

## Acceptance Criteria

- [x] A test proves the real DSV4-Flash-4bit `config.json` quantization block decodes to a `PerLayerQuantization` with at least one `.mxfp4` layer and a `(64, 4, .affine)` default.
- [x] A test proves `QuantizedSwitchLinear(mode: .mxfp4)` returns finite output of the expected shape with `biases == nil`.
- [x] If any gap is found, it is fixed in the smallest possible diff and the fix is covered by a test. (No gap was found: seven of the eight tests passed against unchanged production code. The one red test named a layer the fixture does not hold -- see the comments. Review round 1 added one production refactor: the load-time filter moved out of `loadWeights` into `quantizationParameters(forPath:weights:quantization:perLayerQuantization:)` so the tests run the real gate.)
- [x] `Libraries/MLXLMCommon/SwitchLayers.swift` is either unmodified or changed by fewer than ~40 lines; no Metal kernel source added. (Unmodified.)
- [x] Existing SwitchLayers/quantization tests still pass (no regression to other models).
- [x] A note is recorded (in the task or a follow-up) stating that fused mxfp4 kernels were deliberately deferred.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift`.
- [x] Test: decode the checked-in DSV4 `config.json` fixture; assert the resolved per-layer modes as described.
- [x] Test: construct a small `SwitchLinear` (e.g. 4 experts, 32x64), call `toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)`, run it, assert output shape and `allFinite`.
- [x] Test: same construction with `.affine` still works — guards against a regression in the shared path.
- [x] Run: `swift test --filter DeepseekV4QuantizationPlanTests` — all pass.
- [x] Run the pre-existing quantization suite to prove no regression: `swift test --filter Quantiz` — all pass.

## Workflow
- Use `/tdd` — write the plan-resolution and mxfp4 forward tests first; they may pass immediately, which is itself the finding. Only write code for gaps the tests actually expose.
#deepseek-v4

## Review Findings (2026-08-10 12:05)

- [x] `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift:45` — `quantizationTuple(for:plan:)` puts plan membership in place of the production gate, thus no test pins the filter that `loadWeights` runs. Production gates on `weights["\(path).scales"] != nil` (`Libraries/MLXLMCommon/Load.swift:38`) and then calls `perLayerQuantization.quantization(layer: path)`, which gives the `(64, 4, affine)` default to every path the plan does not name (`Libraries/MLXLMCommon/BaseConfiguration.swift:97`). The helper gives `nil` to those paths instead. Proof by mutation: the whole `else` branch of `quantization(layer:)` changed to `return nil` keeps all 8 tests green. Remove this cause for the whole file — drive the probe through the same two-part gate the production code uses, with a stand-in weights dictionary that holds `<path>.scales`, and not through a plan-membership substitute.
  - Resolved. The gate moved out of `loadWeights` into the package function `quantizationParameters(forPath:weights:quantization:perLayerQuantization:)` in `Load.swift`, and `loadWeights` now calls it. The helper `quantizationTuple(for:plan:)` is deleted. `leavesAfterLoadFilter(plan:scalePaths:)` runs that same production function over the probe tree with a stand-in weights dictionary from `stubScales(for:)`. Three tests now read the gate: `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse`, `loadFilterLeavesAPathWhoseWeightsHoldNoScalesAlone` and `loadFilterGivesTheDefaultToAPathThePlanDoesNotName`, plus `loadFilterQuantizesNothingWhenNoWeightsHoldScales` for the first part of the gate. The reviewer's third mutation now fails.
- [x] `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift:261` — `#expect(!(router is QuantizedLinear), "the router stays in high precision")` states a result about production that this test cannot reach. In `loadWeights` an unnamed path still takes the `(64, 4, affine)` default when the checkpoint holds `<path>.scales`; only the absence of that array keeps the router in high precision. The document comment at lines 37-44 states that the checkpoint holds those arrays for exactly the layers the block names, and the repository holds no evidence for that statement. Same cause as the finding above.
  - Resolved. The document comment on `stubScales(for:)` now states the assumption in as many words: the repository holds no DeepSeek-V4 weight file, thus which paths the published checkpoint truly holds a `scales` array for is an assumption, and the tests state their results as "given these scales, the filter gives this". The router assertion now reads "no scales, thus no quantization", and the companion test shows the other half: a router with a `scales` array does take the plan's default.
- [x] `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift:364` — `ProbeAttention` gives layer 0 a `compressor` submodule, thus the probe tree holds `model.layers.0.attn.compressor.wgate` and `model.layers.0.attn.compressor.wkv`. The fixture names a compressor for layers 2 to 42 only, because `compress_ratios[0]` is 0. This does not agree with the file's own comment at lines 193-195, nor with the document comment at lines 418-419, which states that the probe paths equal the checkpoint key paths the block names. No assertion reads the probe compressor. Delete `ProbeCompressor` from the probe tree, or move it to a layer index the fixture names, and correct the document comment.
  - Resolved by the second option. `ProbeAttention` takes `hasCompressor:` and holds the submodule as `ProbeCompressor?`, the probe tree holds three layers, and only layer 2 holds a compressor. `loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse` now reads `model.layers.2.attn.compressor.wkv` and asserts it quantizes affine at group size 64.
- [x] `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` — the key counts written in the result comment on this card are wrong. The block holds 82 compressor keys across the 41 layers 2 to 42 (`wgate` and `wkv` on each), not 41 keys. It holds 84 indexer keys across the 21 even layers 2 to 42 (`wq_b`, `weights_proj`, `compressor.wgate`, `compressor.wkv` on each), not 21 keys. Both numbers given are layer counts with a key-count label. Correct the comment, because the downstream tasks read it.
  - Resolved. The result comment now reads 82 compressor keys across 41 layers and 84 indexer keys across 21 layers, and it names the four indexer key suffixes. The same counts sit in the test file header, and two new tests measure them: `fixtureNamesCompressorKeysFromLayerTwoUp` and `fixtureNamesIndexerKeysOnTheEvenLayersFromTwoUp`.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:297` — the layers 0 and 1 result is not written where the downstream tasks read it. Task `^r92pjcr` (Indexer) names `index_n_heads`, `index_head_dim` and `index_topk` only, and states nothing about which layers hold indexer weights. The checkpoint holds them for the 21 even layers 2 to 42, which are the `compress_ratios == 4` layers. `hasCompressor(layer:)` exists, and no `hasIndexer(layer:)` exists beside it. An indexer built on all 43 layers, or on all 41 compressor layers, fails the weight load. Task `^tty95f4` (Compressor) states the general rule but names neither layer 0 nor layer 1, and does not state that the submodule itself must be absent. Task `^ag7ant0` holds the rule for the RoPE theta only. The result comment on this card sends the fact to "the MoE and attention tasks", and the compressor and the indexer belong to `^tty95f4` and `^r92pjcr`, which that comment does not name. Write the fact on `^tty95f4`, `^r92pjcr` and `^ag7ant0`.
  - Resolved. A comment now sits on each of `^tty95f4`, `^r92pjcr` and `^ag7ant0`. The `^tty95f4` comment states that the submodule must be absent on layers 0 and 1, and names the optional-submodule pattern. The `^r92pjcr` comment states the even-layer rule and asks for a `hasIndexer(layer:)` beside `hasCompressor(layer:)`. `DeepseekV4Configuration.swift` is unchanged.
- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:298` — `compress_ratios` holds 44 entries, `num_hidden_layers` is 43, and entry 43 is 0. `hasCompressor(layer:)` guards the bound, thus it is correct today. Any downstream code that reads `compressRatios.count` as the layer count gets 44. Write this fact beside the layers 0 and 1 fact on the same downstream tasks.
  - Resolved. Each of the three comments states the 44-against-43 fact and says to read `numHiddenLayers` for a layer count.

## Review Findings (2026-08-10 12:26)

- [x] `Libraries/MLXLMCommon/Load.swift:15` — Public package function `safetensorWeightURLs` lacks documentation. It has no doc comment explaining what it does, its parameters, or its return value. This makes the API harder to understand and use. Add a documentation comment above the function explaining its purpose, parameters, and return value. For example:

```swift
/// Collects the URLs of all safetensor weight files in the model directory.
///
/// This function checks for a safetensors index file and uses it if present;
/// otherwise, it enumerates all .safetensors files in the directory.
///
/// - Parameter modelDirectory: The directory containing model weights.
/// - Returns: An array of URLs to safetensor weight files.
package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
```.
  - Resolved. `safetensorWeightURLs` now holds a `///` comment in the style the rest of the file uses: a one-sentence summary, a paragraph that names both branches (index file, or every `.safetensors` file in the directory and its subdirectories), and `- Parameter`, `- Returns` and `- Throws` entries. The same cause was removed for the whole file: the private `SafetensorsIndex` struct, which held no comment either, now holds one, and the new `WeightLoadingError` and its case are documented. Every other `package`/`public` declaration in the file (`quantizationParameters`, `loadWeights`) was already documented.
- [x] `Libraries/MLXLMCommon/Load.swift:26` — Force unwrap on non-guaranteed result: FileManager.default.enumerator() can return nil, so the force unwrap `!` will crash if it does. Use optional binding or provide a fallback instead. Replace the force unwrap with proper error handling: `guard let enumerator = FileManager.default.enumerator(at: modelDirectory, includingPropertiesForKeys: nil) else { throw /* appropriate error */ }` or return an empty array.
  - Resolved with the throwing form, not the empty-array fallback, because an empty array would swallow the failure. The file reports every other failure by `throws` (`Data(contentsOf:)`, `JSONDecoder.decode`, `loadArraysAndMetadata`, `model.update`), and it held no error type of its own, thus a `package enum WeightLoadingError: LocalizedError, Equatable` with one case, `unreadableModelDirectory(URL)`, now sits beside the function. This matches the file-local error pattern that `KVCacheError`, `UserInputError` and `ModelConversionError` already use in this module. The success path is unchanged term for term. Three new tests in `Tests/MLXLMTests/LoadWeightsTests.swift` hold the behaviour still: the recursive `.safetensors` discovery, a directory with no weight file, and an absent directory. All three passed against the file before the edit and after it, thus the change is behaviour preserving. The `nil` branch itself cannot be reached from a real call: a probe program shows that `FileManager.enumerator(at:includingPropertiesForKeys:)` gives a non-nil enumerator for an absent directory, a plain file and a non-file URL alike. A test for the throw would need a `FileManager` seam in the production signature, which would put test-only surface on the load path of every model, thus none was added. The guard keeps the `nil` case honest for the platforms and sandboxes where Foundation does give `nil`.
- [x] `Libraries/MLXLMCommon/Load.swift:81` — Empty dictionary initialization uses the call form `()` instead of the idiomatic literal with type annotation. This is inconsistent with Swift conventions for empty collections. Change to `var weights: [String: MLXArray] = [:]` — the annotated literal is the idiomatic Swift form for empty collections.
  - Resolved, and the same cause was removed for the whole file. `var weights = [String: MLXArray]()` is now `var weights: [String: MLXArray] = [:]`, and the line below it, `var metadata = [String: String]()`, is now `var metadata: [String: String] = [:]`. No other empty-collection call form is left in the file.

## Review Findings (2026-08-10 12:52)

- [x] `Libraries/MLXLMCommon/Load.swift:49` — Path traversal vulnerability—file paths are constructed from model index entries without validation, allowing `..` sequences to access files outside the model directory. Validate path components before appending them. Reject any containing `..`, starting with `/`, or otherwise escaping the model directory. Example: `guard !$0.contains("..") && !$0.hasPrefix("/") else { throw WeightLoadingError.unreadableModelDirectory(modelDirectory) }`.
  - Resolved, and the cause was removed for the whole function rather than for the one line. Both branches gave an unchecked path to `FileManager`, thus both are now closed. (1) A new private function, `weightFileURL(forIndexEntry:in:)`, maps one index entry onto the model directory. It rejects an empty entry, an entry that starts with `/`, and an entry that holds a `..` component. The examination is of the text of the entry alone, thus it reads no file and it changes no good entry: a plain name and a name in a subdirectory both keep the URL they had. (2) `safetensorWeightURLs` now starts with `guard modelDirectory.isFileURL`, which closes the second face the reviewer found — `FileManager` reads only the path of a URL and gives no attention to the scheme or the host, thus `https://example.com/` gave an enumerator that walked the local file system root. This one guard sits before both branches. A bad entry is reported and not dropped without a word: `WeightLoadingError` is extended with `modelDirectoryIsNotAFileURL(URL)` and `weightFileOutsideModelDirectory(entry:modelDirectory:)`, each with its own `errorDescription`. The reviewer's example threw `unreadableModelDirectory`, which names a different failure, thus a case of its own is more exact. Five new tests in `Tests/MLXLMTests/LoadWeightsTests.swift` cover a `../` entry, a `..` entry from a subdirectory (a check on the prefix alone would let this one through), an absolute-path entry, a `https:` URL, and — for the success path — an entry in a subdirectory. The four rejection tests each failed with "did not throw an error" before the fix and pass after it; the subdirectory test passes on both sides, which is the evidence that the good path did not change. `LoadWeightsTests` is 9 of 9, `DeepseekV4QuantizationPlanTests` 13 of 13, and `Gemma4KVSharedLoadTests` (2), `GLM4LmHeadTiedLoadTests` (6), `BaseConfigurationTests` (2) and `MiniMaxM3Tests` (47) all stay green.
  - One correction to the round-2 record, which the reviewer made: the commit message named `KVCacheError` and `UserInputError` as the precedent for the shape of `WeightLoadingError`. Neither matches. `ModelConversionError` (`Libraries/MLXLMCommon/ModelConversion.swift`) is the `LocalizedError, Equatable` precedent.

## Review Findings (2026-08-10 20:33)

- [x] `Libraries/MLXLMCommon/Load.swift:29` — Public computed property `errorDescription` lacks documentation comment, inconsistent with documented enum cases above it. Add a documentation comment explaining that this property provides localized error descriptions for the `LocalizedError` protocol conformance.
  - Resolved, and the cause was removed for the whole file. `errorDescription` now holds a `///` comment in the style the rest of the file uses: a one-sentence summary that names the `LocalizedError` conformance, then a paragraph that states what each case gives. The same cause elsewhere in the file: `SafetensorsIndex.weightMap`, its `CodingKeys` enum and the `weightMap` case of that enum held no comment either, and each now holds one. Every declaration in `Load.swift` is thus documented. No logic changed — `git diff -- Libraries/` is 8 lines, all of them `///` comments — thus the round-3 path validation stays as the reviewer examined it.
