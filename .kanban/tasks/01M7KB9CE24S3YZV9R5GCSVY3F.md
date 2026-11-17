---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7n0j328ns2wqcnnzp6zfcd0
  text: '2026-11-17 implement: Moved the 4 env-gated LFM2 XCTests (LFM2Tests x2, LFM2MoeRoutingTests x2) plus the 2 env-gated DSpark Swift Testing tests in LFM2DSparkTests (same cause: MLX_LFM2_MODEL_PATH / MLX_LFM2_DSPARK_* gates, silent early return) out of MLXLMTests into IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift. Each now downloads a pinned checkpoint via #hubDownloader + resolve(): LiquidAI/LFM2.5-2.6B-MLX@b41f2b65 (bf16 subdir), LiquidAI/LFM2.5-8B-A1B-MLX-4bit@d16665ca (the revision with the 8-bit layer-2 router override; 760bf10c+ stores the gate in fp32), LiquidAI/LFM2.5-2.6B-DSpark@458cedab. The upstream assertion `generation_config temperature == 0.1` does not match the published checkpoint (0.2 at every revision); the moved test asserts 0.2 at the pin and that the factory-loaded generationConfig equals the file. Removed the now-unused PseudoWordTokenizer stubs and MLXScriptedLM imports. Added CLAUDE.md "After an upstream merge" step 4 on env-gated tests. Found but out of scope: Tests/MLXLMTests/VerifyAttentionBlockPassProfileTests.swift is gated by .enabled(if: MLX_VERIFY_ATTENTION_PROFILE) (fork code, attention kernels).'
  timestamp: 2026-11-17T15:03:43.944382+00:00
- actor: claude-code
  id: 01m7n1y4scx2sh71ce215cnj6d
  text: '2026-11-17 test: The first MLXLMTests run still reported `Suite VerifyAttentionBlockPassProfileTests skipped` (Swift Testing, `.enabled(if: MLX_VERIFY_ATTENTION_PROFILE == "1")`). The cause is the same as for the LFM2 tests, so I moved it to IntegrationTesting/IntegrationTestingTests/VerifyAttentionBlockPassProfileTests.swift without the gate and without the MLX_VERIFY_ATTENTION_PROFILE_KEYS override (fixed key counts 2k/32k/100k), with local Qwen3.8 shape constants in place of the MLXLMTests-only AttentionKernelCase. I extended the CLAUDE.md note to cover it. After swift-format: swift build --build-tests has 0 warnings from project sources. xctest: MLXLMTests 1130 XCTest + 2379 Swift Testing, 0 failures, 0 skipped (before: 1134 with 4 skipped, plus 1 skipped suite); MLXGuidedGenerationTests 78; MLXFoundationModelsTests 519; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11; all 0 failures. Known issues: 3, all existing withKnownIssue tests (AttentionKernelToleranceTests, MiniCPM5ToolCallParserTests). IntegrationTesting xcodebuild build-for-testing: TEST BUILD SUCCEEDED. The LFM2 checkpoints are not in the local HF cache, so the moved LFM2 tests were not run.'
  timestamp: 2026-11-17T15:27:47.500056+00:00
- actor: claude-code
  id: 01m7n234q3c0rrwwz1fgdv7rqx
  text: |-
    ## Review Findings (2026-11-17 09:28) — round 1, review sha f5e9306d..d54a523d

    The validators read file contents from the main checkout, so they could not open the two new IntegrationTesting files, and their line numbers point at the moved code as it stands in the main tree. The cited constructs moved into IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift, so the fixes go there, across that whole file and VerifyAttentionBlockPassProfileTests.swift.

    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:302` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:341` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[Int]()' with ': [Int] = []'.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:343` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[Int]()' with ': [Int] = []'.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:346` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:449` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[Sample]()' with ': [Sample] = []'.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:450` `code-hygiene/idioms-swift` — AlwaysUseLiteralForEmptyCollectionInit: replace '[Sample]()' with ': [Sample] = []'.
    - [ ] `Tests/MLXLMTests/LFM2DSparkTests.swift:469` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
    - [ ] `Tests/MLXLMTests/LFM2MoeRoutingTests.swift:364` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
    - [ ] `Tests/MLXLMTests/LFM2Tests.swift:264` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
  timestamp: 2026-11-17T15:30:31.267797+00:00
- actor: claude-code
  id: 01m7n25jy2deaaq379fphwb0cb
  text: |-
    2026-11-17 fix round 1: every round-1 finding is now resolved.
    - [x] magic-numbers-swift: In LFM2CheckpointIntegrationTests.swift each literal is a named constant (pinned-checkpoint facts: temperature, vocab, layer count, router/module bits, group size, short prompt; plus a BenchmarkShape enum for the benchmark sizes, filler tokens and bytes per GiB). The same cause was fixed in VerifyAttentionBlockPassProfileTests.swift (tensorsPerLayer, querySeed, nanosecondsPerMillisecond, bytesPerMillisecondPerGBPerSecond).
    - [x] idioms-swift AlwaysUseLiteralForEmptyCollectionInit: `[Int]()` and `[Sample]()` are now `: [T] = []`. No such constructs remain in either file.
    - [x] disallowed-constructs-swift no_direct_standard_out_logs: each of the 4 benchmark `print` lines has `// swiftlint:disable:next no_direct_standard_out_logs — a benchmark; its output is the measurement.`, the same form the profile file uses.
    Evidence: IntegrationTesting build-for-testing reports TEST BUILD SUCCEEDED with no warnings in either file, and VerifyAttentionBlockPassProfileTests passes (12.4 s, run under bench.lock).
  timestamp: 2026-11-17T15:31:51.362573+00:00
- actor: claude-code
  id: 01m7n3as0y01hgfycanzw0zfcg
  text: |-
    ## Review Findings (2026-11-17 09:31) — round 2, review sha f5e9306d..896e531f

    Tool defect: the code-hygiene tool rules read file contents from the main checkout (/Users/wballard/github/swissarmyhammer/mlx-swift-lm), not from this worktree. They report "found no file at IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift", and they re-raise the 9 round-1 findings at Tests/MLXLMTests/LFM2DSparkTests.swift:302-469, LFM2MoeRoutingTests.swift:364 and LFM2Tests.swift:264. Those lines do not exist in the reviewed commit: the files there are 297, 277 and 187 lines long, and the change deleted that code. Under the written `review sha` scope rule ("Report a finding only on a line the change added or modified"), they are not findings on this change.

    To get real coverage, I ran `review file` on the worktree files themselves, through the path `.claude/worktrees/agent-a04c80279e15f5723/...`, and fixed each finding it raised on new lines:
    - [x] `LFM2CheckpointIntegrationTests.swift:355` `swift/naming-clarity` — the baseline `Sample.proposed` was filled with the emitted count. The baseline now records `proposed: 0` (it drafts nothing), checks `count == tokens`, and `Sample` documents its fields.
    - [x] `:54`, `:55`, `:81` `code-hygiene/magic-numbers-swift` — the literal prompt ids are gone; the DSpark tests now render `dsparkQuestion` through the target's chat template (`processor.prepare`). `shortPromptTokens` is derived from `shortPromptLength`.
    - [x] `:288` `reuse/reuse` — the copied throughput expectations now go through `expectPositiveRate`, which is reused by the DSpark benchmark (`:431` reuse finding).
    - [x] `:365` `reuse/reuse` — `baseline`/`speculative` share one generic `sample(tokens:makeIterator:drafts:)`.
    - [x] `:400` `swift/immutability` — the samples are built with `map` over the repetitions, which keeps the alternating order.
    - [x] `:151`, `:272` `reuse/reuse` — one `batch(_:)` helper builds every token batch.
    - [x] `:361` `reuse/reuse` — one `timed(_:)` stopwatch serves `measureForwardThroughput` and `sample`.
    - [x] `:236` `reuse/reuse` — `benchmarkForward(_:label:)` holds the measure/print/expect pipeline of both forward benchmarks.
    - [x] `:437` `swift/naming-clarity` — `baselineTPS`/`speculativeTPS` are renamed to `…TokensPerSecond`.
    - [x] `:316`, `:318` `swift/immutability` — `drain(_:)` (inout, so the speculative iterator keeps its counts) replaces the `var` accumulators.
    - [x] Fixed the same cause across the file: in Tests/MLXLMTests/LFM2DSparkTests.swift the 4 `[Int]()` inits are now `: [Int] = []`.
    Final `review file` results: LFM2CheckpointIntegrationTests.swift 0 findings; VerifyAttentionBlockPassProfileTests.swift 0 findings. IntegrationTesting build-for-testing reports TEST BUILD SUCCEEDED with no warnings in the changed files.
  timestamp: 2026-11-17T15:52:10.014398+00:00
position_column: review
position_ordinal: '80'
title: 'MLXLMTests: 4 LFM2 real-checkpoint XCTests skip behind environment variables (from upstream merge 6dc1e4e7)'
---
Found while working ^6gkhfah. `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` on qwen-perf (and on stable: the files are identical) reports `Executed 1134 tests, with 4 tests skipped`. CLAUDE.md says that no test is skipped.

The 4 skipped tests came in with the upstream squash merge 6dc1e4e7. An environment variable gates each one:
- `LFM2MoeRoutingTests.testDownloadedCheckpointLoadsWithMixedPrecisionRouting` (MLX_LFM2_MOE_MODEL_PATH)
- `LFM2MoeRoutingTests.testDownloadedCheckpointPerformance` (MLX_LFM2_MOE_BENCHMARK=1)
- `LFM2Tests.testDownloadedCheckpointLoadsThroughFactory` (MLX_LFM2_MODEL_PATH)
- `LFM2Tests.testDownloadedCheckpointPerformance` (MLX_LFM2_BENCHMARK=1)

These are real-checkpoint integration tests. Under the project rules they belong in the IntegrationTesting target, not in the unit bundle, and no test may be gated with an environment variable.

## Acceptance
- [x] Move the four tests to IntegrationTesting, or remove them from MLXLMTests, so that MLXLMTests reports 0 skipped.
- [x] Add a note to the CLAUDE.md "After an upstream merge" steps to check for upstream tests that an environment variable gates. #test-hygiene