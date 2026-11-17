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
position_column: doing
position_ordinal: '8280'
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