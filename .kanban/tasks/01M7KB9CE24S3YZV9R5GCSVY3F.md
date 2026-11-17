---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7n0j328ns2wqcnnzp6zfcd0
  text: '2026-11-17 implement: Moved the 4 env-gated LFM2 XCTests (LFM2Tests x2, LFM2MoeRoutingTests x2) plus the 2 env-gated DSpark Swift Testing tests in LFM2DSparkTests (same cause: MLX_LFM2_MODEL_PATH / MLX_LFM2_DSPARK_* gates, silent early return) out of MLXLMTests into IntegrationTesting/IntegrationTestingTests/LFM2CheckpointIntegrationTests.swift. Each now downloads a pinned checkpoint via #hubDownloader + resolve(): LiquidAI/LFM2.5-2.6B-MLX@b41f2b65 (bf16 subdir), LiquidAI/LFM2.5-8B-A1B-MLX-4bit@d16665ca (the revision with the 8-bit layer-2 router override; 760bf10c+ stores the gate in fp32), LiquidAI/LFM2.5-2.6B-DSpark@458cedab. The upstream assertion `generation_config temperature == 0.1` does not match the published checkpoint (0.2 at every revision); the moved test asserts 0.2 at the pin and that the factory-loaded generationConfig equals the file. Removed the now-unused PseudoWordTokenizer stubs and MLXScriptedLM imports. Added CLAUDE.md "After an upstream merge" step 4 on env-gated tests. Found but out of scope: Tests/MLXLMTests/VerifyAttentionBlockPassProfileTests.swift is gated by .enabled(if: MLX_VERIFY_ATTENTION_PROFILE) (fork code, attention kernels).'
  timestamp: 2026-11-17T15:03:43.944382+00:00
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
- [ ] Move the four tests to IntegrationTesting, or remove them from MLXLMTests, so that MLXLMTests reports 0 skipped.
- [ ] Add a note to the CLAUDE.md "After an upstream merge" steps to check for upstream tests that an environment variable gates. #test-hygiene