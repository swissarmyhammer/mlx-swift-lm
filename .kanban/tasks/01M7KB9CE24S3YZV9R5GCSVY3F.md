---
assignees:
- claude-code
position_column: todo
position_ordinal: '9380'
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