---
assignees:
- claude-code
position_column: todo
position_ordinal: 9a80
title: Move MTPGreedyVerificationTests.benchmarkVerification off its MTP_GREEDY_BENCHMARK switch
---
## What
Seen again while doing ^kqvf7kp: every `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` run reports `➜ Test benchmarkVerification() skipped.` The test (Tests/MLXLMTests/MTPGreedyVerificationTests.swift) is `@Test(.enabled(if: ProcessInfo.processInfo.environment["MTP_GREEDY_BENCHMARK"] == "1"))`, so the unit bundle never runs it and a green run is never zero-skipped. The test-partitioning rule forbids an environment variable that selects a test: a benchmark belongs in a target of its own (IntegrationTesting, or a benchmark target) that CI and the developer select directly.

Earlier cards (^gg7bvm7 comments) noted it as pre-existing without a card of its own.

- [ ] Move the benchmark into a target that selects it structurally, or delete it if IntegrationTesting `GenerationBenchmarkTests` already covers it.
- [ ] `MLXLMTests` reports zero skipped tests.

#mtp #test