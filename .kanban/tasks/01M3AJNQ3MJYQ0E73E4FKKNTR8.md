---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: TurboQuantIntegrationTests.testStandardAttentionRepeatFactors fails intermittently (cos 0.888 < 0.95 at rep=2)
---
## What

`MLXLMTests.TurboQuantIntegrationTests.testStandardAttentionRepeatFactors` (`Tests/MLXLMTests/TurboQuantTests.swift:2190`) fails on some runs and passes on others, with the same seeds.

Measured on 2026-09-24, on branch `stable`:
- Full MLXLMTests run: failed with `XCTAssertGreaterThan failed: ("0.88828874") is not greater than ("0.95") - rep=2: cos 0.88828874`.
- `xcrun xctest -XCTest MLXLMTests.TurboQuantIntegrationTests .build/out/Products/Debug/MLXLMTests.xctest`, 5 times: 1 failure, 4 passes.

The inputs use fixed `MLXRandom.key` seeds and `TurboQuantKVCache(seed: 5)`, thus the result must be deterministic. A result that changes from run to run points to a race in the compressed-attention kernel for GQA rep=2 (the NR0 fast path), or to uninitialized memory.

MLXLMTests does not depend on MLXFoundationModels. The failure was found during the test step of ^2mk47nr, which changes only MLXFoundationModels. It is not in the known baseline (the ...ContinuationTests suites and NanbeigeTests).

## Acceptance Criteria
- [ ] Find the cause of the nondeterminism and correct it.
- [ ] The test passes on 20 consecutive runs of the TurboQuantIntegrationTests suite.

## Tests
- [ ] `for i in $(seq 20); do xcrun xctest -XCTest MLXLMTests.TurboQuantIntegrationTests .build/out/Products/Debug/MLXLMTests.xctest || break; done`