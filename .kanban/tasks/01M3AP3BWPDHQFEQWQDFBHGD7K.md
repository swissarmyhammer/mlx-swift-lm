---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'MLXLMTests: 36 XCTest failures and 101 Swift Testing issues in numeric model tests on this machine'
---
## What

`xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` fails on the `stable` branch (HEAD 73cb5db plus the uncommitted work of ^jar6qq9, which changes no file in the MLXLMTests target graph). Two runs gave the same result: `Executed 694 tests, with 36 failures`, and `Test run with 1207 tests in 91 suites failed ... with 101 issues`.

The failures are numeric tolerance failures:
- XCTest: `GlmOcrContinuationTests`, `NanbeigeTests.testWarmContinuationMatchesFullPrefill`, `Qwen25VLContinuationTests`, `Qwen35ContinuationTests`, `Qwen3VLContinuationTests` (for example `0.0020094514 is greater than 0.001`), `testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections`, `testQwen35VLMGDNCheckpointMatchesPrefix`, `testSSMAttnChunkedMatchesUnchunkedValuesAndGradients`.
- Swift Testing: `attentionSinkChangesTheOutput`, `compressedLayerReadsTheCompressRopeTheta`, `decodeStepMatchesThePythonReference`, `prefillMatchesThePythonReference`, `theSortedRoutingPathAgreesWithTheUnsortedPath`, and the `theCollapse*` / `theExpand*` / `theHeadReduce*` / `theRoundTrip*` tests.

## Acceptance Criteria
- [ ] Find the cause (OS or Metal change, or a code change) of the divergence.
- [ ] `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes with zero failures.