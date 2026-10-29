---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m634z3xz8skmngg0sx6y56r5
  text: |-
    Research + implementation.

    Research: the micro-benchmark loads no model, so GenerationBenchmarkTests does not cover it (that suite times full decodes). The closest existing pattern is IntegrationTesting `MultiRowQuantizedMatvecBenchmarkTests`: a model-free kernel timing that runs unconditionally in the IntegrationTesting target (a PBXFileSystemSynchronizedRootGroup, so a new file needs no pbxproj edit). Both MLXLMTests and IntegrationTesting already link `BenchmarkHelpers`, so the per-row sequential reference that the unit tests and the benchmark both need now lives there, which avoids copying it.

    Changes:
    - New `Libraries/BenchmarkHelpers/SequentialGreedyMTPVerification.swift`: `public enum SequentialGreedyMTPVerification { static func verify(logits:draftTokens:) }`, the old private `sequentialGreedyVerification` helper moved here unchanged apart from `var tokens: [Int] = []` and the loop variable `row`.
    - `Tests/MLXLMTests/MTPGreedyVerificationTests.swift`: deleted `benchmarkVerification` (the `MTP_GREEDY_BENCHMARK` switch) and the private helper; the two parity tests call `SequentialGreedyMTPVerification.verify`; dropped the now-unused `import Foundation`.
    - New `IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift`: the benchmark with no gate, constants named, median through `BenchmarkStats`, paths alternating order per repetition as before, and a parity `#expect` per configuration (accepted count and tokens against the reference) in place of the old `checksum > 0`.

    Discovery: `MTPGreedyVerificationIntegrationTests.tokenAndCacheParity` still reads `MTP_GREEDY_BENCHMARK` (a `guard ... else { return }` before its end-to-end timing). It is outside this card; filed ^55qc043.

    Results: `swift build --build-tests` compiled with 0 compiler warnings (the only line is SwiftPM's "missing creator for mutated node" note on the Cmlx bundle). xctest on all 5 bundles: 0 failures, 0 skipped. MLXLMTests: 920 XCTest + 1991 Swift Testing. MLXFoundationModelsTests: 404. MLXGuidedGenerationTests: 70. CXGrammarTests: 7. MLXHuggingFaceMacrosTests: 6 XCTest. Release xcodebuild `-only-testing:IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests`: TEST SUCCEEDED, 1 test passed in 5.4 s. Its only warnings are the pre-existing mlx Metal defines.h unused-const lines and the appintents note.

    [MTPGREEDYBENCH] (M5, Release), median us sequential / batched:
    block=2 accepted=0: 504.3 / 261.7 (1.93x), and again 476.9 / 238.2 (2.00x), since (2-1)/2 == 0 as in the original
    block=2 accepted=1: 770.2 / 263.9 (2.92x)
    block=4 accepted=0: 478.3 / 220.3 (2.17x); accepted=1: 776.6 / 261.8 (2.97x); accepted=3: 1165.9 / 252.5 (4.62x)
    block=8 accepted=0: 501.7 / 256.1 (1.96x); accepted=3: 1262.7 / 278.4 (4.54x); accepted=7: 2099.6 / 228.4 (9.19x)

    Note: the card's description update (both subtasks checked) went through; the tool output was too large to display.
  timestamp: 2026-10-29T06:18:43.519019+00:00
- actor: claude-code
  id: 01m634z8s0wvf9yenhn58ecxjd
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/BenchmarkHelpers/SequentialGreedyMTPVerification.swift (new), Tests/MLXLMTests/MTPGreedyVerificationTests.swift, IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift (new). `swift build --build-tests` 0 compiler warnings; xctest 5 bundles 0 failures 0 skipped (MLXLMTests 920 XCTest + 1991 Swift Testing); Release xcodebuild MTPGreedyVerificationBenchmarkTests TEST SUCCEEDED (block 8, all 7 accepted: 2099.6 us sequential vs 228.4 us batched, 9.19x).
    - next: /review. Follow-up ^55qc043 for the remaining MTP_GREEDY_BENCHMARK reader in MTPGreedyVerificationIntegrationTests.
  timestamp: 2026-10-29T06:18:48.480087+00:00
position_column: doing
position_ordinal: '8180'
title: Move MTPGreedyVerificationTests.benchmarkVerification off its MTP_GREEDY_BENCHMARK switch
---
## What
Seen again while doing ^kqvf7kp: every `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` run reports `➜ Test benchmarkVerification() skipped.` The test (Tests/MLXLMTests/MTPGreedyVerificationTests.swift) is `@Test(.enabled(if: ProcessInfo.processInfo.environment["MTP_GREEDY_BENCHMARK"] == "1"))`, so the unit bundle never runs it and a green run is never zero-skipped. The test-partitioning rule forbids an environment variable that selects a test: a benchmark belongs in a target of its own (IntegrationTesting, or a benchmark target) that CI and the developer select directly.

Earlier cards (^gg7bvm7 comments) noted it as pre-existing without a card of its own.

- [x] Move the benchmark into a target that selects it structurally, or delete it if IntegrationTesting `GenerationBenchmarkTests` already covers it.
- [x] `MLXLMTests` reports zero skipped tests.

#mtp #test