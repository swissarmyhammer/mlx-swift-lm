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
- actor: claude-code
  id: 01m63523dpnthsjm3t8t7ann80
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (67e6a852): 3 findings — IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift:17, :26, :92
    - next: implement the 3 open items in the 2026-10-29 01:19 Review Findings section
  timestamp: 2026-10-29T06:20:21.302824+00:00
- actor: claude-code
  id: 01m6352gs750qgt4sd69bavkx5
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 3 files (benchmark to IntegrationTesting, shared SequentialGreedyMTPVerification)
    - test: green — 5 bundles 0 failures 0 skipped; Release benchmark passed
    - commit: 67e6a852
    - review: findings — MTPGreedyVerificationBenchmarkTests.swift:17, :26, :92
  timestamp: 2026-10-29T06:20:34.983995+00:00
- actor: claude-code
  id: 01m6356p3f8n1yqrsr070m028p
  text: |-
    Fixed the 3 findings from 2026-10-29 01:19 in IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift.

    - magic-numbers (block sizes `[2, 4, 8]`): replaced with `private enum BenchmarkBlockSize: Int, CaseIterable { case oneDraft = 2, threeDrafts = 4, sevenDrafts = 8 }`. Enum raw values are a swiftlint carve-out. The test iterates `BenchmarkBlockSize.allCases.map(\.rawValue)`.
    - magic-numbers (`Int32(9)`): now `private let rejectedDraft: Int32 = 9`. A typed declaration is carved out, but a call argument is not.
    - The same cause elsewhere in the file: `(blockSize - 1) / 2` is now `draftCount / halfOfDraftsDivisor`, a named constant. swiftlint had not flagged it.
    - stdout: `print(String(format:))` is now a file-scope `os` `Logger` (`subsystem: "com.apple.FoundationModels-MLX", category: "MTPGreedyVerificationBenchmark"`), the same pattern as Qwen35SessionPromptCacheTests and PromptCacheSpoolCostAssessmentTests, logged with `.info("\(measurement, privacy: .public)")`. The `[MTPGREEDYBENCH]` prefix is kept so the line is still easy to grep. No print/debugPrint/dump/_printChanges remains (rg exit 1).

    Verification:
    - swiftlint 0.65.1 `no_magic_numbers` using the validator's settings (`allowed_numbers: [0, 1, -1, 100]`), with a temp config in scratchpad: 0 findings. Before the change it reported 17:36, 17:39, 17:42 and 26:35.
    - `no_direct_standard_out_logs` is not a built-in swiftlint rule (swiftlint rejects that id), so the stdout check was done with rg.
    - `swift build --build-tests`: Build complete, 0 compiler warnings. The only line is the existing SwiftPM "missing creator for mutated node" note on the Cmlx bundle. The file belongs to the xcodeproj, not to SwiftPM.
    - Release xcodebuild `-only-testing:IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests`: TEST SUCCEEDED, 1 test passed in 5.32 s. No warning names this file; the only warning is the existing appintents metadata note. The Logger lines show up in the xcodebuild output, e.g. `[MTPGreedyVerificationBenchmark] [MTPGREEDYBENCH] block=8 accepted=7 sequential=2208.7 us batched=236.7 us speedup=9.33x`.
  timestamp: 2026-10-29T06:22:51.503521+00:00
- actor: claude-code
  id: 01m6356rf7v3v50xwp3cg99jj0
  text: |-
    ### implement — changed
    - evidence: 1 file — IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift. swiftlint no_magic_numbers 0 findings (was 4). No stdout writes left. `swift build --build-tests` 0 compiler warnings. Release xcodebuild MTPGreedyVerificationBenchmarkTests TEST SUCCEEDED (1 test, 5.32 s). All 3 findings checked.
    - next: /review
  timestamp: 2026-10-29T06:22:53.927405+00:00
- actor: claude-code
  id: 01m6358e4ygmxqmctefbrgw9p4
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (dc132d4a) — 0 findings, 0 confirmed, 0 refuted (7 attempted); all 3 items in ## Review Findings (2026-10-29 01:19) checked
    - next: task moved to done
  timestamp: 2026-10-29T06:23:48.894915+00:00
- actor: claude-code
  id: 01m6358t128q6kq5c7k85w3nar
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 file (named constants, os Logger)
    - test: green — build 0 warnings; Release MTPGreedyVerificationBenchmarkTests passed
    - commit: dc132d4a
    - review: clean — task moved to done
  timestamp: 2026-10-29T06:24:01.058503+00:00
position_column: done
position_ordinal: fffd80
title: Move MTPGreedyVerificationTests.benchmarkVerification off its MTP_GREEDY_BENCHMARK switch
---
## What
Seen again while doing ^kqvf7kp: every `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` run reports `➜ Test benchmarkVerification() skipped.` The test (Tests/MLXLMTests/MTPGreedyVerificationTests.swift) is `@Test(.enabled(if: ProcessInfo.processInfo.environment["MTP_GREEDY_BENCHMARK"] == "1"))`, so the unit bundle never runs it and a green run is never zero-skipped. The test-partitioning rule forbids an environment variable that selects a test: a benchmark belongs in a target of its own (IntegrationTesting, or a benchmark target) that CI and the developer select directly.

Earlier cards (^gg7bvm7 comments) noted it as pre-existing without a card of its own.

- [x] Move the benchmark into a target that selects it structurally, or delete it if IntegrationTesting `GenerationBenchmarkTests` already covers it.
- [x] `MLXLMTests` reports zero skipped tests.

#mtp #test

## Review Findings (2026-10-29 01:19)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 3 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M61YHNPNARJ8AQABVSMN05MJ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M61YHNPNARJ8AQABVSMN05MJ.md` — no validator matches this file
> - `.kanban/tasks/01M634YDZDNSBDD28B055QC043.jsonl` — no validator matches this file
> - `.kanban/tasks/01M634YDZDNSBDD28B055QC043.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift:17` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift:26` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/MTPGreedyVerificationBenchmarkTests.swift:92` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
