---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m63z7mh396a2v3j7sdz4ckgx
  text: |-
    Implementation landed on branch worktree-agent-aeeaf061311927205 (based on mtp 9d7bacfc), commit a110b3ea.

    What changed:
    - `benchmarkMTPBlockSweep` -> `Benchmark.mtpBlockSweep` (public static, `extension Benchmark`, own `// MARK: - Runner`). The internal top-level `plainTokensWithCandidates` also moved onto Benchmark as a static for the same "a function belongs to a type" rule. Its one caller, inside `Benchmark.mtpGeneration`, resolves the unqualified name unchanged.
    - Every `String(format:)` in MTPBlockSweep.swift now goes through BenchmarkLog. The `%.0f` per-depth percentages had no helper, so I added `BenchmarkLog.whole(_:)` beside decimal/hundredths. `MTPBlockSetting.description` uses `hundredths(Double(Float))`. Output is byte-identical.
    - Named constants: `Benchmark.defaultMTPMaxTokens = 192` and `defaultMTPRuns = 3` (public, since public default arguments need public symbols), shared by mtpGeneration and mtpBlockSweep. `MTPBlockSweep.absentValue = "-"`.
    - Removed duplicated code: one private `fastest(among:)` serves both `fastest` and `fastestNoSlowerThanSingleDraft`. `summaryLines` splits into `MTPBlockSweepRow.summaryLine()` plus a private `closingLine()` with `settingName(of:)` and `loadName(_:)` helpers.
    - MTPBlockSweep.swift never used print, and it already wrote through BenchmarkLog (os.Logger).
    - New Tests/MLXLMTests/MTPBlockSweepTests.swift (9 tests): whole rounding, depth summary, result line, setting description, fastest / fastestNoSlowerThanSingleDraft, row line, closing line. RED was a compile failure on the missing `BenchmarkLog.whole`.

    Non-obvious: the validator dump for .swift is 657 KB (mostly per-language tool references), so it cannot be read in one call. I read the Swift and generic rules by section.
  timestamp: 2026-10-29T13:57:45.635208+00:00
- actor: claude-code
  id: 01m63z7tf2ee1hhd88rkbq9rp6
  text: |-
    ### implement — changed
    - evidence: branch worktree-agent-aeeaf061311927205 @ a110b3ea. 5 files: Libraries/BenchmarkHelpers/MTPBlockSweep.swift, Libraries/BenchmarkHelpers/MTPGenerationBenchmark.swift, Libraries/BenchmarkHelpers/BenchmarkHelpers.swift, IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift, Tests/MLXLMTests/MTPBlockSweepTests.swift (new).
    - `swift build --build-tests`: Build complete, no Swift warnings. One SwiftPM build-system line: "missing creator for mutated node ... mlx-swift_Cmlx.bundle". It comes from SwiftPM, not this code.
    - `xcrun xctest MLXLMTests.xctest`: XCTest 920 tests, 0 failures; Swift Testing 2035 tests in 190 suites passed; 0 skipped.
    - IntegrationTesting Release build-for-testing: ** TEST BUILD SUCCEEDED **. No source warnings; the only warning is appintentsmetadataprocessor "Metadata extraction skipped".
    - qwen38FourBitMTPGeneration() passed after 752.5 s, run under the bench lock. Means in tok/s: bs2/p0 82.7, bs3/p0 77.0, bs3/p.75 80.0, bs4/p0 77.4, bs4/p.75 85.6, bs5/p0 67.8, bs5/p.75 87.8, bs6/p0 64.4, bs6/p.75 87.2. Fastest, and fastest no slower than one draft: blockSize 5 pMin 0.75. Noise 8.4%, thermal state 1. Load average 8.75 before, 9.56 after; uptime showed 9.07/10.82/11.17 before and 9.56/12.32/11.81 after, so the machine was busy.
    - next: /review
  timestamp: 2026-10-29T13:57:51.714554+00:00
position_column: doing
position_ordinal: '80'
title: 'BenchmarkHelpers: move benchmarkMTPBlockSweep onto the Benchmark namespace'
---
## What
^gwyhs9h moved every public runner in the BenchmarkHelpers runner files onto `public enum Benchmark` (see Libraries/BenchmarkHelpers/BenchmarkHelpers.swift), per the `swift/immutability` rule "a function belongs to a type". `benchmarkMTPBlockSweep(context:drafter:settings:prompts:maxTokens:runs:)` in Libraries/BenchmarkHelpers/MTPBlockSweep.swift is still a public top-level func; it was outside the reviewed files of that card, so it was left alone.

- [x] Make it `public static func mtpBlockSweep(...)` in an `extension Benchmark` in MTPBlockSweep.swift.
- [x] Update the caller in IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift (`qwen38FourBitMTPGeneration`) and any doc links.
- [x] Also replace the `String(format:)` copies in MTPBlockSweep.swift with `BenchmarkLog.decimal` / `BenchmarkLog.hundredths` where they write result lines.

## Tests
- [x] `swift build --build-tests` clean; IntegrationTesting Release `build-for-testing` clean. #benchmark #mtp