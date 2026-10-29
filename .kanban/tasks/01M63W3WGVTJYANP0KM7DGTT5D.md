---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'BenchmarkHelpers: move benchmarkMTPBlockSweep onto the Benchmark namespace'
---
## What
^gwyhs9h moved every public runner in the BenchmarkHelpers runner files onto `public enum Benchmark` (see Libraries/BenchmarkHelpers/BenchmarkHelpers.swift), per the `swift/immutability` rule "a function belongs to a type". `benchmarkMTPBlockSweep(context:drafter:settings:prompts:maxTokens:runs:)` in Libraries/BenchmarkHelpers/MTPBlockSweep.swift is still a public top-level func; it was outside the reviewed files of that card, so it was left alone.

- [ ] Make it `public static func mtpBlockSweep(...)` in an `extension Benchmark` in MTPBlockSweep.swift.
- [ ] Update the caller in IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift (`qwen38FourBitMTPGeneration`) and any doc links.
- [ ] Also replace the `String(format:)` copies in MTPBlockSweep.swift with `BenchmarkLog.decimal` / `BenchmarkLog.hundredths` where they write result lines.

## Tests
- [ ] `swift build --build-tests` clean; IntegrationTesting Release `build-for-testing` clean. #benchmark #mtp