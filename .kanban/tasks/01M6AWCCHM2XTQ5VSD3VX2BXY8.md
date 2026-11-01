---
assignees:
- claude-code
position_column: todo
position_ordinal: '9480'
title: 'SDK 27.2: Transcript.Entry.data is dropped silently by TranscriptConverter.convert'
---
Found in ^4eecsd4. The 27.2 FoundationModels interface adds `case data(Transcript.DataEntry)` to `Transcript.Entry`. `TranscriptConverter.convert(entry:)` in `Libraries/MLXFoundationModels/TranscriptConverter.swift` ends in a plain `default:` that logs at debug level and returns nil, so such an entry vanishes from the prompt with no error and no compiler warning. Attachments in tool output are dropped too, with a logger warning (`extractToolOutputContent`).

- [ ] Read `Transcript.DataEntry` in the SDK interface and decide whether it renders, is skipped by contract, or throws `unsupportedTranscriptContent`.
- [ ] Handle the case behind `#if canImport(FoundationModels, _version: 2.1.14)`, because the 27.0 SDK does not declare it.
- [ ] Test it. The symbols need macOS 27.2 at run time; macOS 27.0.1 does not have them.

## Moved from ^3vwk2h3

Moved verbatim from the review of ^3vwk2h3, because this card owns the `.data` entry case.

- [ ] `Libraries/MLXFoundationModels/TranscriptConverter.swift:289` `swift/state-modeling` — The `default:` in the `Transcript.Entry` switch swallows every future entry type. It returns `nil` and logs only at debug level. The comment calls the explicit `return nil` a tripwire that surfaces new types for review, but a debug log in a `default` does not surface anything, so a new case would be silently dropped. Name each remaining `Transcript.Entry` case explicitly so it gets deliberate handling, and use `@unknown default` with a `logger.warning` (not `debug`) for future cases, so that a new SDK case is reported rather than dropped quietly.
  - Note: Left for ^vx2bxy8. Naming the cases of `Transcript.Entry` means handling `.data` behind `#if canImport(FoundationModels, _version: 2.1.14)`, which is that card's work: an `@unknown default` without it warns on the 27.2 SDK.

#foundation-models