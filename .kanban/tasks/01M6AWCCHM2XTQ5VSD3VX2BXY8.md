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

#foundation-models