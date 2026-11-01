---
assignees:
- claude-code
position_column: todo
position_ordinal: '9180'
title: 'Xcode 27.2: TranscriptConverter.extractImages warns "switch must be exhaustive" (new .data attachment content)'
---
Found in ^y1hk1vz. On a recompile of MLXFoundationModels with Xcode 27.2 (SDK 27.2), `Libraries/MLXFoundationModels/TranscriptConverter.swift` warns at the `switch attachment.content` in `extractImages`: "switch must be exhaustive", note "add missing case: '.data(_)'", even with its `@unknown default`. The file is unchanged by ^y1hk1vz, and the SDK's FoundationModels interface now has a `case data`. An incremental `swift build --build-tests` that does not recompile the target shows no warning, thus the warning hides until the next change to a dependency.

- [ ] Decide how a `.data` attachment renders or fails, and handle the case.
- [ ] Confirm the fix still compiles against the oldest SDK this package supports.

#foundation-models