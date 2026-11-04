---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6awc5gkq5jqk46kh930jxyz
  text: |-
    Research and decision.

    - SDK facts: `Transcript.Attachment` gains `case data(Transcript.DataAttachment)` at iOS/macOS/visionOS/watchOS 27.2 and is `@nonexhaustive` in the 27.2 interface. The 27.0 interface has only `.image`. FoundationModels module version: 26.5 SDK 1.5.2, 27.0 SDK 2.0.68.1.402, 27.2 SDK 2.1.14.
    - The swiftdoc says a `DataAttachment` holds a value a `LanguageModel` opted into through `supportsDataAttachmentType(_:)`, and the framework validates each content type against that predicate before dispatching to the executor. MLXFoundationModels does not implement `supportsDataAttachmentType`.
    - Decision: `.data` throws `LanguageModelError.unsupportedTranscriptContent`, as `@unknown default` does, for every content type (an image-typed data attachment too). The message names the UTType.
    - Surprise 1: before this change a `.data` value already reached `@unknown default` and threw, so a behavior test of "it throws" would pass without the change. The observable change is the error text naming the content type.
    - Surprise 2: this machine runs macOS 27.0.1. `dlsym` finds the `ImageAttachment` metadata accessor in the OS FoundationModels and not the `DataAttachment` one, so a `Transcript.DataAttachment` cannot be built here. The switch arm cannot run here. Hence the seam `unsupportedDataAttachment(contentType:in:)`, which takes a `UTType`, tested on every OS, plus one `@available(27.2)` end-to-end test that Swift Testing reports as skipped here.
    - Guard: `#if canImport(FoundationModels, _version: 2.1.14)`, the version of the first SDK checked to declare the case, rather than `2.1`. An unknown 27.1 SDK then falls to `@unknown default` and still compiles.

    Did not work: `SDKROOT=<27.0 sdk> swift build`, and `swift build --sdk`, both exit 0 but the default swiftbuild backend ignores them: the `.d` file showed MacOSX27.2.sdk and the object held `DataAttachment` references. Those two builds proved nothing and were thrown away. `DEVELOPER_DIR=CommandLineTools swift build` on swiftbuild fails on `CompileMetalFile`. `--build-system native` does honor the SDK for library targets, and CommandLineTools has no Metal step on that backend. Test targets on native still compile against the Xcode 27.2 SDK, and CommandLineTools has no Testing/XCTest modules. A manual `swiftc -typecheck` of the test file against 27.0 failed with `cannot load module 'MLXLMCommon' built with SDK 'macosx27.2'`, thus the test file was not built against an older SDK.
  timestamp: 2026-11-01T06:22:29.395546+00:00
- actor: claude-code
  id: 01m6awcpe81st724bvkr25c9kx
  text: |-
    ### implement — changed
    - evidence: 2 files — Libraries/MLXFoundationModels/TranscriptConverter.swift, Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift.
    - RED: with a stub returning the old text, `testUnsupportedDataAttachmentErrorNamesContentType` failed on `content.debugDescription.contains(UTType.json.identifier)`, and the recompile showed "switch must be exhaustive" at TranscriptConverter.swift:365.
    - GREEN: `swift build --build-tests` recompiled MLXFoundationModels with no "switch must be exhaustive"; `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` 464 tests in 60 suites, 0 failures, exit 0, one skip: `testPromptDataAttachmentThrowsUnsupportedTranscriptContent` "Requires macOS 27.2". That test has never run, and the `.data` switch arm has never executed, because this machine is macOS 27.0.1.
    - Remaining build warning: "missing creator for mutated node ... mlx-swift_Cmlx.bundle", also in the y1hk1vz baseline log and unrelated to this change.
    - Older SDKs, library target only, `--build-system native`, confirmed from the `.d` dependency files: CommandLineTools Swift 6.4 + MacOSX27.0.sdk exit 0, 614 symbols, 0 `DataAttachment` references, no switch warning. Xcode swiftc + MacOSX27.0.sdk the same. MacOSX26.5.sdk exit 0 on both toolchains, 43 symbols, FoundationModels not imported (the file compiles out at `_version: 2`).
    - Guard probe with `swiftc -typecheck`: `_version: 2.1.14` is true on 27.2, false on 27.0 and 26.5. Without the guard, 27.0 fails with "type 'Transcript.Attachment' has no member 'data'", thus the guard is required.
    - Not built against an older SDK: the test file (its target always compiled against 27.2) and iOS/visionOS (the iPhoneOS 27.2 interface reports 2.1.14; no older iOS SDK is installed). No 27.1 SDK is installed.
    - swift-format lint with .swift-format: clean on both files.
    - next: /review. Follow-up card filed for `Transcript.Entry.data`, which `convert(entry:)` drops silently.
  timestamp: 2026-11-01T06:22:46.728366+00:00
- actor: claude-code
  id: 01m6ax3rcq9d4r4fc12dn9rgz6
  text: |-
    ### review — clean
    - evidence: `review file` on TranscriptConverter.swift counts 9 findings (9 confirmed, 3 refuted, 0 failed) and on TranscriptConverterTests.swift 1 finding (1 confirmed, 5 refuted, 0 failed). None is on a line this change wrote (changed lines: converter 11, 370-376, 378-381, 386-422; tests 9, 927-984). All 10 moved word for word to ^3vwk2h3, section "Moved from ^4eecsd4".
    - test file against older SDKs: the whole MLXFoundationModelsTests target typechecks, `swiftc -typecheck` exit 0, against MacOSX27.0.sdk and MacOSX26.5.sdk. Method: `swift build --build-tests --build-system native --scratch-path <scratch> --sdk <sdk> -v` for the dependency modules, then the logged test-target swiftc command with `-sdk` swapped, `Modules-tool` swapped for `Modules` (the SDK-built ones), and `-plugin-path` to Xcode's libFoundationModelsMacros. Typecheck only, Xcode 27.2 swiftc and its Testing module, no link and no run on those SDKs.
    - seam: `unsupportedDataAttachment` is absent from the 27.0-built MLXFoundationModels.swiftmodule and present in the 27.2-built one. It IS guarded out on pre-27.2 SDKs, and so are its only test caller and the test helper, under the same `#if`, so nothing references it there.
    - guard: module versions read from the interfaces: macOS, iOS, visionOS, watchOS 27.2 are 2.1.14; 27.0 is 2.0.68.1.402; 26.5 is 1.5.2. The `.data` switch probe typechecks on watchOS 27.2, iOS and visionOS targets at 27.0, macOS 27.0. tvOS fails on `Transcript` itself, unavailable there, not on this change.
    - skip: `xcrun xctest` on macOS 27.0.1, 464 tests in 60 suites passed; `testUnsupportedDataAttachmentErrorNamesContentType` passed; `testPromptDataAttachmentThrowsUnsupportedTranscriptContent` skipped "Requires macOS 27.2". The `.data` arm itself has still never executed.
    - next: none on this card. Not committed.
  timestamp: 2026-11-01T06:35:22.391157+00:00
position_column: done
position_ordinal: ffff9a80
title: 'Xcode 27.2: TranscriptConverter.extractImages warns "switch must be exhaustive" (new .data attachment content)'
---
Found in ^y1hk1vz. On a recompile of MLXFoundationModels with Xcode 27.2 (SDK 27.2), `Libraries/MLXFoundationModels/TranscriptConverter.swift` warns at the `switch attachment.content` in `extractImages`: "switch must be exhaustive", note "add missing case: '.data(_)'", even with its `@unknown default`. The file is unchanged by ^y1hk1vz, and the SDK's FoundationModels interface now has a `case data`. An incremental `swift build --build-tests` that does not recompile the target shows no warning, thus the warning hides until the next change to a dependency.

- [x] Decide how a `.data` attachment renders or fails, and handle the case.
- [x] Confirm the fix still compiles against the oldest SDK this package supports.

#foundation-models