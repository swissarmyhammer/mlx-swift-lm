---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6c8hx558vrwqmv6rnwfjgf6
  text: |-
    ### review — findings
    - evidence: `review sha 27198583~1..27198583`: 2 findings, 2 confirmed, 1 refuted — Libraries/MLXFoundationModels/TranscriptConverter.swift:298, Libraries/MLXFoundationModels/TranscriptConverter.swift:461
    - next: resolve both findings on wt-vx2bxy8, then re-review.

    Two candidates for follow-up, outside this card:
    - A `.data` attachment inside a `.response` or `.reasoning` entry is dropped at debug level by `extractText`.
    - Tool-output attachments are dropped with a logger warning (`extractToolOutputContent`).
  timestamp: 2026-11-01T19:14:34.789454+00:00
- actor: claude-code
  id: 01m6c8r90y99vhps3jtp2k5rjt
  text: |-
    ### review — clean
    - evidence: `review sha 95a05c8e~1..95a05c8e`: 0 findings, 0 confirmed, 0 refuted, 7 attempted, 0 failed, 0 skipped; 1 file reviewed, 0 not reviewed. Prior items at TranscriptConverter.swift:298 and :461 ticked, fixed in 95a05c8e. The `#if canImport(FoundationModels, _version: 2.1.14)` line stays at both `.data` cases because Swift cannot share a conditional switch case.
    - next: none on this card. 95a05c8e is on branch wt-vx2bxy8, not yet on mtp.
  timestamp: 2026-11-01T19:18:03.550027+00:00
position_column: done
position_ordinal: ffff9f80
title: 'SDK 27.2: Transcript.Entry.data is dropped silently by TranscriptConverter.convert'
---
Found in ^4eecsd4. The 27.2 FoundationModels interface adds `case data(Transcript.DataEntry)` to `Transcript.Entry`. `TranscriptConverter.convert(entry:)` in `Libraries/MLXFoundationModels/TranscriptConverter.swift` ends in a plain `default:` that logs at debug level and returns nil, so such an entry vanishes from the prompt with no error and no compiler warning. Attachments in tool output are dropped too, with a logger warning (`extractToolOutputContent`).

- [x] Read `Transcript.DataEntry` in the SDK interface and decide whether it renders, is skipped by contract, or throws `unsupportedTranscriptContent`.
- [x] Handle the case behind `#if canImport(FoundationModels, _version: 2.1.14)`, because the 27.0 SDK does not declare it.
- [x] Test it. The symbols need macOS 27.2 at run time; macOS 27.0.1 does not have them.

## Moved from ^3vwk2h3

Moved verbatim from the review of ^3vwk2h3, because this card owns the `.data` entry case.

- [x] `Libraries/MLXFoundationModels/TranscriptConverter.swift:289` `swift/state-modeling` — The `default:` in the `Transcript.Entry` switch swallows every future entry type. It returns `nil` and logs only at debug level. The comment calls the explicit `return nil` a tripwire that surfaces new types for review, but a debug log in a `default` does not surface anything, so a new case would be silently dropped. Name each remaining `Transcript.Entry` case explicitly so it gets deliberate handling, and use `@unknown default` with a `logger.warning` (not `debug`) for future cases, so that a new SDK case is reported rather than dropped quietly.
  - Note: Left for ^vx2bxy8. Naming the cases of `Transcript.Entry` means handling `.data` behind `#if canImport(FoundationModels, _version: 2.1.14)`, which is that card's work: an `@unknown default` without it warns on the 27.2 SDK.

#foundation-models

## Resolution

Commit 27198583 on branch wt-vx2bxy8. `.data` entries throw `unsupportedTranscriptContent`, behind the 2.1.14 guard.

Evidence: build ok; MLXFoundationModelsTests 471 pass; red before the change. The end-to-end test skips below macOS 27.2 and has not run, accepted as with ^4eecsd4.

## Review Findings (2026-11-01 13:12)

> Scope: `review sha 27198583~1..27198583` — reviewed the diffs only — lines this change added or modified. 3 file(s) reviewed, 0 not reviewed.

- [x] `Libraries/MLXFoundationModels/TranscriptConverter.swift:298` `duplication/duplication` — The two-line comment and `#if canImport(FoundationModels, _version: 2.1.14)` guard for the `.data` case are repeated verbatim from the `.data` attachment case at lines 397-399. The version rationale is written twice and can be updated in one place but not the other. Keep the full explanation in one place, such as a single comment on the `#if` guard or a shared constant (for example, `static let dataEntrySDKVersion`), and reference it from the other switch. Alternatively, shorten the second copy to a one-line pointer to the first.
  - Note: Fixed in 95a05c8e. The rationale comment now lives once, on the `#if` guard above the error helpers. The `#if canImport(FoundationModels, _version: 2.1.14)` line stays at both `.data` cases because Swift cannot share a conditional switch case.
- [x] `Libraries/MLXFoundationModels/TranscriptConverter.swift:461` `duplication/duplication` — `unsupportedDataEntry` is a near-verbatim copy of `unsupportedDataAttachment` (lines 440-447). Both build the same `unsupportedContent(described:in:)` call with a `contentType.identifier` message that differs only by the noun ("data entry" vs "data attachment") and the entry-type parameter. The two error messages can drift apart, and a future change to the shared wording has to be made in both places. Replace both with one helper that takes the noun as a parameter, e.g. `unsupportedDataContent(kind: String, contentType: UTType, in entry: Transcript.Entry)`, which builds the message `"This \(kind) of type \(contentType.identifier) carries content the MLX adapter cannot render."` plus the shared sentence. Keep `unsupportedDataAttachment` and `unsupportedDataEntry` as one-line wrappers that pass "data attachment" or "data entry" (or drop them and call the helper directly from the two sites).
  - Note: Fixed in 95a05c8e. Both functions are now thin wrappers over the private helper `unsupportedData(_:contentType:supported:in:)`.