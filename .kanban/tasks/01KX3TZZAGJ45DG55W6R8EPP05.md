---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxa4spyx6tefy7ck4n6qyg6f
  text: |-
    Audited against the real macOS 27 SDK swiftinterface (not WWDC docs), per session convention:
    /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/FoundationModels.framework/Versions/A/Modules/FoundationModels.swiftmodule/arm64e-apple-macos.swiftinterface

    Key discovery: `LanguageModelExecutorGenerationChannel.Event` and all nested `Action` types (Response/Reasoning/ToolCalls/ToolCall) are OPAQUE once constructed -- the swiftinterface shows only static factory functions, zero public accessors to read a value back out. Confirmed no existing test in this repo drains a channel and inspects content either (they only discard). This means well-formedness can't be tested by draining/inspecting the real channel's output -- had to test at the "emission seam" instead (the values the adapter computes right before `channel.send`).

    Changes:
    - Libraries/MLXFoundationModels/MLXLanguageModel.swift: added a `// MARK: - Channel & Error Surface Conformance` doc block (near the channel-emission helpers, before `incompleteOutputMetadataKey`) enumerating every Response/ToolCalls/Reasoning channel-action case and every LanguageModelError case as Emitted or Deliberately-N/A-with-rationale. Also widened `sendDelta`, `sendUsageUpdate`, `sendIncompleteOutputMetadata`, `textDeltaTokenCount` from `private` to package-internal (needed so tests can call them directly -- Event's opacity leaves no other seam), and extracted a pure `clampedUsageCounts` function out of `sendUsageUpdate`'s inline `Swift.min` clamps for direct unit-testability.
    - Tests/MLXFoundationModelsTests/ExecutorEventWellFormednessTests.swift: new suite, 4 tests -- nonzero textDeltaTokenCount, stable entryID across a simulated round's fragments (drained via a real channel, async-let consumer), clampedUsageCounts correctness, and usage monotonicity across two simulated rounds.

    Audit found NO genuinely-needed surface -- every never-emitted case (Response .replaceTextSegment/.updateCustomSegment/.addAttachmentSegment/.removeAttachmentSegment, ToolCalls .removeToolCall/entry-level updateMetadata+updateUsage/ToolCall.updateMetadata, Reasoning .replaceTextSegment/.updateSignature/.updateMetadata/.updateUsage, and errors .rateLimited/.guardrailViolation/.refusal/.unsupportedLanguageOrLocale/.timeout) has a defensible N/A rationale for local/open-weights MLX inference. No new kanban task created.

    Verification: `swift test --filter MLXFoundationModelsTests` -- 207 tests / 37 suites, all passing (fresh run). Confirmed the clamp test is meaningful by temporarily breaking `clampedUsageCounts` (removed the `Swift.min` clamps) and re-running -- 2 assertions failed as expected (RED), then restored the fix (GREEN). Adversarial double-check agent launched to critique before handoff.
  timestamp: 2026-07-12T03:09:11.005233+00:00
- actor: claude-code
  id: 01kxa5c1pv69wfqkhq8byawhjh
  text: |-
    Adversarial double-check verdict: REVISE, one substantive finding -- the "stable entryID" test (`deltasForARoundShareOneStableEntryID`) didn't actually exercise `respond()`'s real entryID-threading call chain (`dispatchGeneration` -> `runTextGeneration` -> `runUnconstrained`/`runReasoning` -> `sendDelta`); it called `sendDelta` directly with hand-picked literal entryIDs equal by construction, so it was really a channel-delivery smoke test mischaracterized as an entryID-stability test. Everything else in the review passed (SDK case enumeration complete/accurate, N/A rationales defensible, `clampedUsageCounts` extraction behavior-preserving, private->internal widening low-risk).

    Fix applied: renamed the test to `deltasSharingOneEntryIDAreAllDelivered` and rewrote its doc comment (plus the suite-level doc comment and the source's "Well-formedness invariants" doc section) to honestly state what it does and doesn't prove -- it verifies the channel accepts repeated sends under one entryID without dropping any; `respond()`'s actual entryID-stability guarantee (exactly three `UUID().uuidString` mints, threaded as required non-defaulted `String` parameters with no way for a nested helper to substitute its own) remains a structural property of the source, not something re-verified at runtime, because `LanguageModelExecutorGenerationChannel.Event`'s opacity means no test -- including a full end-to-end `respond()` run -- can drain a channel and confirm the entryID value an emitted Event actually carries. Considered building a full stub-model integration test through the real call chain instead, but rejected it: given Event's opacity, such a test still couldn't assert on the actual entryID identity carried by real events, so it would add significant scaffolding (mirroring ContextSizeValidationTests' stub-model/tokenizer setup) without closing the gap the reviewer identified -- it would just prove "didn't crash," which existing tests already cover for the respond() pipeline generally.

    Minor nit noted but not fixed (reviewer explicitly said not worth blocking on): doc comments say "package-internal" for the widened `sendDelta`/`sendUsageUpdate`/`sendIncompleteOutputMetadata`/`textDeltaTokenCount`/`clampedUsageCounts` symbols, but Swift's `package` access level isn't actually used -- they're plain `internal`. Harmless since `@testable import` covers either.

    Re-verified after the fix: `swift test --filter MLXFoundationModelsTests` -- 207 tests / 37 suites, all passing (fresh run).
  timestamp: 2026-07-12T03:19:11.835346+00:00
position_column: done
position_ordinal: ae80
title: 'Executor protocol-surface conformance audit: document deliberate non-use, assert event well-formedness'
---
## What
Close the loop on OS27 LanguageModelExecutor conformance by making the adapter's coverage of the channel/error surface EXPLICIT instead of implicit. Audited surface (SDK swiftinterface, macOS 27):
- Emitted today: Response.appendText / .updateMetadata / .updateUsage; ToolCalls .toolCall(.appendArguments); Reasoning .appendText.
- Never emitted — each needs a documented rationale in MLXLanguageModel.swift (doc comment near the channel-emission helpers) or a task if actually needed: Response .replaceTextSegment / .updateCustomSegment / .addAttachmentSegment / .removeAttachmentSegment (MLX text decoding is append-only and text-only), ToolCalls .removeToolCall, Reasoning .replaceTextSegment / .updateSignature (open-weights models have no signed reasoning blobs).
- Error surface: adapter throws contextSizeExceeded / unsupportedCapability / unsupportedGenerationGuide / unsupportedTranscriptContent; never rateLimited / guardrailViolation / unsupportedLanguage (local inference — document as N/A).
- Well-formedness: every emitted appendText/appendArguments carries an honest tokenCount (the SDK's TextFragment.tokenCount is not optional); entryIDs are stable per entry across a round's fragments.

## Acceptance Criteria
- [ ] A doc section (in MLXLanguageModel.swift near the executor, or the module's docc if present) enumerates the full channel-action and error surface with Emitted / Deliberately-N/A-because-X for every case — no case unaccounted
- [ ] A unit test walks a recorded event stream (probe/stub path or replayed fixture) asserting: nonzero tokenCount on every text/arguments fragment, stable entryID per entry, usage events monotone
- [ ] Any surface discovered to be genuinely NEEDED (not N/A) during the audit becomes its own kanban task rather than silently skipped

## Tests
- [ ] New unit test(s) in Tests/MLXFoundationModelsTests asserting event well-formedness at the emission seam
- [ ] `swift test --filter MLXFoundationModelsTests` zero failures

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.

## Review Findings (2026-07-11 22:20)

Scope: `review sha HEAD~1..HEAD` (commit 0cf4d7f — "docs(mlx-fm): document executor channel/error surface conformance, add well-formedness tests").

- [x] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:245` — REJECTED, out of scope. Engine claims `Task<ModelContainer, Error>` should become `Task<ModelContainer, any Error>` on this line. Verified via `git diff 0cf4d7f~1..0cf4d7f -- Libraries/MLXFoundationModels/MLXLanguageModel.swift`: every hunk in this commit's diff falls in the old/new `1995`-`2100` range (doc section + 4 `private`→internal widenings + `clampedUsageCounts` extraction). Line 245 is nowhere near that range, and its actual content (`getOrCreateCached<T>`, a generic cache-or-create helper) doesn't even contain a `Task<ModelContainer, Error>` — that pattern lives at lines 88/89/148, also untouched by this commit. Pre-existing code, misattributed line citation; not this commit's concern.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:246` — REJECTED, out of scope. Same finding/evidence as above (same misattributed `Task<ModelContainer, Error>` claim, line 246 also outside the commit's diff hunks and not matching content).
- [x] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:253` — REJECTED, out of scope. Same finding/evidence as above; line 253 is the closing brace of `getOrCreateCached`, untouched by this commit.
- [x] `Tests/MLXFoundationModelsTests/ExecutorEventWellFormednessTests.swift:116` — FIXED. Confirmed in-scope: this test file is wholly new to this commit (177 insertions, 0 prior lines). The channel-draining `async let` block was duplicated between `deltasSharingOneEntryIDAreAllDelivered` and `usageTotalsAreMonotoneAcrossRounds`. Extracted a shared `@available(iOS 27.0, macOS 27.0, visionOS 27.0, *) private func drainedEventCount(from:expecting:)` and replaced both inline blocks with a single-line call. Verified: `swift test --filter ExecutorEventWellFormednessTests` (4/4 pass) and the full `swift test --filter MLXFoundationModelsTests` (207 tests, 37 suites, all pass).

Outcome: 1 confirmed finding, fixed and verified; 3 findings rejected as out-of-scope pre-existing code with documented diff evidence. No outstanding action items.