---
assignees:
- claude-code
position_column: todo
position_ordinal: '9580'
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