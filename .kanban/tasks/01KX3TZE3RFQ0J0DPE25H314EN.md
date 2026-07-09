---
assignees:
- claude-code
position_column: todo
position_ordinal: '9480'
title: Honor ContextOptions.includeSchemaInPrompt in guided generation (protocol fidelity gap)
---
## What
OS27 protocol gap found by auditing the SDK surface against the adapter: `FoundationModels.ContextOptions` has TWO fields — `reasoningLevel` (fully handled in MLXLanguageModel.swift) and `includeSchemaInPrompt: Bool?` (ZERO references in Libraries/, Tests/, IntegrationTesting/). The framework/app uses it to signal whether the generation schema is already embedded in the prompt; the executor must not make its own injection decision when told.

In Libraries/MLXFoundationModels/MLXLanguageModel.swift's guided-generation path: read `request.contextOptions.includeSchemaInPrompt` and gate the adapter's own schema-envelope injection accordingly — when the app says the schema is already in the prompt, do NOT inject a second copy (wasted context tokens + model confusion); when false/nil, preserve current behavior (adapter injects). Trace exactly where the schema text currently enters the prompt (SchemaConverter / guided prompt assembly) and thread the flag there. Document the chosen nil-semantics against Apple's docs for the field.

## Acceptance Criteria
- [ ] With includeSchemaInPrompt == true, the tokenized prompt contains exactly one schema rendering (the transcript's own), and the adapter injects nothing extra — asserted at the prompt-construction seam
- [ ] With false/nil, current behavior is unchanged (regression-guarded by existing guided-generation tests)
- [ ] Constrained decoding (grammar) still applies in BOTH cases — the flag governs prompt text only, not the sampling constraint

## Tests
- [ ] Unit tests at the prompt-assembly seam (Tests/MLXFoundationModelsTests): schema appears once vs twice across the flag's three values
- [ ] Existing guided-generation integration suites stay green
- [ ] `swift test --filter MLXFoundationModelsTests` zero failures

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass.