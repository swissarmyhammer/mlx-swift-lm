---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: Guided generation pins the prefill step to 512 and overrides the model default
---
## Why
Found in ^tty395a. `PrefillParameters.stepSize == nil` lets each model pick its own prefill step (Gemma3Text 128, Flint 2048 since ^tty395a). Guided generation never passes nil: `GuidedGenerationLoop.run` defaults `prefill` to `.init(stepSize: PrefillParameters.defaultStepSize)`, and `MLXLanguageModelGeneration` passes `Self.prefillStepSize ?? PrefillParameters.defaultStepSize`. So a guided (schema / grammar) FoundationModels request on Flint prefills in 512-token chunks, which measured ~25% slower than 2048 at 8k-16k on oQ4e, and Gemma 3 gets 512 instead of its tuned 128.

## What
- Pass `stepSize: nil` (or the caller's value) through both sites, so `prepare` picks the model default.
- Check every other site that writes `PrefillParameters.defaultStepSize` explicitly where nil was meant.

## Acceptance
- A test shows a guided-generation prefill on a model with its own default reads that default.
- All five bundles are green. #flint-perf