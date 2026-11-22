---
assignees:
- claude-code
position_column: done
position_ordinal: '8880'
title: 'MTP drafters: share the compatibility error and the configuration registration'
---
The ^04yn8q4 review added two shared helpers in MLXLMCommon, and only the Inkling 2 drafter uses them so far:
- `MTPDrafterModel.requireCompatibleTarget(_:expected:)` (MTPDrafterModel.swift) throws `MTPDrafterCompatibilityError.incompatibleTarget` naming the drafter type and the target type.
- `ModelTypeRegistry<any MTPDrafterModel>.registerDrafter(modelTypes:configuration:matches:create:)` (MTPDrafterModelFactory.swift) decodes a drafter configuration, validates it, and builds the drafter.

These drafters still write the guard-and-throw by hand: `LFM2DSparkDraftModel.validateCompatibility` (MLXLLM/Models/LFM2DSpark.swift), `Qwen35MTP` in MLXLLM and MLXVLM, and `Gemma4Assistant`. Check that each one implements `isCompatible(with:)` itself before you switch it, because the default `isCompatible` calls `validateCompatibility`. Gemma's error text names the mismatched property, so it may not fit the helper. Also move `LFM2DSparkRegistration.register` to `registerDrafter` if its `architectures` check can become a predicate on a decoded configuration. Existing tests: Gemma4AssistantDraftModelTests, MTPSpeculativeTokenIteratorTests, LFM2 DSpark tests.

#inkling

## Resolution

Commits 2e4a6bec, 3c306a58 (branch inkling2-side).

- `ModelTypeRegistry<any MTPDrafterModel>` gains the unconditional `registerDrafter(modelTypes:configuration:create:)` (replaces the creator of each model type); both variants register the creator of `drafterCreator(_:create:)`: json5 decode, validation of a `ModelConfigurationValidating` configuration, build. `create` may throw.
- Moved to `registerDrafter`: `Qwen35TextMTPRegistration`, `Qwen35VLMMTPRegistration`, `Gemma4AssistantRegistration`, and `LFM2DSparkRegistration` (its `architectures` check is now a predicate on the decoded `LFM2DSparkConfiguration`).
- `LFM2DSparkDraftModel.validateCompatibility` calls `requireCompatibleTarget`; error text unchanged.
- Kept hand-written: Qwen35 MTP (MLXLLM, MLXVLM) and Gemma4 `validateCompatibility`. Neither implements `isCompatible` (the default `isCompatible` calls `validateCompatibility`, so the helper would recurse), and their errors name the mismatched target dimensions/property, which the helper's `actual` (type name only) cannot carry.
- Behaviour change: a `qwen3` config with the DSpark architecture that fails to decode now reports `unsupportedModelType` instead of the decoding error (the Inkling drafter registration already behaves so).
- Tests added: DSpark validation through the registry, the exact DSpark error text, the unconditional variant validating, replacing an earlier creator and passing the decoding error through.
- Tests: all 5 bundles green (MLXLMTests 1258 XCTest + 2573 Swift Testing, MLXGuidedGenerationTests 86, MLXFoundationModelsTests 583, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11).
- Review: second pass flagged only the unchanged `MTPDrafterRegistry` constants -> follow-up ^vwarz1.
