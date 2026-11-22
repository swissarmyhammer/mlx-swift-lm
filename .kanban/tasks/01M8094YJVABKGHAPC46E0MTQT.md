---
assignees:
- claude-code
position_column: todo
position_ordinal: '8880'
title: 'MTP drafters: share the compatibility error and the configuration registration'
---
The ^04yn8q4 review added two shared helpers in MLXLMCommon, and only the Inkling 2 drafter uses them so far:
- `MTPDrafterModel.requireCompatibleTarget(_:expected:)` (MTPDrafterModel.swift) throws `MTPDrafterCompatibilityError.incompatibleTarget` naming the drafter type and the target type.
- `ModelTypeRegistry<any MTPDrafterModel>.registerDrafter(modelTypes:configuration:matches:create:)` (MTPDrafterModelFactory.swift) decodes a drafter configuration, validates it, and builds the drafter.

These drafters still write the guard-and-throw by hand: `LFM2DSparkDraftModel.validateCompatibility` (MLXLLM/Models/LFM2DSpark.swift), `Qwen35MTP` in MLXLLM and MLXVLM, and `Gemma4Assistant`. Check that each one implements `isCompatible(with:)` itself before you switch it, because the default `isCompatible` calls `validateCompatibility`. Gemma's error text names the mismatched property, so it may not fit the helper. Also move `LFM2DSparkRegistration.register` to `registerDrafter` if its `architectures` check can become a predicate on a decoded configuration. Existing tests: Gemma4AssistantDraftModelTests, MTPSpeculativeTokenIteratorTests, LFM2 DSpark tests.

#inkling