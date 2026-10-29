---
assignees:
- claude-code
position_column: todo
position_ordinal: a180
title: 'FoundationModels: Qwen3.8 loaded by the default factory order is the VLM model, so a `.qwen35Text` MTP drafter never drafts'
---
## What
`loadModelContainer(from:using:configuration:)` tries `MLXVLM.TrampolineModelFactory` first (Libraries/MLXLMCommon/ModelFactory.swift, `ModelFactoryRegistry.init`), and `mlx-community/Qwen3.8-27B-4bit` (`model_type: qwen3_5`, with a vision config) loads as the MLXVLM `Qwen35`. An `MLXLanguageModel` with the default loader and `MLXSpeculativeDecoding(registration: .qwen35Text)` then gets a text drafter that is not compatible with its VLM target: every turn (text, reasoning, tools, guided) decodes plainly, silently.

Found while running the guided MTP benchmark of ^9xe94tr: through `testLoad()` it drafted 0 tokens; through `LLMModelFactory.shared.loadContainer` it drafted and ran 74.7 vs 54.1 tok/s.

- [ ] Decide the fix: pick the drafter registration from the loaded target (VLM target -> `.qwen35VLM`), or report the incompatibility (log / availability / error at init), or document that a Qwen 3.5-family text drafter needs the LLM factory.
- [ ] Test: a VLM-loaded Qwen 3.5 target with a text drafter either speculates or reports why it does not.

#mtp #foundation-models #qwen38