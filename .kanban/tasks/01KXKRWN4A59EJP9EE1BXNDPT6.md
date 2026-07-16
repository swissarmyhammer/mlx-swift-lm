---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxm2sxr28p5b0twc5xyhszvw
  text: Picked up by /finish (scoped-batch, iteration 1). Dispatching /implement.
  timestamp: 2026-07-15T23:46:45.122607+00:00
- actor: claude-code
  id: 01kxm30mhnxtbbvc4az8x8131b
  text: 'Research done. Key finding: Libraries/MLXLLM/Models/Mistral3Text.swift is already a port of upstream mlx-lm''s ministral3.py (header comment says so; config default modelType is "ministral3"). Upstream mlx-lm resolves model_type -> module by name (ministral3.py is its own module, no remapping), so aliasing "ministral3" to Mistral3TextConfiguration/Mistral3TextModel in LLMTypeRegistry is exactly the upstream architecture — no new config needed. Verified the cached Devstral-2-123B config.json: yarn rope_parameters (rope_type "yarn", llama_4_scaling_beta 0.0, rope_theta inside rope_parameters), no layer_types, sliding_window null — all paths Mistral3Text.swift + initializeRope (yarn supported) already handle. Two production seams to change: (1) LLMTypeRegistry entry "ministral3", (2) ToolCallFormat.infer prefix match for "ministral3" (does NOT share the "mistral3" prefix). Structural-tag seam needs no extra wiring: infer -> .mistral covers it since all formats resolve through the format, not the type string. Proceeding TDD: RED tests for registry resolution + infer + real-config decode.'
  timestamp: 2026-07-15T23:50:25.077412+00:00
- actor: claude-code
  id: 01kxm3gww8swz19sdw8t1w5t43
  text: |-
    Implementation landed (TDD, RED observed then GREEN):

    - Libraries/MLXLLM/LLMModelFactory.swift: added LLMTypeRegistry entry "ministral3" -> Mistral3TextConfiguration/Mistral3TextModel (Mistral3Text.swift is the port of mlx-lm's ministral3.py, so this mirrors upstream exactly).
    - Libraries/MLXLMCommon/Tool/ToolCallFormat.swift: infer now matches hasPrefix("ministral3") alongside hasPrefix("mistral3") -> .mistral (the extra "ni" breaks the old prefix match).
    - Tests/MLXLMTests/Ministral3RegistryTests.swift (new): registry resolves "ministral3" to Mistral3TextModel using a tiny config mirroring the real Devstral-2-123B config.json shape (yarn rope_parameters carrying rope_theta + llama_4_scaling_beta, null sliding_window, no layer_types, untied embeddings); plus a decode pin of that shape into Mistral3TextConfiguration.
    - Tests/MLXLMTests/ToolTests.swift: testToolCallFormatInference asserts infer("ministral3"/"Ministral3") == .mistral.

    No other seams need wiring: message generator keys off the model class (shared with mistral3), the FoundationModels structural tag/tool-call rendering key off ToolCallFormat == .mistral (cf4fa81/cd52383), and initializeRope already supports yarn.

    Full `swift test` green: 256+65+217+7 Swift Testing tests passed, all XCTest bundles 0 failures, no warnings.

    Deferred to GPU verification (acceptance criterion 3): loading the real 123B weights (~65GB, cached at ~/.cache/huggingface/hub/models--mlx-community--Devstral-2-123B-Instruct-2512-4bit) and one end-to-end tool-call round trip, e.g. via FoundationModelsMultitool's gated suite on the M3 Ultra.
  timestamp: 2026-07-15T23:59:17.896052+00:00
position_column: doing
position_ordinal: '8180'
title: Support model_type "ministral3" (Devstral-2-123B) in LLMModelFactory
---
## What

`mlx-community/Devstral-2-123B-Instruct-2512-4bit` (Mistral's large Devstral 2, 14 shards ~65GB at 4-bit) fails to load with `.unsupportedModelType("ministral3")` — its `config.json` declares `model_type: "ministral3"`, which has no entry in `LLMModelFactory`'s registry. The 24B `Devstral-Small-2` declares `mistral3` and loads fine (via `Mistral3TextModel`), so `ministral3` is likely the same or a close variant of the Mistral3 architecture — check the upstream `transformers`/`mlx-lm` implementations for whether it can simply alias to `Mistral3TextConfiguration`/`Mistral3TextModel` or needs its own config (e.g. different attention/rope details).

Reproduced 2026-07-15 via FoundationModelsMultitool's gated suite on M3 Ultra; weights fully cached locally (`~/.cache/huggingface/hub/models--mlx-community--Devstral-2-123B-Instruct-2512-4bit`), so verification needs no re-download.

Also check `ToolCallFormat.infer` — it matches the `mistral3` prefix; `ministral3` does NOT start with "mistral3" (extra "ni"), so it needs its own prefix match to get the Mistral tool-call format/parser too.

## Acceptance Criteria

- [x] `model_type: "ministral3"` resolves to a working model implementation in `LLMModelFactory`.
- [x] `ToolCallFormat.infer` maps `ministral3` to the Mistral tool-call format.
- [ ] Devstral-2-123B-Instruct-2512-4bit loads and completes at least one tool-call round trip end to end. *(Deferred to GPU verification: requires ~65GB of inference on the cached weights — see comments. Registry resolution, config decoding against the real config.json shape, and format inference are unit-tested.)*
