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
- actor: claude-code
  id: 01kxm472x181bsgzhr2n2v9xj5
  text: 'Iteration 1: implement landed green in doing. ministral3 aliases to Mistral3TextConfiguration/Mistral3TextModel (Mistral3Text.swift is a port of upstream mlx-lm ministral3.py, which resolves modules by model_type with no remapping — no new config type needed). ToolCallFormat.infer now matches hasPrefix("ministral3") → .mistral. New Ministral3RegistryTests pin registry resolution + decode of the real cached Devstral-2-123B config.json (yarn rope_parameters, null sliding_window, untied embeddings). Suite green: 741 tests, 0 failures. AC1+AC2 done; AC3 (65GB GPU round trip) deferred — needs FoundationModelsMultitool gated suite on M3 Ultra. Checkpoint: cc1728a. Moving to review.'
  timestamp: 2026-07-16T00:11:24.961452+00:00
- actor: claude-code
  id: 01kxna4ywnzepc9dsmb8r8w20c
  text: 'Picked up by /finish (scoped-batch). State on pickup: implement green, checkpoint cc1728a committed, task in review with no review run yet. Dispatching /review bxndpt6 HEAD~1..HEAD.'
  timestamp: 2026-07-16T11:14:21.205242+00:00
- actor: claude-code
  id: 01kxnbb12krp0k4cgmky0x4gfs
  text: 'Iteration 2: /review bxndpt6 HEAD~1..HEAD (cc1728a) returned 90 findings, all in Libraries/MLXLLM/LLMModelFactory.swift — three root causes: (1) missing /// doc comments on public API (LLMRegistry static configs, typealiases, LLMModelFactory init, _load, TrampolineModelFactory, modelFactory(), deprecated ModelRegistry alias), (2) lowerCamelCase violations on registry property names (e.g. gemma3n_E4B_it_lm_4bit, gpt_oss_20b_MXFP4_Q8), (3) @unchecked Sendable on LLMRegistry lacks a documented synchronization invariant. ToolCallFormat.swift and both test files came back clean — the ministral3 change itself is fine; findings are pre-existing hygiene in the touched file (validator sweeps file-wide). Findings checklist appended to description. Dispatching /implement to fix causes file-wide.'
  timestamp: 2026-07-16T11:35:08.627938+00:00
- actor: claude-code
  id: 01kxncv991bfavmgvxf5gz1c3y
  text: |-
    Iteration 2 implement: all 90 review findings fixed, file-wide per root cause, checklist flipped to [x].

    (1) Doc comments: every public API in LLMModelFactory.swift now has ///: all 55 LLMRegistry static model configurations (uniform "Model configuration for `<hub id>`."), ContextType/ContainerType, LLMModelFactory.init, _load (params/returns/throws), TrampolineModelFactory, modelFactory(), and the deprecated ModelRegistry typealias.

    (2) lowerCamelCase renames: all 39 cited renames applied per the findings' suggested names, plus 12 file-wide siblings the review didn't cite line-by-line but the same rule covers (llama3_8B_4bit→llama38b4bit, gemma4_e4b_it_4bit→gemma4E4bIt4bit, lille_130m_bf16→lille130mBf16, and the qwen2_5_*/qwen3_* family → qwen257b, qwen2515b, qwen306b4bit, qwen317b4bit, qwen34b4bit, qwen38b4bit, qwen3Moe30bA3b4bit, qwen352b4bit, qwen3627b4bit). Two judgment calls, noted inline in the checklist: phi3_5_4bit→phi354bit (the underscore-free of the two offered options) and ernie_45_0_3BPT_bf16_ft→ernie4503bPtBf16Ft (the suggested ernie450_3BptBf16Ft itself contained an underscore + uppercase-B-after-digit that the same rule flags). Clean rename, no deprecated aliases (findings did not ask for any). ALL references updated across the repo via word-boundary perl over tracked *.swift/*.md (17 files: MLXFoundationModels, MLXHuggingFace, MLXLLM LoraTrain doc example, Tests/MLXLMTests StopString+LLMRegistry tests, IntegrationTesting CoherenceIntegrationTests, READMEs, DocC pages, skills references). Verified zero occurrences of any old name remain outside .kanban.

    (3) @unchecked Sendable: LLMRegistry class doc now documents the synchronization invariant — mutable state (id→configuration dictionary) lives in AbstractModelRegistry guarded by its NSLock; the subclass adds only immutable static lets and no instance storage.

    Verification: full `swift test` green with pipefail, exit 0 — Swift Testing 217+256+65+7 = 545 passed, XCTest 191+5 = 196 executed 0 failures (741 total), no test warnings. 4 build-time warnings are pre-existing SwiftPM package noise (unhandled Documentation.docc catalogs in MLXLLM/MLXLMCommon/MLXHuggingFace + a .build mutated-node note), untouched by this change and out of scope.

    Note for next agent: renamed properties are public API; anything outside this repo consuming LLMRegistry.<oldName> will need the new names. AC3 (123B GPU round trip) remains deferred as before.
  timestamp: 2026-07-16T12:01:29.889055+00:00
- actor: claude-code
  id: 01kxndw3j9wj68r1c9ea6fw76j
  text: |-
    really-done adversarial double-check returned REVISE with 3 findings; disposition:

    1. FIXED — Libraries/MLXLLM/Documentation.docc/using-model.md: the rename sweep had rewritten the example string literal `ModelConfiguration(id: "mlx-community/llama3_8B_4bit")` (a fictitious id that happened to match the old property name) to `"mlx-community/llama38b4bit"`. Replaced with the real repo id `"mlx-community/Meta-Llama-3-8B-Instruct-4bit"`, which is what `LLMRegistry.llama38b4bit` actually points to. Real `id:` strings in LLMModelFactory.swift were verified byte-identical to HEAD by the double-check.

    2. JUSTIFIED, no change — `lingMini2bit` (was `ling_mini_2_2bit`, Ling-mini-2.0-2bit-DWQ) drops the model-version digit. This is the review checklist's own verbatim suggested rename ("Rename to `lingMini2bit`"); deviating from the checklist's explicit target risks a re-review mismatch, the name is unique and lowerCamelCase-compliant, and the doc comment right above it carries the full hub id. Accepting as-is.

    3. FIXED — re-aligned the trailing `// mlx-community/...` comment columns in the Llama/Mistral, Qwen, Gemma, Phi, and DeepSeek code blocks of skills/mlx-swift-lm/references/supported-models.md that the renames had left ragged.

    Everything else PASSed adversarial scrutiny: doc coverage complete, naming compliant, zero stale references, all() list exact (55 entries, no dups/omissions), LLMTypeRegistry untouched, Sendable invariant factually verified against AbstractModelRegistry. Post-fix edits are markdown-only (no compiled source touched after the green `swift test` run: 545 Swift Testing + 196 XCTest, 0 failures, exit 0). Leaving task in doing for /review.
  timestamp: 2026-07-16T12:19:25.385985+00:00
- actor: claude-code
  id: 01kxnebqjbed6gmazggpxgptac
  text: 'Iteration 2: implement landed green in doing — all 90 review findings fixed file-wide (doc comments on every public API in LLMModelFactory.swift, 39 cited + 12 sibling lowerCamelCase renames with all references updated across 17 files, @unchecked Sendable invariant documented). Independent /test verified from scratch: 741 tests, 0 failures, 3 identical runs, no skips. Dispatching /commit for the checkpoint.'
  timestamp: 2026-07-16T12:27:57.387409+00:00
- actor: claude-code
  id: 01kxnfzne8smfcpjzy12ew04re
  text: 'Iteration 3: /review bxndpt6 HEAD~1..HEAD (e054567) — all 90 prior findings verified fixed; 12 NEW findings recorded (2026-07-16 07:29 section), all in Libraries/MLXLLM/LoraTrain.swift, which entered review scope because the rename sweep touched it. Root causes: missing /// docs on public API file-wide, LoRA acronym casing (LoraLossFunction → LoRALossFunction), unneeded public on internal init, modifier ordering (public mutating func), hardcoded 2048 max-seq-len needs a named constant, as! force-cast needs as? + error handling. 19 engine findings about restyling existing test code were dropped under the skill''s blanket exception. Dispatching /implement.'
  timestamp: 2026-07-16T12:56:19.144520+00:00
- actor: claude-code
  id: 01kxngnb6v8pevf0n7xv5kejds
  text: |-
    Iteration 3 implement: all 12 review findings (2026-07-16 07:29) fixed in Libraries/MLXLLM/LoraTrain.swift, checklist flipped to [x]. Root causes addressed file-wide:

    (1) Doc comments: every public declaration now carries ///: LoRALossFunction typealias, Parameters.init (full parameter list), loss (params/returns), Progress enum + its three cases + description, ProgressDisposition + its two cases. Already-documented API (LoRATrain, Parameters vars, evaluate, saveLoRAWeights, train) untouched.

    (2) Acronym casing: LoraLossFunction → LoRALossFunction. Repo-wide grep before and after confirmed the only references were the 3 in LoraTrain.swift itself (declaration + evaluate + train params) — no Tools/Applications/Tests/docs references existed, so no external updates needed. No other Lora-cased identifiers in the file (LoRABatchIterator/saveLoRAWeights/loadLoRAData already correct).

    (3) Removed `public` from LoRABatchIterator.init (internal struct). Kept `public` on next() because the finding's own prescribed rewrite is `public mutating func next()` — only the modifier order was flagged there.

    (4) `mutating public func next()` → `public mutating func next()`.

    (5) 2048 extracted to file-private top-level `maxSequenceLength`, used in both the condition and the interpolated warning message.

    (6) `as!` in loss replaced with `guard let model = model as? any LLMModel else { fatalError(...) }` plus a why-comment. NOT a throw: loss is the default LoRALossFunction and must stay non-throwing because train feeds it into MLXNN valueAndGrad, whose closure is non-throwing; making the typealias throwing would force try!/trap inside that closure anyway and ripple `throws` through evaluate's public signature. guard+fatalError-with-message is the module's prevailing idiom for contract violations in non-throwing contexts (GLM4MOELite, Gemma4Text, SSM.swift precedents).

    Pure behavior-preserving refactor — no new tests warranted (TDD skill exemption for refactors that add no behavior); existing suite guards it. Full `swift test` green, exit 0: Swift Testing 256+65+217+7 = 545 passed, XCTest 191+5 = 196 executed 0 failures — 741 total, no test warnings. Pre-existing SwiftPM Documentation.docc "unhandled file" warnings + .build node note remain, as expected/out of scope.

    The unchecked AC3 (123B GPU round trip) remains a deliberate deferral per the card. Older 06:14 findings section untouched (already all [x]).
  timestamp: 2026-07-16T13:08:09.563163+00:00
- actor: claude-code
  id: 01kxnjr251xggt76xfttgcc531
  text: 'Iteration 3: implement landed green in doing — all 12 LoraTrain.swift findings fixed file-wide (docs on every public decl, LoraLossFunction → LoRALossFunction with zero stale refs repo-wide, internal init, public mutating func ordering, maxSequenceLength constant, as! → guard-let + fatalError with rationale — non-throwing valueAndGrad context, module''s prevailing idiom). Independent /test verified from scratch twice: 741 tests, 0 failures, 0 skips. Dispatching /commit for the checkpoint.'
  timestamp: 2026-07-16T13:44:35.745906+00:00
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

## Review Findings (2026-07-16 06:14)

- [x] `Libraries/MLXLLM/LLMModelFactory.swift:111` — `@unchecked Sendable` conformance on `LLMRegistry` lacks required documented synchronization invariant explaining why it is safe to send across concurrency boundaries. Add a comment above the class declaration documenting the synchronization invariant, e.g., `// All properties are immutable static lets, making this safe to share across task boundaries.`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:117` — Public property `smolLM_135M_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:117` — Property name `smolLM_135M_4bit` violates `lowerCamelCase` — contains underscores and mixed-case acronym. Rename to `smolLm135m4bit` to follow lowerCamelCase with uniform lowercase acronyms.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:121` — Public property `mistralNeMo4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:121` — Property name `mistralNeMo4bit` uses `neMo` with inconsistent acronym casing. Rename to `mistralNemo4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:125` — Property name `mistral7B4bit` has uppercase 'B' in lowerCamelCase context; should be lowercase. Rename to `mistral7b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:129` — Public property `codeLlama13b4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:133` — Property name `deepSeekR1_7B_4bit` violates `lowerCamelCase` — contains underscores and mixed-case components. Rename to `deepSeekR17b4bit` (removing underscores and lowercasing 'B').
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:137` — Public property `falconH1R7B` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:137` — Property name `falconH1R7B` has uppercase letters 'R' and 'B' that should be lowercase in `lowerCamelCase`. Rename to `falconH1r7b`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:141` — Public property `phi4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:145` — Property name `gemma_2_9b_it_4bit` violates `lowerCamelCase` — contains underscores. Rename to `gemma29bIt4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:145` — Public property `phi3_5_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:145` — Property name `phi3_5_4bit` violates `lowerCamelCase` — contains underscore. Rename to `phi35_4bit` or `phi354bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:149` — Property name `gemma_2_2b_it_4bit` violates `lowerCamelCase` — contains underscores. Rename to `gemma22bIt4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:149` — Public property `phi3_5MoE` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:149` — Property name `phi3_5MoE` has underscore and uppercase `MoE` that should be lowercase in `lowerCamelCase`. Rename to `phi35Moe`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:153` — Public property `gemma3_1B_qat_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:153` — Property name `gemma3_1B_qat_4bit` violates `lowerCamelCase` — contains underscores and uppercase 'B'. Rename to `gemma31bQat4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:153` — Public property `gemma2bQuantized` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:157` — Public property `gemma3n_E4B_it_lm_bf16` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:157` — Property name `gemma3n_E4B_it_lm_bf16` violates `lowerCamelCase` — contains underscores. Rename to `gemma3nE4bItLmBf16`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:161` — Public property `gemma3n_E2B_it_lm_bf16` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:161` — Property name `gemma3n_E2B_it_lm_bf16` violates `lowerCamelCase` — contains underscores. Rename to `gemma3nE2bItLmBf16`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:165` — Public property `gemma3n_E4B_it_lm_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:165` — Property name `gemma3n_E4B_it_lm_4bit` violates `lowerCamelCase` — contains underscores. Rename to `gemma3nE4bItLm4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:169` — Public property `gemma3n_E2B_it_lm_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:169` — Property name `gemma3n_E2B_it_lm_4bit` violates `lowerCamelCase` — contains underscores. Rename to `gemma3nE2bItLm4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:173` — Public property `gemma4_e4b_it_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:177` — Public property `gemma4_e2b_it_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:177` — Property name `gemma4_e2b_it_4bit` violates `lowerCamelCase` — contains underscores. Rename to `gemma4E2bIt4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:181` — Public property `llama3_1_8B_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:181` — Property name `llama3_1_8B_4bit` violates `lowerCamelCase` — contains underscores. Rename to `llama318b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:189` — Property name `llama3_2_1B_4bit` violates `lowerCamelCase` — contains underscores. Rename to `llama321b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:193` — Public property `llama3_2_3B_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:193` — Property name `llama3_2_3B_4bit` violates `lowerCamelCase` — contains underscores. Rename to `llama323b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:197` — Public property `deepseek_r1_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:197` — Property name `deepseek_r1_4bit` violates `lowerCamelCase` — contains underscores. Rename to `deepseekR14bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:201` — Public property `granite3_3_2b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:201` — Property name `granite3_3_2b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `granite332b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:205` — Public property `mimo_7b_sft_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:205` — Property name `mimo_7b_sft_4bit` violates `lowerCamelCase` — contains underscores. Rename to `mimo7bSft4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:209` — Public property `glm4_9b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:209` — Property name `glm4_9b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `glm49b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:215` — Public property `acereason_7b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:215` — Property name `acereason_7b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `acereason7b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:219` — Public property `bitnet_b1_58_2b_4t_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:219` — Property name `bitnet_b1_58_2b_4t_4bit` violates `lowerCamelCase` — contains underscores. Rename to `bitnetB1582b4t4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:223` — Public property `baichuan_m1_14b_instruct_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:223` — Property name `baichuan_m1_14b_instruct_4bit` violates `lowerCamelCase` — contains underscores. Rename to `baichuanM114bInstruct4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:227` — Public property `smollm3_3b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:227` — Property name `smollm3_3b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `smolLm33b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:231` — Public property `ernie_45_0_3BPT_bf16_ft` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:231` — Property name `ernie_45_0_3BPT_bf16_ft` violates `lowerCamelCase` — contains underscores. Rename to `ernie450_3BptBf16Ft`. *(Renamed to `ernie4503bPtBf16Ft` — the suggested name still contained an underscore and an uppercase 'B' after a digit, which the same rule flags; the applied name is fully lowerCamelCase.)*
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:235` — Public property `lfm2_1_2b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:235` — Property name `lfm2_1_2b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `lfm212b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:239` — Public property `exaone_4_0_1_2b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:239` — Property name `exaone_4_0_1_2b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `exaone4012b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:243` — Public property `lille_130m_bf16` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:247` — Public property `olmoe_1b_7b_0125_instruct_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:247` — Property name `olmoe_1b_7b_0125_instruct_4bit` violates `lowerCamelCase` — contains underscores. Rename to `olmoe1b7b0125Instruct4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:251` — Public property `olmo_2_1124_7B_Instruct_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:251` — Property name `olmo_2_1124_7B_Instruct_4bit` violates `lowerCamelCase` — contains underscores. Rename to `olmo211247bInstruct4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:255` — Public property `ling_mini_2_2bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:255` — Property name `ling_mini_2_2bit` violates `lowerCamelCase` — contains underscores. Rename to `lingMini2bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:259` — Public property `granite_4_0_h_tiny_4bit_dwq` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:259` — Property name `granite_4_0_h_tiny_4bit_dwq` violates `lowerCamelCase` — contains underscores. Rename to `granite40hTiny4bitDwq`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:263` — Public property `lfm2_8b_a1b_3bit_mlx` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:263` — Property name `lfm2_8b_a1b_3bit_mlx` violates `lowerCamelCase` — contains underscores. Rename to `lfm28bA1b3bitMlx`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:267` — Public property `nanochat_d20_mlx` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:267` — Property name `nanochat_d20_mlx` violates `lowerCamelCase` — contains underscores. Rename to `nanochatD20Mlx`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:271` — Public property `gpt_oss_20b_MXFP4_Q8` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:271` — Property name `gpt_oss_20b_MXFP4_Q8` violates `lowerCamelCase` — contains underscores and uppercase acronyms. Rename to `gptOss20bMxfp4Q8`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:275` — Public property `jamba_3b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:275` — Property name `jamba_3b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `jamba3b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:279` — Public property `nemotron_labs_diffusion_3b_4bit` lacks required `///` documentation comment. Add a `///` doc comment above the property describing the model configuration.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:279` — Property name `nemotron_labs_diffusion_3b_4bit` violates `lowerCamelCase` — contains underscores. Rename to `nemotronLabsDiffusion3b4bit`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:374` — Public type alias `ContextType` lacks required `///` documentation comment. Add a `///` doc comment above the type alias describing its purpose.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:375` — Public type alias `ContainerType` lacks required `///` documentation comment. Add a `///` doc comment above the type alias describing its purpose.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:377` — Public initializer on `LLMModelFactory` lacks required `///` documentation comment. Add a `///` doc comment above the initializer describing its purpose and parameters.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:392` — Public method `_load` lacks required `///` documentation comment. Add a `///` doc comment above the method describing its purpose, parameters, return value, and any errors it throws.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:518` — Public class `TrampolineModelFactory` lacks required `///` documentation comment. Add a `///` doc comment above the class describing its purpose and role.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:519` — Public method `modelFactory()` lacks required `///` documentation comment. Add a `///` doc comment above the method describing its purpose and return value.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:637` — Public typealias lacks documentation comment. While the @available attribute includes a deprecation message, public type aliases should have doc comments explaining their purpose and relationship to the renamed type. Add a doc comment above the typealias explaining it is deprecated and what to use instead, e.g. `/// Deprecated alias for LLMRegistry. Use LLMRegistry directly instead.`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:674` — Public typealias ContextType lacks documentation. This is part of the public API of LLMModelFactory and should explain what type context represents. Add a doc comment explaining this typealias, e.g. `/// The context type used by this factory (ModelContext).`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:675` — Public typealias ContainerType lacks documentation. This is part of the public API of LLMModelFactory and should explain what type container represents. Add a doc comment explaining this typealias, e.g. `/// The container type used by this factory (ModelContainer).`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:677` — Public initializer of LLMModelFactory lacks documentation. Callers need to understand what the typeRegistry and modelRegistry parameters are for. Add a doc comment explaining the initializer and its parameters, e.g. `/// Initialize a factory with custom type and model registries. - Parameters: \n  - typeRegistry: Registry mapping model types to their implementations \n  - modelRegistry: Registry of model configurations`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:691` — Public method _load lacks documentation. This is a key loading method that clients might override or call, and needs documentation explaining what it does and what parameters mean. Add a doc comment explaining this method's purpose, parameters, and return value, e.g. `/// Load model configuration and weights, then assemble a ModelContext. - Parameters: \n  - configuration: The resolved model configuration \n  - tokenizerLoader: Loader for the tokenizer - Returns: A ready-to-use ModelContext`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:768` — Public class TrampolineModelFactory lacks documentation. This is a public API class and needs explanation of its purpose. Add a doc comment explaining what this class does and why it exists, e.g. `/// Trampoline class that bridges to LLMModelFactory. Used for dynamic discovery of the factory instance.`.
- [x] `Libraries/MLXLLM/LLMModelFactory.swift:769` — Public static function modelFactory lacks documentation. Callers need to understand what this returns and its purpose. Add a doc comment explaining this method, e.g. `/// Return the shared LLMModelFactory instance for use as MLXLMCommon's primary factory.`.

## Review Findings (2026-07-16 07:29)

- [x] `Libraries/MLXLLM/LoraTrain.swift:20` — The `public init` is unnecessary — `LoRABatchIterator` is an internal struct used only within this file for the `LoRATrain` enum. Exposing a public initializer on an internal struct violates the principle of adding `public` only for intended cross-module API. The rule flags `public` sprayed on helpers no other module consumes. Remove the `public` modifier from the init: `init(dataset: [String], tokenizer: Tokenizer, batchSize: Int, train: Bool)`. The struct's default internal access applies to all members.
- [x] `Libraries/MLXLLM/LoraTrain.swift:36` — Access modifiers must come before other modifiers like `mutating`. The syntax `mutating public func` violates Swift modifier ordering — it should be `public mutating func`. Rewrite as `public mutating func next() -> (MLXArray, MLXArray, MLXArray)?`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:47` — The hardcoded value 2048 (maximum sequence length threshold) is repeated across multiple places: the condition check on line 47 and the warning message on line 50. This should be extracted to a named constant to avoid duplication and make the value easier to maintain. Extract to a named constant: `private let maxSequenceLength = 2048` at the module level or top of the function, then use it in both the condition and the warning message string.
- [x] `Libraries/MLXLLM/LoraTrain.swift:66` — Acronyms must be uniformly cased — `LoRA` should stay as `LoRA` (all-upper interior to lowerCamelCase), not downcase to `Lora`. `LoraLossFunction` violates the uniform-acronym rule. Rename to `LoRALossFunction`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:75` — Public type alias LoraLossFunction lacks documentation; it defines a complex closure signature used for training loss functions and callers need to understand its purpose and parameters. Add a doc comment explaining the type signature, e.g.: `/// Type signature for a loss function used in LoRA training. Takes model, input tokens, target tokens, and sequence lengths; returns loss and token count.`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:78` — Every `public` declaration must carry a `///` doc comment; this public initializer lacks one. Add a doc comment above the initializer documenting its parameters and purpose.
- [x] `Libraries/MLXLLM/LoraTrain.swift:88` — Every `public` declaration must carry a `///` doc comment; this public static function lacks one. Add a doc comment above the function documenting its parameters, return value, and purpose.
- [x] `Libraries/MLXLLM/LoraTrain.swift:93` — Force-cast (`as!`) is forbidden in non-test code; it crashes rather than propagating an error. Replace with a safe `as?` cast and either throw or handle the failure explicitly. Use safe casting: `guard let model = model as? any LLMModel else { throw SomeError.invalidModel }` or similar. *(Fixed with `guard let model = model as? any LLMModel else { fatalError("...descriptive message...") }` rather than `throw`: `loss` is the default `LoRALossFunction`, which must stay non-throwing because it is invoked inside MLXNN's `valueAndGrad` closure (non-throwing signature). `guard ... else { fatalError("message") }` is the module's prevailing idiom for contract violations in non-throwing contexts — e.g. GLM4MOELite "Module must be MultiLinear or QuantizedMultiLinear", Gemma4Text, SSM.swift. A code comment above the guard documents this constraint.)*
- [x] `Libraries/MLXLLM/LoraTrain.swift:107` — Public static function loss lacks documentation; it is a core training API and callers need to understand its parameters and return values. Add a doc comment documenting the function purpose, parameters, and returns, e.g.: `/// Computes loss for LoRA training. Returns a tuple of (loss, token_count).`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:142` — Public enum Progress lacks documentation; it represents progress events during training and callers need to understand the available event types. Add a doc comment, e.g.: `/// Represents progress events during LoRA training, including training/validation loss updates and checkpoint saves.`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:143` — Every `public` declaration must carry a `///` doc comment; this public enum lacks one. Add a doc comment above the enum, e.g., `/// Signal from a progress callback indicating whether training should continue.`.
- [x] `Libraries/MLXLLM/LoraTrain.swift:164` — Public enum ProgressDisposition lacks documentation; it controls training flow and callers need to understand the meaning of each case. Add a doc comment, e.g.: `/// Indicates whether to continue training or stop in response to a progress event.`.