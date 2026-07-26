---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kydbafw2dgjet7nxhdgfmgjr
  text: |-
    Read GLM4.swift, GLM4MOELite.swift, and the reference GLM4MOE.swift. Confirmed GLM4MOE.swift's sanitize() ALSO strips `lm_head.weight` when tied (identical to GLM4.swift's existing sanitize) -- so per the "mirror GLM4MOE.swift" mandate, GLM4.swift's sanitize step should be KEPT as-is, not removed (it's belt-and-suspenders against checkpoints that redundantly include lm_head.weight despite being tied -- Load.swift passes verify:[.all] which includes .noUnusedKeys, so an unconsumed key would throw UpdateError.unhandledKeys). Only GLM4MOELite.swift's sanitize needs the strip *added* (currently completely inert).

    Wrote failing tests first (TDD) in Tests/MLXLMTests/GLM4LmHeadTiedLoadTests.swift: tied+untied load tests for both GLM4Model and GLM4MoELiteModel (following Gemma4KVSharedLoadTests.swift template: dense/untied sibling supplies weights, tied variant gets checkpoint with lm_head.weight stripped), plus sanitize drop/keep tests for GLM4MoELiteModel (new behavior).

    Confirmed RED: `swift test --filter GLM4LmHeadTiedLoadTests` shows exactly the predicted failures -- `UpdateError.keyNotFound(path: ["lm_head", "weight"], modules: ["GLM4Model"/"GLM4MoELiteModel", "Linear"])` for both tied-checkpoint tests, and sanitize-drops-when-tied fails for GLM4MoELite (currently keeps the key). Untied tests and GLM4MoELite sanitize-keeps test already pass (expected, since untied path already worked and MoELite currently never strips).

    Now implementing the guarded-Optional fix in both files.
  timestamp: 2026-07-25T19:16:37.378081+00:00
- actor: claude-code
  id: 01kydbkpas0p3r1g1fyff4d1s6
  text: |-
    Implementation complete. Changes:

    - Libraries/MLXLLM/Models/GLM4.swift: `lmHead` changed to `Linear?`, guarded creation `if !args.tieWordEmbeddings`, `callAsFunction` falls back to `model.embedTokens.asLinear(out)` when nil. Kept the existing sanitize() strip of `lm_head.weight` when tied unchanged -- verified GLM4MOE.swift's own sanitize does the identical strip, so this matches the reference pattern exactly (and guards against checkpoints that redundantly include lm_head.weight despite being tied, given Load.swift's verify:[.all] includes .noUnusedKeys).
    - Libraries/MLXLLM/Models/GLM4MOELite.swift: same `lmHead: Linear?` guarded pattern + fallback. Added the previously-missing `if configuration.tieWordEmbeddings { sanitized["lm_head.weight"] = nil }` to sanitize(), wiring up the formerly-inert `tieWordEmbeddings` field, mirroring GLM4MOE.swift.
    - GLM4MOE.swift untouched (read-only reference).

    Tests added: Tests/MLXLMTests/GLM4LmHeadTiedLoadTests.swift -- 6 tests (tied+untied load tests for both GLM4Model and GLM4MoELiteModel following the Gemma4KVSharedLoadTests.swift template, plus sanitize drop/keep tests for GLM4MoELiteModel since that's new behavior). Watched all 3 relevant tests fail RED first with the exact predicted `UpdateError.keyNotFound(path: ["lm_head","weight"], ...)` and sanitize-not-dropped failure, then confirmed GREEN after the fix.

    Verification: `swift test --filter MLXLMTests` -- 353 tests in 25 suites (Swift Testing) plus all XCTestCase suites, 0 failures. `swift build` clean, 0 warnings/errors. MCP diagnostics check on working tree: 0 errors, 0 warnings.

    Leaving task in `doing` per process (not moving to review).
  timestamp: 2026-07-25T19:21:38.905790+00:00
- actor: claude-code
  id: 01kydc4epdjf4kjdyp4p1c5q7w
  text: |-
    Adversarial double-check (double-check agent): PASS. Independently verified diff matches description exactly, GLM4MOE.swift untouched (zero diff lines), sanitize equivalence claim confirmed by reading both files' sanitize() directly, `.all` VerifyUpdate semantics confirmed against the vendored MLXNN source (includes .noUnusedKeys), untied test confirmed as a genuine ground-truth forward-pass comparison (not a weak check), no external `.lmHead` call sites outside the two files + new test, and fresh build/test/diagnostics runs all green. One noted (acceptable, pre-existing-pattern) gap: no test wires sanitize()'s output directly into update() for a checkpoint that redundantly includes lm_head.weight while tied -- mirrors the same limitation in the cited Gemma4KVSharedLoadTests template, not a regression introduced here.

    Task remains in `doing`, ready for /review.
  timestamp: 2026-07-25T19:30:48.141020+00:00
- actor: claude-code
  id: 01kydfhydchb8dys3a8jpm053j
  text: |-
    ## Review disposition (review sha HEAD~1..HEAD, commit fed151f)

    `review sha HEAD~1..HEAD` returned 55 findings (counts: findings 55, confirmed 55, refuted 17, attempted 14, failed 0), all documentation/duplication/complexity style findings — no correctness findings. Cross-checked each against the actual diff (`git diff HEAD~1..HEAD` for the two touched files) to classify:

    **In-scope (about the new lm_head Optional logic / fallback path) — fixed in code, 10 findings:**
    - GLM4.swift:154 — `GLM4Model.callAsFunction` duplicated the new `if let lmHead { ... } else { model.embedTokens.asLinear(out) }` fallback verbatim with `GLM4MoELiteModel.callAsFunction`. Fixed by extracting a shared `applyLMHead(_:embedTokens:_:)` helper (added in GLM4.swift, used from both files — both are in the `MLXLLM` module so no import needed).
    - GLM4.swift:181, :183 — missing doc comment on `GLM4Model.callAsFunction`, one explicitly citing "lm_head application or embedding fallback for tied embeddings". Added a doc comment covering the tied/untied fallback and cache usage.
    - GLM4MOELite.swift:502 — missing doc on the now-Optional `lmHead` property, explicitly citing "its optional nature when word embeddings are tied". Added a doc comment.
    - GLM4MOELite.swift:508, :537 — missing doc on `init` (which now contains the new `if !args.tieWordEmbeddings` guarded creation). Added a doc comment.
    - GLM4MOELite.swift:516, :547 — missing doc on `callAsFunction` (now contains the new fallback + uses the shared helper). Added a doc comment.
    - GLM4MOELite.swift:522, :553 — missing doc on `sanitize` (now contains the new tied-embedding `lm_head.weight` strip block, in addition to pre-existing expert-stacking/kv_b_proj logic). Added a doc comment covering all of it.

    Verified: `swift build --target MLXLLM` clean; `swift test --filter MLXLMTests.GLM4LmHeadTiedLoadTests` — 6/6 pass; re-running `review file` on both files afterward shows zero remaining findings mentioning lm_head/tied-embeddings/fallback/duplication between the two files — the fix eliminated the cited cause, not just the cited line.

    **Pre-existing debt — dispositioned as "skip, pre-existing debt" per established precedent (^mv9aq7w/^xgvth41/^wz8y8qq/^ayw1xee/^9a2aw98/^m1f02cw/^akvqxxz), 45 findings:**

    All remaining findings sit 100% on lines this diff never touched, confirmed line-by-line against `git diff HEAD~1..HEAD`:
    - Missing `///` doc comments and missing-`final` on `GLM4DecoderLayer`/`GLM4ModelInner`/`GLM4MoELiteModelInner`/`GLM4Model`/`GLM4MoELiteModel` class declarations, and on unrelated properties (`vocabularySize`, `kvHeads`, `model`, `modelType`) — none of these declarations or their surrounding lines were touched by fed151f.
    - `GLM4ModelInner.callAsFunction` vs `GLM4MoELiteModelInner.callAsFunction` duplication (embed+mask+loop+norm, GLM4.swift:143) — pre-existing, unrelated to lm_head; neither Inner class's `callAsFunction` was touched by this diff.
    - Nested-loop complexity findings (GLM4MOELite.swift:485 expert-stacking loop, :534 `numMptLayers` MTP-layer filter) — both are pre-existing `sanitize` logic below the new 3-line tied-strip block this diff added; the diff only inserted lines above them, it didn't touch or add this nesting.
    - `GLM4.swift` sanitize doc-comment findings (:188, :190) — GLM4.swift's `sanitize` body (the `tie_word_embeddings` strip) already existed before fed151f and was not modified by it (only the class's `lmHead` property/init/callAsFunction changed); the missing-doc gap predates this task.
    - `GLM4Configuration`/`GLM4MoELiteConfiguration` struct docs, `CodingKeys`, and `init(from:)` Codable initializer docs — neither struct's body was touched; `tieWordEmbeddings` decoding already existed pre-diff.
    - `loraLayers` extension doc findings — untouched by this diff.

    None of these 45 are about the new Optional-`lmHead`/fallback/sanitize-tie-strip logic or the new test file; each is either in a different class/method entirely, or in a part of `sanitize` that predates fed151f. Flagging per the established precedent so this disposition is auditable, same as the six prior sessions cited above.

    Commit for the fix work above is pending (uncommitted in the working tree as of this review pass) — code changes: `Libraries/MLXLLM/Models/GLM4.swift` (added `applyLMHead` helper + doc comment), `Libraries/MLXLLM/Models/GLM4MOELite.swift` (doc comments on `lmHead`/`init`/`callAsFunction`/`sanitize`, `callAsFunction` now calls `applyLMHead`).
  timestamp: 2026-07-25T20:30:35.948861+00:00
position_column: done
position_ordinal: db80
title: 'Fix GLM4.swift/GLM4MOELite.swift: unconditional lm_head crashes on genuinely tied checkpoints'
---
## What

Discovered while researching task `01KYD40FNPTNHBJXM4RPC9W4RD` (tied lm_head quantization gap — closed not-applicable). Two architectures diverge from this codebase's own documented tied-embedding convention and will crash at load time for a genuinely `tie_word_embeddings: true` checkpoint that follows the standard convention (HF/mlx_lm omit the tied weight from the safetensors entirely).

`Libraries/MLXLLM/Models/GLM4.swift`:
- `@ModuleInfo(key: "lm_head") var lmHead: Linear` — non-optional, created unconditionally in `init`, `callAsFunction` always calls `lmHead(out)` with no `embedTokens.asLinear(out)` fallback.
- `sanitize(weights:)` actively strips `lm_head.weight` from the loaded dict whenever `configuration.tieWordEmbeddings == true`.
- Since `Libraries/MLXLMCommon/Load.swift`'s `loadWeights` calls `model.update(parameters:, verify: [.all])`, which requires every model parameter path to have a value (`.allModelKeysSet`), a tied GLM4 checkpoint will *always* throw `UpdateError.keyNotFound` for `lm_head.weight` — guaranteed crash, not a graceful fallback.

`Libraries/MLXLLM/Models/GLM4MOELite.swift`:
- Same non-optional, unconditionally-created `lmHead: Linear`.
- `tieWordEmbeddings` is decoded from JSON but never referenced anywhere else in the file (confirmed by grep) — no sanitize special-casing, no guarded creation.
- For a checkpoint that follows the standard HF/mlx_lm convention of omitting the tied weight when `tie_word_embeddings: true`, this will also throw `UpdateError.keyNotFound` for `lm_head.weight` at load time.

Both diverge from the correct, established pattern used by ~30 other architectures in this codebase (documented in `Libraries/MLXLMCommon/Documentation.docc/porting.md`, and correctly implemented in the *sibling* file `Libraries/MLXLLM/Models/GLM4MOE.swift` — the non-lite MoE variant): `lmHead` as `Linear?`, created only `if !args.tieWordEmbeddings`, with `callAsFunction` falling back to `embedTokens.asLinear(out)` when `lmHead` is nil.

## Why this matters

Any real checkpoint for these two architectures published with `tie_word_embeddings: true` (following the standard convention) is currently unloadable — a hard crash, not a silent correctness issue. Confirmed this isn't a Swift-specific bug in one case: fetched upstream `ml-explore/mlx-lm`'s current `mlx_lm/models/glm4_moe.py` and its `Model` class has the identical unconditional-`lm_head`/inert-`tie_word_embeddings` shape — this Swift file is a faithful port of an equally-inert upstream reference. GLM4.swift's active `sanitize` stripping of `lm_head.weight` when tied appears to be Swift-specific (not present upstream) and makes the crash unconditional rather than checkpoint-dependent.

## Suggested fix

Bring both files in line with the established convention (mirror `GLM4MOE.swift`'s existing correct implementation):
- Change `lmHead` to `Linear?`.
- Only instantiate it `if !args.tieWordEmbeddings`.
- `callAsFunction`: `if let lmHead { lmHead(out) } else { model.embedTokens.asLinear(out) }` (verify exact fallback receiver path against `GLM4MOE.swift`'s implementation).
- For GLM4.swift: remove (or adjust) the `sanitize` step that deletes `lm_head.weight` when tied, since with the guarded-Optional pattern there's no `lmHead` module for that key to conflict with, and `model.update`'s `.allModelKeysSet` check won't require it.
- For GLM4MOELite.swift: wire `tieWordEmbeddings` into the same guarded-creation/fallback pattern (currently completely inert).

## Tests

- Follow the existing pattern in `Tests/MLXLMTests/Gemma4KVSharedLoadTests.swift` (builds a synthetic checkpoint dict missing certain keys, calls `model.update(parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])`, asserts throws/doesn't-throw `UpdateError.keyNotFound`) — this is the exact regression-test template for this bug.
- Also follow `Tests/MLXLMTests/Gemma4AssistantDraftModelTests.swift`'s `testSanitizeDropsLmHeadWhenTied`/`testSanitizeKeepsLmHeadWhenNotTied` convention if `sanitize` behavior changes.
- Add tied + untied cases for both GLM4Model and GLM4MoELiteModel.
- `swift test --filter MLXLMTests` must stay green.

## Acceptance Criteria

- [ ] GLM4.swift and GLM4MOELite.swift both use the guarded-Optional `lmHead: Linear?` pattern matching `GLM4MOE.swift`
- [ ] A synthetic-weights test proves a tied checkpoint (omitting `lm_head.weight`) loads successfully for both models (no `UpdateError.keyNotFound`)
- [ ] A synthetic-weights test proves an untied checkpoint (providing `lm_head.weight`) still loads and uses the separate `lm_head` correctly
- [ ] Existing GLM4/GLM4MoELite tests remain green, no regressions

## Review Findings (2026-07-25 14:36)

- [ ] `Libraries/MLXLLM/Models/GLM4.swift:95` — Public class `GLM4ModelInner` is missing a documentation comment. Every public declaration must carry a `///` doc comment stating its purpose. Add a documentation comment above the class definition describing its role as the inner transformer model for GLM4.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:95` — Public class `GLM4ModelInner` should be marked `final` if it is not designed for subclassing. Non-final classes allow external subclassing, which is likely unintended here. Change to `public final class GLM4ModelInner: Module` to prevent unintended subclassing.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:101` — Public initializer inside `GLM4ModelInner` is missing a documentation comment. Every public declaration must carry a `///` doc comment. Add a documentation comment describing the initialization of the inner model with its configuration.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:114` — Public function `callAsFunction` inside `GLM4ModelInner` is missing a documentation comment. Add a documentation comment describing the forward pass of the inner model.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:129` — Public class GLM4ModelInner lacks documentation explaining its purpose and role in the GLM4 architecture. Add documentation comment explaining that GLM4ModelInner encapsulates the core model layers and how it fits within GLM4Model.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:135` — Public initializer of GLM4ModelInner lacks documentation of construction behavior. Add documentation explaining what GLM4Configuration parameters are used and how the initializer sets up embeddings and decoder layers.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:143` — GLM4ModelInner.callAsFunction is verbatim identical to GLM4MoELiteModelInner.callAsFunction. Both methods embed tokens, create an attention mask, iterate through layers, and return the normalized result — this 9-line sequence is duplicated across two inner model classes and will drift if one is updated without the other. Extract into a protocol extension on a shared base protocol (e.g., add a default implementation of callAsFunction to a protocol that requires embedTokens, layers, and norm as protocol requirements). Alternatively, create a protocol-default implementation in a common module (MLXLMCommon) that both classes can adopt.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:148` — Public method callAsFunction in GLM4ModelInner lacks documentation of forward pass behavior. Add documentation explaining the forward pass through embeddings and decoder layers, and what cache parameter does.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:154` — GLM4Model.callAsFunction is verbatim identical to GLM4MoELiteModel.callAsFunction in the sibling file. This 7-line method is copy-pasted between two model classes and will drift if logic changes are applied to only one. Extract the shared callAsFunction logic into a protocol extension on LLMModel or a shared helper function. Both classes conform to LLMModel and have identical lmHead: Linear? properties and model: *ModelInner properties, making a protocol-based extraction feasible. Pattern: define a default implementation in a protocol extension that calls self.model(...), checks if let self.lmHead, and falls back to self.model.embedTokens.asLinear(out).
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:159` — Public class `GLM4Model` is missing a documentation comment. Every public declaration must carry a `///` doc comment. Add a documentation comment describing the GLM4 language model class.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:159` — Public class `GLM4Model` should be marked `final` if not designed for subclassing. Change to `public final class GLM4Model: Module, LLMModel, KVCacheDimensionProvider`.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:161` — Public class GLM4Model lacks documentation as the primary entry point for GLM4 language models. Add documentation explaining that GLM4Model is the main interface for using GLM4 models and describing its conformances.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:161` — Public property `vocabularySize` is missing a documentation comment. Every public declaration must carry a `///` doc comment. Add a documentation comment describing what this property represents.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:162` — Public property vocabularySize in GLM4Model lacks documentation explaining what it represents. Add documentation explaining that vocabularySize is the vocabulary size of the model.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:162` — Public property `kvHeads` is missing a documentation comment. Add a documentation comment describing the key-value heads per layer.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:163` — Public property kvHeads in GLM4Model lacks documentation explaining its purpose. Add documentation explaining that kvHeads contains the number of key-value heads per layer for attention computation.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:164` — Public property `model` is missing a documentation comment. Add a documentation comment describing the inner model instance.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:165` — Public property model in GLM4Model lacks documentation explaining its role. Add documentation explaining that model is the core GLM4ModelInner instance that contains layers and embeddings.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:181` — Public function `callAsFunction` is missing a documentation comment. Add a documentation comment describing the forward pass, inputs, and return value.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:183` — Public method callAsFunction in GLM4Model lacks documentation of inference behavior. Add documentation explaining the forward pass including lm_head application or embedding fallback for tied embeddings, and cache usage.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:188` — Public function `sanitize` is missing a documentation comment. Add a documentation comment explaining that this function removes weights for tied embeddings.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:190` — Public method sanitize in GLM4Model lacks documentation of its weight processing logic. Add documentation explaining that sanitize removes lm_head weights when tie_word_embeddings is true to match checkpoint structure.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:197` — Public struct `GLM4Configuration` is missing a documentation comment. Add a documentation comment describing the configuration parameters for GLM4.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:201` — Public struct GLM4Configuration lacks documentation of its purpose and usage. Add documentation explaining that GLM4Configuration holds hyperparameters for GLM4 models and is loaded from model config files.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:217` — Public Codable initializer `init(from:)` is missing a documentation comment. Add a documentation comment for the Codable decoder initializer.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:248` — Public initializer for GLM4Configuration (init from decoder) lacks documentation of deserialization. Add documentation explaining that this is a Codable deserializer for loading GLM4Configuration from JSON format config files.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:254` — Public property `loraLayers` is missing a documentation comment. Add a documentation comment describing the layers available for LoRA fine-tuning.
- [ ] `Libraries/MLXLLM/Models/GLM4.swift:271` — Public property loraLayers in LoRA extension lacks documentation of its purpose. Add documentation explaining that loraLayers returns the decoder layers that can be fine-tuned with LoRA.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:469` — Public class `GLM4MoELiteModelInner` is missing a documentation comment. Add a documentation comment describing the inner transformer model for GLM4-MoE-Lite.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:469` — Public class `GLM4MoELiteModelInner` should be marked `final` if not designed for subclassing. Change to `public final class GLM4MoELiteModelInner: Module`.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:485` — Condition nested 4 levels deep in triply-nested loops (for l, for n, for k, then if), exceeding the 3-level threshold. Multiple closure bodies inside the condition add further cognitive load. Extract the innermost logic into a named helper function. For example, create a function like `updateExpertWeights(prefix:n:k:)` that encapsulates the triply-nested loop logic, reducing nesting to 1-2 levels.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:493` — Public class `GLM4MoELiteModel` is missing a documentation comment. Add a documentation comment describing the GLM4-MoE-Lite language model.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:493` — Public class `GLM4MoELiteModel` should be marked `final` if not designed for subclassing. Change to `public final class GLM4MoELiteModel: Module, LLMModel, KVCacheDimensionProvider`.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:495` — Public property `vocabularySize` is missing a documentation comment. Add a documentation comment describing the vocabulary size.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:496` — Public property `kvHeads` is missing a documentation comment. Add a documentation comment describing the key-value heads per layer.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:498` — Public property `model` is missing a documentation comment. Add a documentation comment describing the inner model instance.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:502` — Public property `lmHead` is missing a documentation comment. Add a documentation comment describing the language model head and its optional nature when word embeddings are tied.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:508` — Public initializer is missing a documentation comment. Add a documentation comment with parameter documentation for the configuration argument.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:516` — Public function `callAsFunction` is missing a documentation comment. Add a documentation comment describing the forward pass, inputs, and return value.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:517` — Public class GLM4MoELiteModelInner lacks documentation of its purpose in the GLM4MoELite architecture. Add documentation explaining what GLM4MoELiteModelInner does and how it relates to the full GLM4MoELiteModel.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:522` — Public function `sanitize` is missing a documentation comment. Add a documentation comment explaining that this function removes weights for tied embeddings and handles MoE expert stacking.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:531` — Public class GLM4MoELiteModel lacks documentation as the main entry point for GLM4MoELite models. Add documentation explaining that GLM4MoELiteModel is a Mixture of Experts variant of GLM4 and its usage.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:532` — Public property vocabularySize in GLM4MoELiteModel lacks documentation explaining what it represents. Add documentation explaining that vocabularySize is the vocabulary size of the model.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:533` — Public property kvHeads in GLM4MoELiteModel lacks documentation explaining its purpose. Add documentation explaining that kvHeads contains the number of key-value heads per layer for attention computation.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:534` — Condition nested 4 levels deep (if numMptLayers > 0, filter closure body, for loop, then if), exceeding the 3-level threshold. Combining closures with nested loops and conditionals creates high cognitive overhead. Extract the filter predicate into a named helper method: `func shouldRemoveLayer(_ key: String, _ numMptLayers: Int) -> Bool`. This makes the intent clear and reduces the apparent nesting by one level, improving readability.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:535` — Public property model in GLM4MoELiteModel lacks documentation explaining its role. Add documentation explaining that model is the core GLM4MoELiteModelInner instance containing MoE layers and embeddings.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:537` — Public initializer of GLM4MoELiteModel lacks documentation of construction behavior. Add documentation explaining how GLM4MoELiteModel is instantiated and MoE-specific initialization.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:547` — Public method callAsFunction in GLM4MoELiteModel lacks documentation of forward pass behavior. Add documentation explaining the forward pass through the Mixture of Experts model, including routing behavior and cache usage.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:553` — Public method sanitize in GLM4MoELiteModel lacks documentation of its complex weight transformation logic. Add documentation explaining that sanitize handles expert stacking, kv_b_proj conversion, quantization handling, and tied embedding removal.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:603` — Public struct `GLM4MoELiteConfiguration` is missing a documentation comment. Add a documentation comment describing the configuration parameters for GLM4-MoE-Lite.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:631` — Public struct GLM4MoELiteConfiguration lacks documentation of its purpose. Add documentation explaining that GLM4MoELiteConfiguration holds hyperparameters for GLM4MoELite Mixture of Experts models.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:653` — Public Codable initializer `init(from:)` is missing a documentation comment. Add a documentation comment for the Codable decoder initializer.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:692` — Public property `loraLayers` is missing a documentation comment. Add a documentation comment describing the layers available for LoRA fine-tuning.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:711` — Public initializer for GLM4MoELiteConfiguration (init from decoder) lacks documentation. Add documentation explaining that this is a Codable deserializer for loading GLM4MoELiteConfiguration from JSON config files.
- [ ] `Libraries/MLXLLM/Models/GLM4MOELite.swift:746` — Public property loraLayers in LoRA extension lacks documentation. Add documentation explaining that loraLayers returns the model's decoder layers for LoRA fine-tuning support.
