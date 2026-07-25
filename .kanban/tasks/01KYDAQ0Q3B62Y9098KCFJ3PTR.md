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
position_column: doing
position_ordinal: '80'
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