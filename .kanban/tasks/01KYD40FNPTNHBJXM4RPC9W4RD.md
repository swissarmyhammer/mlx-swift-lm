---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kydapbmef4fn3tr762yvd0bd
  text: |-
    RESEARCH COMPLETE — closing as not-applicable. No production code changed.

    **Step 1: architecture-wide survey.** Grepped `tieWordEmbeddings`/`tie_word_embeddings` + `lmHead`/`lm_head` across every file in `Libraries/MLXLLM/Models/*.swift` and `Libraries/MLXVLM/Models/*.swift`. Every architecture with a `tieWordEmbeddings` config property except two follows the documented `porting.md` convention exactly (verified directly in Qwen2.swift, GLM4MOE.swift [non-lite], and ~28 others): `@ModuleInfo(key: "lm_head") var lmHead: Linear?`, created only `if !args.tieWordEmbeddings`, with `lmHead(out)` when present or `embedTokens.asLinear(out)` as fallback. When tied, no separate `lmHead` module instance exists in the module tree at all — `quantize(model:)` never visits a second path for it, so the hypothesized gap (embedding quantized, separate head silently left unquantized) is structurally impossible for all of these.

    **Step 2: the two exceptions.** `Libraries/MLXLLM/Models/GLM4.swift` and `Libraries/MLXLLM/Models/GLM4MOELite.swift` (the "lite" MoE variant — NOT the sibling `GLM4MOE.swift`, which is correctly guarded) both declare `lmHead` as a non-optional `Linear`, created unconditionally in `init`, with `tieWordEmbeddings` decoded but otherwise inert. This satisfies AC1's literal question ("separate, non-shared-instance embedding/lm_head modules under tie_word_embeddings: true" — confirmed for 2 architectures, not refuted for all).

    **Step 3 (the part that determines the actual outcome): does the checkpoint shape the task cares about actually occur for these two?** Traced this against real checkpoint conventions:
    - `GLM4.swift`'s `sanitize(weights:)` actively deletes `lm_head.weight` from the loaded dict whenever `tieWordEmbeddings == true`. Since `Load.swift`'s `model.update(parameters:, verify: [.all])` requires every model parameter key to be present (`.allModelKeysSet`), and `lmHead` is a non-optional `Linear` requiring `lm_head.weight`, ANY genuinely-tied GLM4 checkpoint crashes with `UpdateError.keyNotFound` at load time — before quantization params matter at all. The "silent full-precision head" outcome never happens; it's a hard crash instead.
    - `GLM4MOELite.swift` has no such sanitize step, so it depends on what the checkpoint actually contains. Verified against upstream: fetched `ml-explore/mlx-lm`'s current `mlx_lm/models/glm4_moe.py` — its `Model` class also creates `lm_head` as a plain unconditional `nn.Linear` and also never references `tie_word_embeddings` (this Swift file is a faithful port of an equally-inert upstream implementation, not a Swift-specific regression). Also fetched `mlx_lm/utils.py`'s `quantize_model`: the default `wrapped_predicate` quantizes every module with `to_quantized` uniformly (`bool_or_params = True` unless a model-specific `quant_predicate` overrides it — `glm4_moe.py` defines none) — there is no default skip-lm_head-when-tied behavior in the upstream quantization pipeline. So a checkpoint built by the normal `mlx_lm.convert -q` path would quantize `lm_head` right alongside `embed_tokens`, giving it its own `.scales` — no gap. The only way a real checkpoint could exhibit "embed quantized, `lm_head.weight` present but no `lm_head.scales`" is a hand-rolled/selective conversion nobody in this ecosystem's tooling produces by default; the realistic failure mode for a genuinely-tied HF-sourced checkpoint (where HF's `save_pretrained` dedupes tied weights and omits `lm_head.weight` from the safetensors entirely) is the same `UpdateError.keyNotFound` crash as GLM4, not a silent precision downgrade. (Corroborated by real-world precedent: HF/mlx-community discussion threads report exactly "Missing parameters: lm_head.weight" errors for tied-embedding conversions — a crash, not a silent gap.)

    **Conclusion:** the specific shape this task investigates — a checkpoint that loads successfully but leaves a separately-declared, tied `lm_head` silently at full precision because `Load.swift`'s predicate only checks for `.scales` — does not occur for any architecture currently in this codebase. Building the described `Load.swift` predicate extension would be dead code: for the ~30 well-behaved architectures there's no separate module to backfill, and for the 2 exceptions the realistic checkpoint shapes either already quantize `lm_head` normally (no gap) or fail to load entirely for an unrelated reason (missing weight key, not missing scales) that the proposed fix would not address.

    Per the task's own explicit escape hatch, closing as not-applicable. No synthetic weight-dict test added (nothing to test — no behavior change). Ran `swift test --filter MLXLMTests`: 347 tests / 24 suites, exit 0, no regressions (unsurprising since no production code was touched).

    **Separate discovery, filed as its own task (out of this task's scope, per "no bonus scope"):** `GLM4.swift` and `GLM4MOELite.swift` both diverge from this codebase's own documented tied-embedding convention (`porting.md`, and the correctly-implemented sibling `GLM4MOE.swift`) and will crash with `UpdateError.keyNotFound` on `model.update` for any checkpoint that is genuinely `tie_word_embeddings: true` and follows the standard convention of omitting the tied weight from the safetensors. That's a real, distinct bug worth its own card — see the new task filed below.

    Task left in `doing` per instructions; not moving to review.
  timestamp: 2026-07-25T19:05:37.678310+00:00
- actor: claude-code
  id: 01kydaq66ej56a70ngtvzmeces
  text: 'Follow-up task filed for the GLM4/GLM4MOELite unconditional-lm_head crash bug discovered during this research: ^cfj3ptr ("Fix GLM4.swift/GLM4MOELite.swift: unconditional lm_head crashes on genuinely tied checkpoints"). This task (pc9w4rd) remains closed as not-applicable and stays in `doing` per instructions.'
  timestamp: 2026-07-25T19:06:04.878484+00:00
position_column: done
position_ordinal: da80
title: 'Perf: detect and quantize tied lm_head when checkpoint quantizes embeddings but not the head'
---
## What

Confirmed via reading `Libraries/MLXLMCommon/Load.swift`'s `loadWeights(modelDirectory:model:quantization:perLayerQuantization:)` (currently ~line 42-83): quantization is applied per-module purely by checking whether the checkpoint's weights dict has a `.scales` key at that module's path:

```swift
quantize(model: model) { path, module in
    if weights["\(path).scales"] != nil {
        ...
        return quantization?.asTuple  // or perLayerQuantization.quantization(layer: path)
    } else {
        return nil  // module stays unquantized
    }
}
```

For a model with `tie_word_embeddings: true` whose Swift port defines the output projection (`lm_head`) as a **separate module** from the input embedding (`embed_tokens`) — rather than literally sharing the same `Embedding`/`Linear` module instance — a checkpoint that quantizes the embedding table but omits a separate `lm_head.scales` entry (because the two are conceptually tied and the conversion tooling only emitted one set of quantization params) leaves `lm_head` silently unquantized: full-precision weights on what is often the single largest, most bandwidth-heavy matrix multiply in the entire decode step (vocab-sized projection). This is a real, previously-undocumented gap in this codebase — confirmed by inspecting `Load.swift`'s quantization predicate, which has no special-casing for tied heads at all. (Inspired by osaurus-ai/vmlx-swift's `TiedHeadQuantizationPolicy`, but this repo's design should be its own — do not port their implementation verbatim, their model/module structure differs from this codebase's.)

**Design task**: extend the quantization predicate (or add a pre-pass before it) in `loadWeights` so that when:
1. The model's configuration reports `tieWordEmbeddings == true` (check `BaseConfiguration`/per-architecture config protocols for how this is currently surfaced — e.g. `MiniMaxM3TextConfiguration.tieWordEmbeddings`, or the equivalent on other architectures), AND
2. The embedding module's path has a `.scales` entry (i.e. it got quantized), AND
3. A separate `lm_head`-equivalent module path exists in the model's module tree but has NO `.scales` entry

...then the head is quantized using the SAME quantization parameters resolved for the embedding path, rather than left at full precision. Needs research into exactly how "the lm_head module path" is identified generically across this repo's various model architectures (there may not be a single conventional path name — check a handful of existing model files, e.g. `Libraries/MLXLLM/Models/Llama.swift`, `Libraries/MLXVLM/Models/MiniMaxM3.swift`, for how `lmHead`/output-projection modules are named and whether tied-embedding handling already exists at the Swift `Module` level for any of them).

**This is narrower/lower-priority than the `MLX.compile()` decode-loop task** — it only matters for checkpoints with this exact shape (quantized embeddings + separate, unquantized, tied head). If research reveals this shape doesn't actually occur in any checkpoint this codebase currently loads (e.g. because every ported architecture shares the literal module instance for embed/head when tied), report that finding and close the task rather than building speculative handling for a shape that never occurs.

## Acceptance Criteria

- [ ] Research confirms (or refutes) that at least one architecture in this codebase has separate (non-shared-instance) embedding and lm_head modules under `tie_word_embeddings: true` -- if refuted for all current architectures, document why in a kanban comment and close as not-applicable rather than building unused code
- [ ] If applicable: a synthetic weight-dict unit test proves that a tied-embeddings config with a quantized embedding but no separate head `.scales` entry results in the head module being quantized to match, and an untied config (or a config where the head genuinely has its own `.scales`) is unaffected (no behavior change)
- [ ] Existing quantization-loading tests for at least one real architecture remain green (no regression to normal, non-tied quantization loading)

## Tests

- [ ] New test in `Tests/MLXLMTests/` (exact file TBD by research -- likely alongside existing `Load.swift`/quantization tests, find via `grep -rl "loadWeights\|perLayerQuantization" Tests/MLXLMTests/`) with synthetic weight dicts covering: tied+quantized-embedding+unquantized-head (should quantize head), untied (no change), tied+already-quantized-head (no change/idempotent)
- [ ] Run: `swift test --filter MLXLMTests` → green, no regressions

## Workflow

- Use `/tdd` — write the failing synthetic-weight-dict test first, then implement. #performance