---
assignees:
- claude-code
position_column: todo
position_ordinal: 9f80
title: Fix blank-line doc-comment formatting on 7 pre-existing MLXLanguageModel.swift functions
---
## What
Surfaced by review pressure on task `2sdt6dj`'s byte-budget commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — zero matches, this commit's diff only ADDS new content, none of the 7 flagged doc comments were touched (their line numbers just shifted due to earlier insertions in the file).

Same finding class as one already partially addressed in task `9jtbtkd` (done) — several pre-existing doc comments have a multi-sentence opening summary with no blank `///` line separating the one-sentence summary from elaboration, violating this project's documentation convention (first line = single-sentence summary ending in a period; elaboration follows after a blank `///` line).

## Review Findings (2026-07-11 02:15) — to fix
- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:345` — "Loads the model container for this model, returning a cached instance when one exists." + "Shares the process-global cache…" need a blank-line split.
- [ ] `:375` — "Drops one model's remembered prompt cache without touching the container/tokenizer/constraint caches." + "Used when a round's actual generated content can't be reconciled…" need a blank-line split.
- [ ] `:418` — "Whether an `GrammarTokenizer` is already cached for the given model." + "Internal test seam (not public API)…" need a blank-line split.
- [ ] `:474` — "Evicts every cached model, tokenizer, constraint template, and per-model tokenizer bias, freeing the GPU memory held by model weights." + "Subsequent requests reload from the on-disk cache." need a blank-line split.
- [ ] `:524` — "Drops this model from the shared cache, freeing the GPU memory held by its weights." + "A subsequent `respond()`/`preload()` triggers a fresh load…" need a blank-line split.
- [ ] `:541` — "Whether the shared cache has a *genuine download* in flight for the given model…" + "Used by ``availability`` to surface a `.downloading` state." need a blank-line split.
- [ ] `:688` — 3-sentence run-on before any blank line (sampling-mode translation shim) needs splitting to single-sentence summary + blank line + elaboration.

## Acceptance Criteria
- [ ] All 7 doc comments reformatted to single-sentence summary + blank `///` line + elaboration, per this project's documentation convention.
- [ ] No behavior change — doc comments only.
- [ ] Build clean, full test suite green.

## Scope
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only. Line numbers will have shifted by the time this is picked up — relocate by the quoted doc-comment text, not line number. Not urgent/blocking — pre-existing documentation-formatting debt, not a correctness bug.