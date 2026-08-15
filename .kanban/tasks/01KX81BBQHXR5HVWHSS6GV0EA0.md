---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx9b5xeynhc9ktmwjq9wbt96
  text: |-
    Located and fixed all 7 doc comments by quoted text (line numbers had shifted, as expected):
    1. loadContainer() -- "Loads the model container..." (blank line inserted after "when one exists.")
    2. removePromptCache(modelID:) -- "Drops one model's remembered prompt cache..." (blank line after "container/tokenizer/constraint caches.")
    3. hasCachedXgTokenizer(modelID:) -- "Whether an `GrammarTokenizer` is already cached..." (blank line after "given model.")
    4. evictAll()-equivalent cache-eviction static func -- "Evicts every cached model..." (blank line after "model weights.")
    5. evict() -- "Drops this model from the shared cache..." (blank line after "held by its weights.")
    6. isDownloadInFlight-equivalent (genuine-download check) -- "Whether a *genuine download* is in flight..." (blank line after "...already-present model.")
    7. samplingMode(from:) shim (nested private/internal type) -- 3-sentence run-on, blank line inserted after "...no seed-injection hook)."

    Confirmed via `git diff` that only these 7 blank-line insertions changed -- no wording changes, no other doc comments touched, no production code changed.

    Verification:
    - `swift build`: exit 0, clean (one pre-existing unrelated deprecation warning at a different line, not touched by this change).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - `xcrun xctest` (unfiltered) per bundle, all green:
      - MLXFoundationModelsTests: 203 tests / 36 suites passed
      - CXGrammarTests: 7 tests / 5 suites passed
      - MLXGuidedGenerationTests: 63 tests / 14 suites passed
      - MLXLMTests: 245 tests / 19 suites passed

    Adversarial double-check dispatched for sign-off. Task left in `doing` per process -- not committed, ready for /review.
  timestamp: 2026-07-11T19:41:27.902446+00:00
- actor: claude-code
  id: 01kx9bavsyan3bgpd45b8yc82y
  text: 'Adversarial double-check: PASS. Confirmed diff is exactly the 7 blank-line insertions (19 insertions/6 deletions, 1 file), correct placement at each site, no wording changes, no other doc comments touched, doc-comment attachment to declarations intact. Noted 18 other pre-existing doc comments in the file have the same run-on pattern but are explicitly out of scope for this task. Task left in `doing`, not committed, ready for /review.'
  timestamp: 2026-07-11T19:44:10.046817+00:00
- actor: claude-code
  id: 01kx9c3zxdh0fjx8bb0y3e706y
  text: |-
    Committed at `495823f885d4d6283f3f6b684eb664a680934084` (1 file, 19 insertions/6 deletions — pure blank-line splits at all 7 doc comments, verified by adversarial double-check before committing).

    Review (`review sha HEAD~1..HEAD`) returned 5 findings, ALL REJECTED as out-of-scope/stale: `git diff HEAD~1..HEAD` confirms the only touched hunks are at lines 206-987 (the 7 doc-comment sites). The 5 findings cite abbreviated variable names (`finalAnswerDef`, `obj`, `argsStr`) at lines 1571-1902 — entirely pre-existing tool-calling code this commit never touched. Genuine pre-existing debt, out of scope for a doc-comment-only fix; not tracked further here since it's a separate naming-convention concern in unrelated code, not part of this task's own diff.

    Task's own scope (7 doc-comment blank-line splits) is complete and verified. Moving to done.
  timestamp: 2026-07-11T19:57:53.453108+00:00
position_column: done
position_ordinal: ab80
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