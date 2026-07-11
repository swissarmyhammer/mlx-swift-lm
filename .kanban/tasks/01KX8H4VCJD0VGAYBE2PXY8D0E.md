---
assignees:
- claude-code
position_column: done
position_ordinal: a280
title: Review of 54d7f24..HEAD (suite rewrite + byte-budget eviction)
---
Scope: 54d7f24..HEAD (5 commits: 1c7e549 chunk-semantics suite rewrite = task ^t71kdmj; ed42775+0d26e8b byte-budget LRU eviction + slot-limit API removal = task ^2sdt6dj; b20ecd9/0cce82d test fixture dedup + docs).

Empirically verified alongside the engine pass: 181 tests / 34 suites all green; public setPromptCacheByteBudget landed with the peak-memory model documented ("peak ≈ byteBudget + in-flight requests × assembled-prefix size"); orphan-lineage reclamation implemented in PromptCache.swift and tested in PromptCacheByteBudgetTests.swift.

## Review Findings (2026-07-11 06:52) — resolved, commit de41a70

- [x] `suppressedLoadIDs: Set<String> = []` → constructor form. Fixed (also legitimately requested by this exact finding); a LATER review round contradicted this and asked to revert to `= []` — REJECTED as a reversed-direction false positive, see below.
- [x] `makeXgTokenizer`/`makeTokenizerBias` cache-get-or-create duplication — extracted shared `getOrCreateCached<T>` helper.

## Review Findings (2026-07-11 08:47) — triage

- [ ] Finding demanding `suppressedLoadIDs` revert from `Set<String>()` back to `Set<String> = []`. REJECTED — directly contradicts the immediately-prior, already-applied correct fix (Set has no literal syntax in Swift; `Set<String>()` is the idiomatic empty-Set construction). Same ping-pong pathology already documented for `entryID`/`quantizedKVStart` elsewhere this session.
- [ ] 4 findings demanding `makeXgTokenizer`/`hasCachedXgTokenizer` (both instance and static) → `makeXGTokenizer`/`hasCachedXGTokenizer` (uppercase-both-letters). REJECTED — this is the exact reverse of the deliberate `XG`→`Xg` convention established early in this session's history (100+ existing uses throughout this file), already explicitly rejected at least once before under the identical finding class. Not touching.
- [ ] `prepareRespondSetup` complexity (~line 2138) — confirmed pre-existing via `git diff HEAD~1..HEAD`, same function/gates already tracked in `w7m0jm2`. Noted there, no new task.
- [ ] `commitPromptCache`-area do-catch/cache-management complexity (~line 2855) — confirmed pre-existing via `git diff HEAD~1..HEAD`. Deferred to new tracking task `dfk2b64`.

All findings from both rounds now resolved (fixed, rejected with documented reasoning, or deferred to tracking tasks). Moving to done.