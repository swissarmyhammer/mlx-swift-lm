---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m63psny5hh34dqqjrhqxbyyw
  text: |-
    Picked up. The worktree branch started on `main` (no GuidedSpeculation files), so I reset it to `mtp` (438974bd) before any work. Research notes:
    - Today `GuidedSpeculation.decode` sends a feed of count > 1 (a target decision plus its forced tokens) to `feedForcedTokens`. That is one verify-shaped forward, then a decision, with no drafts. `GuidedDraftChain.record` also ends the chain at any draft whose commit forces tokens.
    - The Qwen shifted drafter chains on its own hidden rows, so it can step through forced tokens: the processor returns logits whose argmax is the forced token. Laziness drops the unused LM-head logits of a forced step. Only the one-layer MTP forward runs, which the later drafts need anyway.
    - The verify capacity is `GatedDeltaDecodeKernel.mostTokens` = 6 tokens. The shipped block size is 5 (4 drafter steps) with a floor of 0.75. So a decision forcing k < 4 tokens can share one verify with 4-k drafts. Longer forced runs keep the chunked `feedForcedTokens` path.
    - The plan: the chain holds pending forced tokens (from the round's own decision or from a draft's commit). It shapes those steps to the forced token and records no matcher commit for them. The verify layout is [bonus] + bonus forced + each draft's feed, with a per-draft verify index for acceptance rows and kept counts. The rollback is unchanged: it counts each draft's acceptances, and the forced tokens of the round's own decision are never rolled back.
    - `theChainEndsAtADraftThatForcesTokens` in GuidedGenerationMTPTests asserts the old behavior that this card changes. It will be replaced by a test of the new behavior in the new test file.
  timestamp: 2026-10-29T11:30:19.717987+00:00
- actor: claude-code
  id: 01m63qn2eqnyz98btekh1tm3qv
  text: |-
    Implementation landed (unit-green, Release benchmark pending):
    - `GuidedDraftChain` holds `forcedAhead`: the forced tokens of the round's decision (new `init(rules:start:forced:)`) or of the last draft's commit. `shape` returns 0/-inf logits that allow the forced token alone. `record` pops it and precondition-checks it, with no matcher call and no draft record. A draft that forces tokens no longer ends the chain. `rollBack` clears `forcedAhead`. The drafter's own logits on a forced step are unread, so MLX laziness skips their LM head. Only the one-layer MTP forward runs, and the later drafts need its hidden row.
    - `GuidedSpeculation`: `draftStepsThisRound(after: feed)` = min(maximumDrafts, forced + budget left). When the forced tokens number fewer than the steps, `speculate(after: feed)` drafts after the decision with them, and the verify is `feed + feeds of the drafts`. `VerifyBatch.draftIndices` gives each draft's acceptance row (index - 1) and the kept count on rejection. Longer forced runs keep `feedForcedTokens`. Matcher rollback still sums each rejected draft's `acceptanceCount` (draft + its forced tokens). The decision's own forced tokens are never rolled back. `commitDrafter` pairs `batch.tokens[1..<kept]` + next with hidden rows `0..<kept` unchanged.
    - Tests: new `Tests/MLXFoundationModelsTests/GuidedForcedDraftTests.swift` (6 tests: 3 chain, 3 loop at block sizes 5 and 6 over every schema x drafter mode). RED before the change: 3 behavioral failures. The plain-equality and cache-keeps-emitted guards passed both before and after. `CacheRecordingTarget` is now internal and records `forwardTokens`. I removed `theChainEndsAtADraftThatForcesTokens`, which asserted the behavior this card replaces.
    - All 5 bundles green: CXGrammar 7, MLXGuidedGeneration 72, HuggingFaceMacros 6, FoundationModels 420, MLXLM 2014. 0 failures, 0 skipped, no Swift warnings.
  timestamp: 2026-10-29T11:45:17.271508+00:00
- actor: claude-code
  id: 01m63s32f7n8hnfbwp6e5fhgvc
  text: |-
    ### implement — no-change
    - evidence: I implemented three variants of forced-tokens-as-certain-drafts and unit-tested the first: 5 bundles green, 7/72/6/420/2014 tests, 0 failures, 0 skipped. Release `qwen38GuidedJSONWithMTP` runs alternated before/after against base 438974bd, under the coordinator's bench.lock, with load recorded (table in the description). A: MTP 61.67/62.35 vs 62.68/63.71 before. B: 62.03/62.34 vs 64.17/63.47. C: 64.58/65.70 vs 64.77/65.02 (identical 31/40 accepts, so the path never fires). Divergence none in every run. No variant is faster, so I reverted all code and kept the patch outside the repo at /private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/d4dd68db-1555-410f-b902-6376979e7c8c/scratchpad/forced-drafts-attempt.patch, plus GuidedForcedDraftTests.swift beside it. The commit carries only this card.
    - What did not work: continuing a chain through a draft's forced tokens (A, B) costs about 2-3%, from one MTP forward plus a host sync per forced step. The decision-only form (C) never triggers with the shipped block of 5: every decision that forces tokens in this benchmark forces at least 4.
    - Process notes: the worktree first sat on `main`; I reset it to `mtp` before any work. The old `pgrep -f` wait deadlocked on other agents' wait loops; I followed the coordinator's rule (`pgrep -x` plus bench.lock).
    - next: the card is a measured no-go. Close it, or re-scope it to a case where decisions force 1-3 tokens (for example a tokenizer and schema with short forced keys) before trying again.
  timestamp: 2026-10-29T12:10:24.615175+00:00
- actor: claude-code
  id: 01m63t9j2hmbpaamv6kvy8yppt
  text: |-
    ### closed — measured no-go (user decision 2026-10-29)
    All three variants (forced tokens as certain drafts; plus spare verify rows; decision-only forced drafts) measured no faster on Release qwen38GuidedJSONWithMTP (A 61.67/62.35 vs 62.68/63.71, B 62.03/62.34 vs 64.17/63.47, C 64.58/65.70 vs 64.77/65.02 MTP tok/s; divergence none). Code reverted; attempt kept in the session scratchpad (forced-drafts-attempt.patch). The user chose to close this card as a no-go rather than re-scope it.
  timestamp: 2026-10-29T12:31:25.777308+00:00
position_column: done
position_ordinal: ffff8680
title: 'Guided MTP: verify grammar-forced tokens inside the draft round instead of a separate forward'
---
After ^ashrw1b (feed the decided token ahead of the forced tokens), guided JSON MTP dropped from 74.71 to 67.59 tok/s (plain 54.05 → 52.27). Part is the MTP-only cost: a decision that forces tokens now costs a 2-token forward plus a target decision before the next draft round. The implementer checked starting the round after the last forced token (fewer tokens per forward) but not treating forced tokens as zero-cost drafts in the verify batch.

User guidance: small perf gains add up; keep plain and MTP guided output identical.

- [ ] Append grammar-forced tokens to the draft chain as certain drafts (no drafter step), so one verify forward covers decision + forced + further drafts — built and measured, NOT shipped: no variant is faster (see Measurements)
- [ ] Keep matcher rollback counts and drafter hidden-row pairing correct on rejection after forced tokens — held in the measured variants (unit-green), not shipped
- [ ] Test: plain vs MTP guided output identical over every fixture schema; fed tokens equal emitted tokens — written with the variant (GuidedForcedDraftTests, 6 tests, RED then GREEN), not shipped with it
- [x] Release `qwen38GuidedJSONWithMTP`: report tok/s before/after (baseline plain 52.27, MTP 67.59)

## Measurements (Release, `qwen38GuidedJSONWithMTP`, base 438974bd, interleaved before/after runs, load is the 1-minute average)

Machine load stayed ~5–10 from other projects (EditorKit, Testudo), so absolute rates sit below the 52.27 / 67.59 baselines; compare within each pair.

| variant | run | plain | MTP | accepted | load |
|---|---|---|---|---|---|
| A: forced tokens of the decision and of every draft are certain drafts (block = shipped 5) | before | 49.01 / 49.89 | 62.68 / 63.71 | 31/40 | 10.4 / 9.2 |
| | after | 49.82 / 50.43 | 61.67 / 62.35 | 32/41 | 10.4 / 9.2 |
| B: A + forced tokens use the spare verify rows (steps up to 5, drafts capped at 4) | before | 49.87 / 49.88 | 64.17 / 63.47 | 31/40 | 6.1 / 6.7 |
| | after | 50.17 / 50.79 | 62.03 / 62.34 | 32/41 | 6.5 / 6.2 |
| C: only the decision's forced tokens are certain drafts; a forcing draft still ends its chain | before | 49.83 / 51.07 | 64.77 / 65.02 | 31/40 | 5.1 / 5.3 |
| | after | 50.97 / 51.45 | 64.58 / 65.70 | 31/40 | 5.3 / 5.7 |

Divergence was `none` and 58 tokens in every run. MTP/plain ratio: A/B ~1.24 vs 1.28 before (about 2–3% slower); C equal within noise.

Findings:
- C changes nothing in this benchmark (identical 31/40): with the shipped block of 5, no target decision here forces fewer than 4 tokens, so the decision path never fires; the decisions that force tokens force whole keys/values and keep the chunked `feedForcedTokens` path anyway.
- A/B are slower: continuing the chain through a draft's forced tokens spends drafter steps (one MTP forward plus a host sync per forced step) on tokens the old path verified in the same round as the forcing draft's tail, and the extra drafts rarely survive.
- The measured patch is saved outside the repo (scratchpad `forced-drafts-attempt.patch`); nothing ships.

#mtp #guided-generation #perf