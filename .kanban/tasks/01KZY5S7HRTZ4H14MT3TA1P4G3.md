---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzy74v2tk97gvdmnxj2ky3ja
  text: |-
    ## Finding: upstream still has the bug

    Read upstream's current `GuidedGenerationLoop.run` in full. In the `if !ffTokens.isEmpty` branch, it feeds only `ffTokens` through the model (`for (i, ffToken) in ffTokens.enumerated() { ... model(...) ... }`). The sampled `token` that triggered the FF batch is never fed through the model in that branch — `CommitResult.tokens` never echoes the sampled token back, and the only other forward-pass site is the `else` branch (`advanceSingleSampledToken`-equivalent), which never runs when `ffTokens` is non-empty. This is the exact same bug our pre-merge fork's `onTokenCommitted`/`processFastForwardTokens` fix addressed: the sampled token's KV-cache entry is silently skipped.

    ## Fix applied

    Ported the fix onto upstream's (simpler, non-`LoopState`-struct) file as an additive change:

    1. Restored `onTokenCommitted: ((Int) -> Void)? = nil` as the last parameter of `GuidedGenerationLoop.run`, documented as firing for every token actually fed through the model during generation (not prefill).
    2. In the FF branch, added a forward pass for the sampled `token` FIRST (before the `ffTokens` loop), calling `onTokenCommitted?(tokenId)` after it — mirroring what `advanceSingleSampledToken` already did for the non-FF path.
    3. Added `onTokenCommitted?(Int(ffToken))` inside the existing FF-token loop, and `onTokenCommitted?(tokenId)` in the non-FF `else` branch.

    Kept upstream's simpler design otherwise (no `LoopState` struct, no `RunResult`/external-cache-passing — production call sites in `MLXLanguageModel.swift` already consume the bare `Int` return and upstream dropped the KV-cache pass-through feature, which nothing in this codebase's "must keep working" list depends on).

    File: `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`. Test kept as-is (`Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift`), with only the `result.tokenCount` → `result` change (upstream returns bare `Int`, not a `RunResult`) — same adjustment was needed in `Tests/MLXGuidedGenerationTests/GuidedLoopDegenerationTests.swift`, which also asserts against this API and had the identical stale `.tokenCount` reference (that file was otherwise untouched by the merge, confirmed via diff against the pre-merge tag).
  timestamp: 2026-08-13T18:46:25.114589+00:00
- actor: claude-code
  id: 01kzycxn3ejgvtgad9x7m9xmk5
  text: |-
    ### test — additional restoration found and fixed

    `onTokenCommitted` and the FF-triggering sampled-token feed (this card's original scope) were already fixed by commits f9d797f and d37bbb0.

    Running the full suite (not narrowed) surfaced two more pieces of kanban `y4s0w2j`'s fix that the wholesale replacement also dropped, both now restored and green:

    1. `ClosingTokenBias.eosOnlyBoost(stopTokenIDs:count:)` — was missing. The soft zone was applying the full `+100` closing-token bias to every sampled token, corrupting string content.
    2. `RepetitionCycleTracker` wiring into `GuidedGenerationLoop.run` — the tracker type still shipped in the target but nothing called it. Restored: record every sampled token, rebuild a `-1000` suppression array on each new cycle detection, apply it every zone, clear the window on each fast-forward splice.

    Also fixed a related, unrelated-to-the-merge bug found in the same run: `SchemaConverter.encodeToolCallingGrammar` never hoisted/namespaced a tool's `$defs` the way `encodeToolCallingEnvelopeJSON` does, so `ToolCallingSchemaTests.grammarBuilderHoistsNestedDefsInBothArms` failed. Fixed to match.

    Commits: 27464a9 (SchemaConverter), f504f99 (ClosingTokenBias + GuidedGenerationLoop).

    This card's own scope (onTokenCommitted) is done. Leaving the card open only if a person wants a final read-through of the diff; the build and tests are green.
  timestamp: 2026-08-13T20:27:21.070153+00:00
- actor: claude-code
  id: 01kzyvfdd90mn57bfmb980wc6q
  text: |-
    ### Closed — the work is committed

    The port landed in commit `f9d797f` on branch `catch-up-upstream`, during the catch-up to the official `ml-explore/mlx-swift-lm`.

    **The finding that made the card necessary:** upstream STILL holds the bug that `onTokenCommitted` guarded. In `GuidedGenerationLoop.run`, the fast-forward branch feeds only the jump-forward tokens through the model. The sampled token that STARTED the batch never goes through, thus its KV-cache entry goes away without a sign. `CommitResult.tokens` does not give the sampled token back, and the other forward-pass site is the `else` branch, which does not run while `ffTokens` holds anything.

    **The fix**, applied on top of upstream's simpler file as an addition:
    1. `onTokenCommitted: ((Int) -> Void)? = nil` is a parameter of `GuidedGenerationLoop.run` again.
    2. The fast-forward branch now feeds the sampled token through the model FIRST, before the `ffTokens` loop.
    3. `onTokenCommitted` fires for each token that the model reads.

    Upstream's simpler design stays otherwise: no `LoopState` struct and no `RunResult`, because the call sites in `MLXLanguageModel.swift` read the bare `Int`.

    `swift test` is green: 1068 tests, 0 failures.

    **This bug belongs upstream.** It is a defect of `ml-explore/mlx-swift-lm` that this fork now carries a fix for. A pull request to the official repository would give the fix back.
  timestamp: 2026-08-14T00:41:43.081729+00:00
position_column: done
position_ordinal: f480
title: Port onTokenCommitted onto the upstream GuidedGenerationLoop
---
`swift build --build-tests` fails with exactly one error:

```
Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift:186:31:
error: extra argument 'onTokenCommitted' in call
```

`swift build` and `xcodebuild build-for-testing -scheme IntegrationTesting` both pass. This one test target is the only thing left.

## Cause

The upstream catch-up merge (`-X theirs`) replaced `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`. Both sides created that file after the merge base (ours 1235 lines, upstream 598), so the merge took the upstream file whole.

Our `onTokenCommitted: ((Int) -> Void)?` parameter went with it. The upstream `GuidedGenerationLoop.run` has no such parameter and returns `Int`, while the test expects a value with a `tokenCount` member.

## Why this matters

The test is a regression guard for a real bug. Its own comment states it:

> without the fix, `committedTokenIDs` is only [66, 67, 68] ("B", "C", "D") -- the sampled token 'A' (65) that triggered the FF batch is silently never fed through the model

So a sampled token that starts a fast-forward batch never reaches the model. Check whether the upstream loop has this same bug. If it does, the port is a bug fix and not only an API addition.

## Decide first

Either port `onTokenCommitted` through the upstream loop and its helpers, or drop the test if upstream's structure makes the bug impossible. Do not delete the test without proving the bug cannot happen.

Recover the old code with:
`git show pre-upstream-merge-2026-08-13:Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift` #upstream-catch-up-guided-generation