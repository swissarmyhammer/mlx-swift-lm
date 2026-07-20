---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxy1ykn24pnzwg25p0g2zqns
  text: |-
    Raw diagnostic trace from the instrumented real-weights run (2026-07-19, mlx-community/Qwen3.6-27B-mxfp4, PromptCacheHybridReuseTests):

    ```
    resolveHybrid: no checkpoints stored, newTokens=19
    commit(text): prompt=19 finalOffset=27 actual=8 reencoded=8
    commit(ids): prompt=19 observed=8 refOffset=27 advance=8 reconcile=matches
    store: hybrid snapshot OK tokens=27
    resolveHybrid: candidates=1 newTokens=46
    resolveHybrid: candidate len=27 MISS firstDivergence=Optional(17)
    resolveHybrid: winner len=nil
    commit(text): prompt=46 finalOffset=54 actual=8 reencoded=8
    commit(ids): prompt=46 observed=8 refOffset=54 advance=8 reconcile=matches
    store: hybrid snapshot OK tokens=54
    ```

    Token arrays at the MISS:
    stored (27) = [248045, 846, 198, 44240, 359, 5834, 6, 303, 6681, 799, 3299, 13, 248046, 198, 248045, 74455, 198, **248068, 198**, 8160, 579, 264, 7047, 1817, 25, 271, 16]
    new (46)    = [248045, 846, 198, 44240, 359, 5834, 6, 303, 6681, 799, 3299, 13, 248046, 198, 248045, 74455, 198, 8160, 579, 264, 7047, 1817, 25, 271, 16, 248046, 198, 248045, 846, 198, 6820, 1910, 359, 27450, 6, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198, **248068, 198**]

    Reading: 248045/248046 = turn start/end, 846 = "user", 74455 = "assistant", 198 = "\n". The 8 generated tokens [8160...16] reappear verbatim in round 2. Only divergence: [248068, 198] after the FINAL assistant header — the generation-priming/thinking-suppression injection, present mid-sequence in the stored round-1 tokens but absent from round 2's re-render of that turn (and re-appended at round 2's own tail). Store path is healthy; reconcile was `.matches` both rounds (EOS-trim theory disproven for this shape).
  timestamp: 2026-07-19T20:44:14.370196+00:00
depends_on:
- 01KXY1WB8NT75Q1ECJC2JCT06J
position_column: todo
position_ordinal: '9180'
title: 'PromptCache: hybrid checkpoints must snapshot at the transcript-stable boundary (fixes Qwen3.6 cachedTokenCount == 0)'
---
## What

**ROOT CAUSE — verified on real weights 2026-07-19, do not re-litigate.** The hybrid checkpoint mechanism (task ^r9rf5g7) stores and matches correctly at the PromptCache level, but never produces reuse for real Qwen3.6 sessions. Instrumented run of the (currently failing, untracked) `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift` against `mlx-community/Qwen3.6-27B-mxfp4`:

- Round 1: prompt 19 tokens, 8 generated, reconcile `.matches`, checkpoint stored OK (27 tokens). The EOS-trim theory in the code's "KNOWN, ACCEPTABLE DEGRADATION" comment is NOT what fires.
- Round 2: prompt 46 tokens; the stored 27-token sequence MISSES with first divergence at index **17** — inside round 1's own prompt.
- Token-level evidence: positions 14–16 (`<|im_start|>`, `assistant`, `\n`) match in both renders, and round 1's 8 generated tokens reappear verbatim in round 2 — the ONLY divergence is tokens 17–18 of round 1's prompt: `[248068, 198]`, a **generation-priming/thinking-suppression special token + newline** that Qwen3.6's chat template injects after the FINAL assistant header only (the executor renders reasoning-suppressed prompts for this family: `ReasoningConfig` row `(.prefix, "qwen3", qwen3ThinkConfig)`). Round 2's re-render of that turn as a PAST turn has no such tokens, and ends with its own priming suffix instead.

Consequence: round N's fed token sequence is NEVER a prefix of round N+1's prompt for this template family. A post-generation hybrid checkpoint bakes the priming tokens into the middle of its sequence; Mamba state cannot be trimmed backward past them, so such checkpoints are structurally unmatchable. (The pure-attention chunk store shares the same divergence but recovers differently; do not touch it here.)

**THE FIX — checkpoint at the transcript-stable boundary, before generation:**

1. **Stable boundary**: in `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, when creating the prompt-cache slot for a hybrid stack (`PromptCache.isHybridMambaAttention`), compute `stableLen = commonPrefixLength(promptTokens, baseTokens)` where `baseTokens` is the SAME messages/tools rendered with `addGenerationPrompt: false` and WITHOUT reasoning-suppression additionalContext (use the new Tokenizer capability from ^2jct06j; it returns nil for tokenizers that can't → skip, behavior unchanged). `baseTokens` is exactly what future rounds re-render for these turns, so `promptTokens[0..<stableLen]` is the guaranteed-reusable prefix.
2. **Split prefill + pre-generation store**: in `makePromptCacheSlot` (which already has `context: ModelContext`), for a hybrid cache where `matchedLen < stableLen`: manually forward-pass `promptTokens[matchedLen..<stableLen]` through `context.model` with the resolved cache (chunked steps, e.g. 512, mirroring prefill conventions), then `await MLXLanguageModel.storePromptCache(modelID:, tokens: Array(promptTokens[0..<stableLen]), cache:)` — `PromptCache.store`'s hybrid branch snapshots it (its `offset == tokens.count` verification passes at the boundary by construction). Then hand `feedInput = promptTokens[stableLen...]` to generation. `cachedTokenCount` (promptTokens.count − fed) semantics unchanged.
3. **Keep** the existing post-round hybrid store in `commitPromptCache` (it is correct and useful for templates without generation-region injection); the pre-generation stable-boundary checkpoint simply competes in `resolveHybridCheckpoint`'s longest-prefix scan.
4. **Revise the integration test's bound** (`PromptCacheHybridReuseTests.swift`, currently untracked — coordinate with whatever session owns it): full-prefix reuse (`>= prompt+output − 1`) is UNACHIEVABLE for this template family by the physics above; the correct maximum is the stable prefix. Assert `second.cachedTokenCount >= first.promptTokenCount - 8` (allowance = assistant header + priming region) AND `> 0`, and document the root cause in the test comment (replace the now-disproven EOS-trim explanation there and in `PromptCache.cacheAdvanceOffset`'s doc).

Unit-test the boundary logic with synthetic hybrid models (extend `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift` patterns): split-prefilled checkpoint at a boundary restores and matches a continuation that diverges only in the dropped generation-region suffix — logits equivalent to full forward pass (same 1e-3 bar as existing tests).

Diagnostic technique if re-instrumentation is needed: temporary `pcdbg()` file-logger traces at `PromptCache.store`/`resolveHybridCheckpoint`/`commitPromptCache` writing to /tmp/pcdebug.log (this run's traces are in the task comment thread) — REMOVE before committing.

## Acceptance Criteria

- [ ] `PromptCacheHybridReuseTests` passes against real `mlx-community/Qwen3.6-27B-mxfp4` with the revised stable-prefix bound (round 2 `cachedTokenCount >= first.promptTokenCount - 8` and `> 0`)
- [ ] New unit test: split-prefill checkpoint at a stable boundary + continuation with a divergent generation-region suffix → checkpoint matched, suffix-only feed, logits match full forward pass within 1e-3
- [ ] Pure-attention path untouched: `PromptCacheReuseTests` (Qwen2.5) and the full `swift test` suite stay green
- [ ] Stale/disproven EOS-trim explanations corrected in code comments where they claim to explain the Qwen3.6 zero-reuse behavior
- [ ] No temporary instrumentation left in the committed diff

## Tests

- [ ] Extend `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift` (stable-boundary unit tests, tiny models)
- [ ] Run: `swift test --filter MLXFoundationModelsTests` → green
- [ ] Run: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheHybridReuseTests` → passes
- [ ] Run: `... -only-testing:IntegrationTestingTests/PromptCacheReuseTests` → still passes

## Workflow

- Use `/tdd` — the failing integration test already exists; make it pass via the design above. #qwen