---
assignees:
- claude-code
position_column: todo
position_ordinal: '9280'
title: 'Qwen3.6: set preserve_thinking + replay reasoning_content so prompt-cache continuity extends through responses'
---
## What

Qwen3.6's chat template (`chat_template.jinja` in the model repo; same in unsloth's copy) breaks prompt-cache prefix continuity two ways, both fixable from our side:

1. **History think-stripping**: past assistant turns re-render WITHOUT their `<think>` reasoning unless `preserve_thinking` is true (template line ~100: `{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}` renders `<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}`; otherwise content-only).
2. **Generation-priming block**: the live turn ends with `<think>\n\n</think>\n\n` (when `enable_thinking=false`, template ~line 149) or `<think>\n` (thinking on) — absent from default history re-renders.

**Key insight — `preserve_thinking: true` makes BOTH problems disappear:**
- No-think sessions: a past assistant turn with empty `reasoning_content` renders `<think>\n` + `''` + `\n</think>\n\n` + content = **exactly the live priming block** — round N's fed tokens become byte-continuous with round N+1's re-render.
- Thinking sessions: reasoning is replayed verbatim in history, so the generated `<think>...` tokens re-match too.

Combined with the hybrid-checkpoint mechanism (^er33v06 / commit `83c43e8`), this upgrades Qwen3.6 reuse from "prompt prefix only (~17 of 27 in the probe run)" to near-full-transcript reuse — the strong bound sibling-repo tests like FoundationModelsRouter's `secondTurnReusesFirstTurnsKVCache` (expects `cached ≈ turn1 input+output`) actually want.

Implementation:
1. `Libraries/MLXFoundationModels/TranscriptConverter.swift`: when converting transcript entries, attach each assistant response's reasoning (from the transcript's reasoning entries, which the executor already routes to `.reasoning`) as `reasoning_content` on the corresponding assistant message dict. Empty/absent reasoning stays empty — that's correct for no-think sessions.
2. Pass `preserve_thinking: true` in the template `additionalContext` for the Qwen3.6/qwen3_5 family render. Wire it via `ReasoningConfig` (a new field alongside the `.templateFlag` strategy, e.g. `historyPreservationKey: "preserve_thinking"`) rather than hard-coding in the executor; only families whose config declares the key get it. Templates without the variable ignore unknown kwargs — but gate anyway to avoid surprising other models.
3. **Verify empirically against real weights** — the retokenization seam (`<think>\n` + generated text re-tokenized as one string) must reproduce the live token ids exactly for continuity to hold; assert via the real-model integration test, not by inspection.

Caveats to document: reasoning stays in context (context growth per turn — the tradeoff the template flag exists for); confirm response quality is acceptable with preserved thinking (Qwen's default convention strips it; the template authors added the flag expressly for agentic/cache-friendly flows).

## Acceptance Criteria

- [ ] Past assistant messages carry `reasoning_content`; render for qwen3_5-family includes `preserve_thinking: true`
- [ ] Real-weights two-round test (PromptCacheHybridReuseTests pattern, mlx-community/Qwen3.6-27B-mxfp4): round 2 `cachedTokenCount` ≈ round 1's `promptTokenCount + outputTokenCount` within a small fixed slack (the strong bound, previously unreachable) — in BOTH suppressed and thinking modes
- [ ] Non-Qwen3.6 families' renders byte-identical to before (regression: existing TranscriptConverter + template-mirror tests green)
- [ ] `swift test` fully green

## Tests

- [ ] Extend `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift`: reasoning_content attachment + preserve_thinking context gating (unit, no weights)
- [ ] Extend/adjust `IntegrationTesting/.../PromptCacheHybridReuseTests.swift`: strengthen the bound back toward full-prefix reuse once this lands
- [ ] Run: `swift test --filter MLXFoundationModelsTests` → green; the integration test → passes with the strong bound

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #qwen