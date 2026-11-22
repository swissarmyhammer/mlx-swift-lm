---
assignees:
- claude-code
position_column: done
position_ordinal: ffffda80
title: Inkling 2 guided (@Generable) generation emits junk JSON (every string value "}"), with or without MTP
---
Found while fixing ^hdq4m9p (branch inkling2-mtp-fm, test Inkling2FoundationModelsMTPTests). Through MLXLanguageModel + LanguageModelSession, a @Generable response on mlx-community/Inkling-2-Mini-4bit produces schema-valid JSON whose string values are all `"}"`. Plain and MTP give the same output, so the drafter is not the cause; the grammar/tokenizer interplay for Inkling under guided generation is suspect (e.g. the frame tokens, the reasoning frame before the constrained answer, token-to-bytes mapping for Inkling's tokenizer in the grammar matcher, or the constrained region starting inside a frame).

- [ ] Reproduce with a small @Generable schema in IntegrationTesting (plain decoding), and print the raw token stream.
- [ ] Find the cause (grammar vocabulary/bytes for Inkling's tokenizer, where the constrained region starts relative to <|message_model|> frames, thinking-on vs off).
- [ ] Fix with unit tests; guided output must be sensible on the real model, MTP == plain, cache reuse intact.

Relevance: Router/acp-agent may run guided passes, so this could affect bench quality.

#inkling
## Resolution (2026-11-22)

Fixed on branch inkling2-guided (f7a2ddf6, 9bffffd3, cbfa470d, 77096052, 2fd58eab), merged into inkling2-mini at d21147d8. Three manual review rounds (the sah review cannot read worktrees); last round had no serious findings and its items are fixed in 2fd58eab.
- Cause 1 (Inkling): the schema pass constrained output before Inkling's self-written answer header, with thinking on. Reasoning protocols gain `answerOpening`; schema passes render thinking-off and append it.
- Cause 2 (all models): the hard closing zone could cover the whole budget, and inside an open string the closing bias filled it with digits/braces. Fix: `closingReserves` caps the zones (hard floor 3×s), and the hard zone prefers `"` only inside an open string (byte-level `JSONStringTokens` tracking with escapes, all decode/draft/rollback paths).
- Real model: city and ranker give real values; HardReserveStressTests (Gemma, 4 tiers × 4/8/16 s) all complete, tiers 1–2 re-enabled; Qwen guided suites pass.
