---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: Inkling 2 guided (@Generable) generation emits junk JSON (every string value "}"), with or without MTP
---
Found while fixing ^hdq4m9p (branch inkling2-mtp-fm, test Inkling2FoundationModelsMTPTests). Through MLXLanguageModel + LanguageModelSession, a @Generable response on mlx-community/Inkling-2-Mini-4bit produces schema-valid JSON whose string values are all `"}"`. Plain and MTP give the same output, so the drafter is not the cause; the grammar/tokenizer interplay for Inkling under guided generation is suspect (e.g. the frame tokens, the reasoning frame before the constrained answer, token-to-bytes mapping for Inkling's tokenizer in the grammar matcher, or the constrained region starting inside a frame).

- [ ] Reproduce with a small @Generable schema in IntegrationTesting (plain decoding), and print the raw token stream.
- [ ] Find the cause (grammar vocabulary/bytes for Inkling's tokenizer, where the constrained region starts relative to <|message_model|> frames, thinking-on vs off).
- [ ] Fix with unit tests; guided output must be sensible on the real model, MTP == plain, cache reuse intact.

Relevance: Router/acp-agent may run guided passes, so this could affect bench quality.

#inkling