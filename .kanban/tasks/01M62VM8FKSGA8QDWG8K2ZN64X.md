---
assignees:
- claude-code
position_column: todo
position_ordinal: '9e80'
title: 'FoundationModels: a follow-up turn recomputes the previous assistant reply (160 of 235 tokens) instead of extending the cache'
---
Found during ^jyqwe0c with the real-model two-turn test (IntegrationTesting MTPPromptCacheTests, Qwen3.8-27B-4bit): turn 2 recomputes 160 of 235 prompt tokens, with or without MTP. The ExecutorPromptCache main cache only reaches the end of the turn-1 user message, so the turn-1 reply and everything after it are recomputed. Also, a turn that ends with its stop token in the cache cannot extend on the next turn (the next prompt omits the stop token) and must restore from the checkpoint.

User priority: a follow-up turn must never recompute the chat; only the new user message (plus template glue) should be prefilled, as ChatSession achieves (edit turn 123 tokens).

- [ ] Find why the cached entry ends at the user message (template re-render of the reply? thinking-block stripping? commit point?) and record it
- [ ] Commit the generated reply into the cache entry so turn 2 extends it; handle the trailing stop token (trim or treat as matching)
- [ ] Unit test: turn 2 prefill count equals only the new tokens, with and without MTP, and the drafter resumes
- [ ] Release integration: report recomputed tokens per turn for MTPPromptCacheTests; target turn 2 ≈ new-message tokens only

#mtp #foundation-models #prompt-cache