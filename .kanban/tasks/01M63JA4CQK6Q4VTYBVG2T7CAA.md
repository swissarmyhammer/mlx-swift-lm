---
assignees:
- claude-code
position_column: todo
position_ordinal: a280
title: 'FoundationModels: an MTP turn that stops on its token budget leaves its last token out of the cache, and the Qwen splice drops it from the next turn'
---
Found in ^k2zn64x (Release MTPPromptCacheTests, Qwen3.8-27B-4bit, `.reasoning`, 128-token budget). The MTP turn 1 ledger is 206 tokens (79 prompt + 127 generated), the plain ledger 207: the MTP iterator returns its final verified token without feeding it. Turn 2 splices at the `<|im_end|>` commit (reused=207 for both). `reconcilePromptCache` gets no `previousGenerationUncommittedTokens` from the executor, thus `CommittedTurnSplice.suffixStart` takes the empty-uncommitted branch and the model reads its answer without that last token before `<|im_end|>`. `ChatSession` tracks `uncommittedTokens` and passes them, so its splice declines (or feeds the token).

- [ ] Commit the uncommitted generated tokens on the executor entry (as `ChatSession.Conversation.uncommittedTokens`) and pass them to the reuse policy through `reconcilePromptCache`
- [ ] Decide with the splice what the follow-up feeds: the dropped token before the commit, so the next turn reads the whole answer
- [ ] Unit test with the Qwen35 MTP fixture: an MTP turn stopped on its budget, follow-up reads every generated token
- [ ] Release: MTPPromptCacheTests still recomputes only the new message (26 of 233) and the drafter still resumes

#mtp #foundation-models #prompt-cache