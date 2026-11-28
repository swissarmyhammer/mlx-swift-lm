---
assignees:
- claude-code
position_column: todo
position_ordinal: '8280'
title: 'Prompt cache: the think-then-call phase of a required tool turn takes the carried cache'
---
## What
In a required-tool turn with think-then-call (`runToolCallReasoningPhase` in Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift), phase 1 generates its reasoning from a new cache of its own (`generateTokensTask`), so it prefills the whole tool-aware prompt cold on every turn. Since ^nztjxdh the grammar phase (phase 2) plans from the session's prompt cache, but phase 1 still pays a full prefill.

## Fix
- Phase 1 plans from the slot (`planPromptCache`) and commits through `withCommittedGeneration`, as a plain pass does.
- Phase 2 then plans from what phase 1 committed: its prompt (tool-aware prompt + opener + reasoning) extends phase 1's ledger up to the protocol stop; the tail after the `ProtocolPassStop` needs a rewind, which a hybrid model cannot do — take a prompt-end checkpoint in phase 1 (or stop the tail from entering the cache) so phase 2 still reuses on Qwen 3.5 / Inkling 2.
- Output must not change.

## Tests
- A required-tool turn with think-then-call after a plain turn reports cached tokens > 0 for its prompt, local and clustered, and the same tool call as a cold turn.
#cluster