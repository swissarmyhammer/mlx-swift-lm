---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
title: 'Qwen3.8 prefill: checkpoint feeds use the model''s prefill step'
---
## Why
Found while doing ^mn44v3n. `PromptCacheCheckpoint.prefill` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift) feeds the prompt up to each checkpoint boundary with `prefill.forEachChunk(total:reserving: 0)`, which takes the generic default step of 512 when the caller sets none. It never calls `model.prepare`, so the 2048 default that ^mn44v3n gave Qwen 3.5 `prepare` does not reach it. On a hybrid model, ChatSession and the FoundationModels executor feed nearly the whole prompt through this path (leading-system checkpoint, then last-message-end checkpoint), and only the generation-prompt tail reaches `prepare`. So multi-turn Qwen3.8 prefill still runs at step 512.

## What
- Let the checkpoint feed use the model's own default prefill step when the caller sets none (for example a `LanguageModel` requirement or a property with a default of `PrefillParameters.defaultStepSize`, overridden by Qwen 3.5 and Flint), keeping any caller-specified step.
- The checkpoint position must stay the message boundary.

## Acceptance
- [ ] A unit test shows that a checkpoint feed of a Qwen 3.5 model with no caller step runs in 2048-token chunks.
- [ ] `qwencheck.sh` session check passes (checkpoint restore on follow-ups); first-turn prefill time of a long ChatSession prompt recorded before and after.
- [ ] All five bundles green. #qwen-perf