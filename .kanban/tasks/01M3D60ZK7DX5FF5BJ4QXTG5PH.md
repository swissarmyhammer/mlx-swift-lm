---
assignees:
- claude-code
position_column: todo
position_ordinal: '8980'
title: SpeculativeTokenIterator.prepare drops the first main token when a model prepare returns .logits
---
## What

Found during ^k8hm41j. In `SpeculativeTokenIterator.prepare(input:prefill:)` (`Libraries/MLXLMCommon/Evaluate.swift`), the `.logits` branch of the main prefill samples a token into `y`, and the `.logits` branch of the draft prefill samples the draft's own token into `draftY`.

Facts:
- `y` is fed to the verify pass, but it is never put into `pendingTokens`, thus the first main token is not emitted.
- `draftY` is the draft sample, not `y`. The draft cache then holds a different token at that position than the main cache, and the drafts continue from the draft's own token.
- `Gemma3TextModel.prepare` returns `.tokens`, thus the Gemma3 tests do not reach this branch.

Find if a model that returns `.logits` from `prepare` can reach `SpeculativeTokenIterator`. If it can, emit `y`, and set `draftY` to `y`.

## Acceptance Criteria

- [ ] A test with a model whose `prepare` returns `.logits` shows that speculative decoding emits the same tokens as `TokenIterator` at temperature 0, or the task records why the branch is not reachable.