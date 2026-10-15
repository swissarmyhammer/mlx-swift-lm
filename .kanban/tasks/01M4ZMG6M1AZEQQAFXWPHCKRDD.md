---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
title: 'ChatSession: the checkpoint probe renders the whole conversation again on each turn of a hybrid model'
---
## What

To find the end of the last message, `ChatSession` (from ^95mp4sw) renders and tokenizes the conversation plus one user message `.` on each turn of a hybrid model (Qwen 3.5/3.8). At ~29k tokens this costs about 0.17 s for each turn (qwen38bench `render` mode: prompt 0.17 s, probe 0.17 s). Turn 2 at 32k prefills in 0.45 s, thus the probe adds about 35 % to the time before the first token. The time is not in `promptTime`.

## Approach

1. Measure the probe time at 32k in the bench with the render mode.
2. Find a cheaper way to the same boundary: for example, render the probe once per conversation shape and reuse the length of the generation prompt, or render only the tail. The boundary must stay the token index where the render of the messages plus one more message parts from the prompt.
3. Keep the unit tests of `ChatSessionRecurrentCheckpointTests` green.

## Acceptance Criteria

- [ ] The probe cost at 32k is measured before and after.
- [ ] The checkpoint boundary is the same as before for Qwen 3.8 (thinking on and off).
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance