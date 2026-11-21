---
assignees:
- claude-code
depends_on:
- 01M7ZVJJNB56PTH6BP74Y1ZBGN
position_column: todo
position_ordinal: '8180'
title: 'Inkling 2 Mini: tool calling usable end to end (ChatSession and MLXFoundationModels), with cache reuse'
---
Make Inkling 2 Mini usable as a coding agent with tools. This means both of these, not just one parsed call:
- the ChatSession path;
- the MLXFoundationModels `MLXLanguageModel` path that the Router and ACPAgent sessions use.

Work on branch `inkling2-mini`, at real checkpoint `mlx-community/Inkling-2-Mini-4bit`.

- A full tool loop must work on both paths:
  - The model calls a tool.
  - The tool result goes back into the transcript in Inkling's format.
  - The model uses the result.
  - Multiple tool calls per turn and parallel calls must work, if the format allows them.
- Tool results and history must render through the Inkling chat template on both paths, including the TranscriptConverter / ExecutorPromptCache path.
- Multi-turn prompt-cache reuse must hold across tool turns:
  - The cached prompt tokens after a tool turn are > 0.
  - Find any place where re-rendering breaks the prefix, and fix it.
- Thinking (reasoning frames) must stream as reasoning, not as text, on both paths, and a thinking-off setting must work.
- Unit tests come first where it is possible with scripted models. Add real-checkpoint integration tests in IntegrationTesting with no environment gate, for:
  - a tool loop through ChatSession;
  - a tool loop through MLXLanguageModel;
  - cache reuse after a tool turn.

Acceptance:
- [ ] Real Inkling 2 Mini completes a 2-step tool loop on both paths, with correct answers.
- [ ] The cached prompt tokens are > 0 on the turn after a tool result, on both paths.
- [ ] All 5 bundles are green, and the new integration tests pass.

#inkling