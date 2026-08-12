---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzs5nq1sa4tmenb37dxxxcd2
  text: |-
    ### finish — stuck
    - evidence: The card requires a person to write the choice, A or B. The agent must not make this choice. The /finish batch on 2026-08-11 skipped this card for that reason.
    - next: The user writes the choice on this card. Then a new /finish or /implement run can do the follow-up work.
  timestamp: 2026-08-11T19:44:28.729269+00:00
- actor: claude-code
  id: 01kzsr9n8w4nvcvj5kkw7gs570
  text: |-
    ### The user delegates the decision — 2026-08-11
    The user wrote: "so i don't know enough to decide this-- you need to make DeepSeek work. research and find the right answer."
    Thus the agent has sign-off to research the question, write the choice on this card with its evidence, and do the follow-up work. The criterion for the choice is: DeepSeek must work — a correct prompt, or a loud error. A silent wrong prompt is not acceptable.
  timestamp: 2026-08-12T01:09:56.636412+00:00
- actor: claude-code
  id: 01kzsrms3pngk3cs1q3zabg42m
  text: |-
    ### Research — what the repository does today

    1. The prompt path for each LLM is `LLMUserInputProcessor.prepare` in `Libraries/MLXLLM/LLMModelFactory.swift`. It calls `tokenizer.applyChatTemplate`. When the tokenizer has no chat template, it catches `TokenizerError.missingChatTemplate`, writes one line with `print`, joins the message contents with two line feeds, and returns those tokens. This is a silent wrong prompt — the exact failure the criterion forbids.
    2. `deepseek_v4` is not in `LLMTypeRegistry` yet — only `deepseek_v3` is. The wiring is card `^mjrzkgm`. Thus today a DeepSeek-V4 checkpoint stops at load with an unsupported-model-type error, which is loud. When `^mjrzkgm` lands, the silent fallback becomes reachable.
    3. No `.jinja` file is in this repository. `DSV4Minimal.jinja` was never ported. `DeepSeekV4ChatEncoder` exists in `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift` but no pipeline code selects it yet.
    4. The encoder file header verifies the checkpoint facts: `deepseek-ai/DeepSeek-V4-Flash` @ 60d8d70 has no `chat_template` key in `tokenizer_config.json` or `tokenizer.json`, and `encoding/encoding_dsv4.py` is the only prompt builder DeepSeek publishes.
    5. A Jinja template cannot state the `drop_earlier_reasoning` rule (a whole-conversation rule that also counts `context_len`) and cannot choose the task-marker order (the choice follows the task type, not the message role). Thus a template can only disagree with the encoder, and a second path that disagrees gives a wrong prompt with no error.
  timestamp: 2026-08-12T01:16:01.014704+00:00
- actor: claude-code
  id: 01kzss49d6xbh42mgf18fnqwqr
  text: |-
    ### implement — changed
    - evidence: Decision B is on the card, with the evidence chain and the refusal path. The user delegated the choice (see the comment of 2026-08-12). Code, by TDD (the tests failed first for the missing API, then passed):
      - `Libraries/MLXLLM/LLMModel.swift` — `LLMModel` gets `missingChatTemplateRefusal: String?`, default `nil`.
      - `Libraries/MLXLLM/LLMModelFactory.swift` — new public `PromptPreparationError.plainTextFallbackForbidden`; `LLMUserInputProcessor` (now internal, for the pinning tests) throws it instead of the silent plain-text fallback when the model gives a refusal; `_load` reads the refusal from the model.
      - `Libraries/MLXLLM/Models/DeepSeekV4.swift` — `DeepSeekV4Model` returns the refusal message, which names `DeepSeekV4ChatEncoder`.
      - `Tests/MLXLMTests/DeepSeekV4PromptFallbackTests.swift` — three tests pin the refusal and pin the unchanged fallback for models that permit it, so a later edit cannot bring a silent template fallback back.
      - `swift build` clean. Full `swift test`: 457+80+282+7 Swift Testing tests passed, 245 XCTest tests with 0 failures. The 2 skipped tests are pre-existing, documented, and belong to kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ.
    - next: /review. Note for `^mjrzkgm` (registry wiring): when `deepseek_v4` lands in `LLMTypeRegistry`, the refusal makes generation fail loudly until the encoder path is wired into the input processor.
  timestamp: 2026-08-12T01:24:29.222885+00:00
position_column: doing
position_ordinal: '8180'
title: 'Decision needed: keep or drop the DSV4Minimal.jinja fallback template'
---
## What

A person must decide. Do not write code before the decision.

The card `^gbsaqc2` (Port DeepseekV4ChatEncoder core rendering) named a second,
Jinja path for DeepSeek-V4 prompts: `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`,
from `osaurus-ai/vmlx-swift-lm`. The port did not carry that file. This card
records why, and puts the choice to a person.

## The technical finding

DeepSeek-V4 ships **no** `chat_template`, neither in `tokenizer_config.json` nor
in `tokenizer.json`. DeepSeek gives prompt construction as a separate Python
file, `encoding/encoding_dsv4.py`. The Swift encoder
`Libraries/MLXLMCommon/DeepSeekV4ChatEncoder` follows that Python.

A Jinja template cannot state two of the rules that the Python states:

1. **The `drop_earlier_reasoning` rule.** The rule looks at the whole
   conversation and removes the reasoning block of every turn before the last
   one. It also counts the context messages to compute `context_len`. Jinja has
   no way to state this multi-message rule.
2. **The task markers.** The six task markers (token ids 128829 to 128845) and
   the action-task path choose a different marker order for the assistant turn.
   The choice depends on the task type, not on the message role, thus a template
   that only walks the message list cannot make it.

Therefore a Jinja fallback cannot agree with the Swift encoder. A second path
that disagrees is worse than no second path: it gives a wrong prompt with no
error.

## The choice

Pick one:

- **A. Keep a partial template.** Carry `DSV4Minimal.jinja`, but only for simple
  turns: user and assistant messages, no reasoning drop, no task markers, no
  tools. The template must fail loudly, or the loader must refuse it, when the
  conversation needs a rule the template cannot state.
- **B. Drop the fallback.** DeepSeek-V4 prompts always come from
  `DeepSeekV4ChatEncoder`. A DeepSeek-V4 model with no encoder is an error, not
  a fall back to a template.

## The decision — B. Drop the fallback.

The user delegated the decision to the agent on 2026-08-12 (see the comments).
The criterion: DeepSeek must work — a correct prompt, or a loud error. A silent
wrong prompt is not acceptable.

The evidence chain:

1. `deepseek-ai/DeepSeek-V4-Flash` @ 60d8d70 has no `chat_template` key in
   `tokenizer_config.json` or `tokenizer.json`. DeepSeek publishes
   `encoding/encoding_dsv4.py` as the only prompt builder. The header of
   `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift` records both facts
   against the pinned revisions.
2. A Jinja template cannot state the `drop_earlier_reasoning` rule or the
   task-marker choice (see the technical finding above). Thus a template can
   only disagree with the encoder, and the disagreement is silent.
3. This repository never carried `DSV4Minimal.jinja`. No `.jinja` file is in
   the tree. Nothing depends on a template path for DeepSeek-V4.
4. The generic prompt path, `LLMUserInputProcessor.prepare` in
   `Libraries/MLXLLM/LLMModelFactory.swift`, had a silent plain-text fallback
   when a tokenizer has no chat template. For DeepSeek-V4 that fallback is a
   wrong prompt with no error. That path, not a missing template, was the
   real risk to the criterion.

## The refusal path (acceptance criterion 3)

- `LLMModel` gets `missingChatTemplateRefusal: String?`. The default is `nil`,
  which permits the plain-text fallback for other models.
- `DeepSeekV4Model` returns a refusal message. `LLMUserInputProcessor.prepare`
  then throws `PromptPreparationError.plainTextFallbackForbidden` instead of
  the silent plain-text prompt.
- The error message is: "DeepSeek-V4 has no chat template. Build the prompt
  with DeepSeekV4ChatEncoder. The plain-text prompt fallback is not permitted
  for this model, because it makes a wrong prompt and gives no error."
- Tests in `Tests/MLXLMTests/DeepSeekV4PromptFallbackTests.swift` pin the
  refusal, so a later edit cannot bring a silent template fallback back.

## Acceptance Criteria

- [x] A person writes the choice, A or B, on this card. (The user delegated the
      choice to the agent. The choice is B. The evidence is above.)
- [ ] If A: a new task states the exact set of turns the template handles, and
      the error the loader gives for every other turn. (Not applicable — the
      choice is B.)
- [x] If B: a new task, or this card, records the refusal path and its error
      message. (Recorded above, on this card.)

## Provenance

- Fallback template: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`.
- Source of truth: `deepseek-ai/DeepSeek-V4-Flash` — `encoding/encoding_dsv4.py`.
- Moved from `^gbsaqc2` on 2026-08-11.
#deepseek-v4