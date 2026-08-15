---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzs00mdvg3cjrk0kny0sqyaw
  text: |
    ### Work arrived from `^gbsaqc2` on 2026-08-11 — the encoder wiring

    The section `## Added Work: wire the DeepSeek-V4 chat encoder into the chat path`
    came from the card `^gbsaqc2` (Port DeepseekV4ChatEncoder core rendering).

    **What arrived.** Call `DeepSeekV4ChatEncoder` from `Libraries/MLXLMCommon/Chat.swift`,
    `Libraries/MLXLMCommon/Tokenizer.swift` and `Libraries/MLXLMCommon/ChatSession.swift`.
    The type is already written and proved; it is in
    `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`. Nothing calls it yet.

    **Why it moved.** The wiring needs a rule that tells a caller that the loaded
    model is DeepSeek-V4. This card makes that rule. `^gbsaqc2` cannot make it.

    The plan had a circle: `^gbsaqc2` blocks `^35aw7vy`, and `^35aw7vy` blocks this
    card, but the wiring item on `^gbsaqc2` waited for this card. A card cannot wait
    for a card that waits for it. The orchestrator moved the item to the card that
    can do it. No work is dropped.

    **Do not start this section before item 1 of `## What`.** Make the
    `deepseek_v4` registry entry and the detection rule first, then wire the encoder
    to that rule.
  timestamp: 2026-08-11T18:05:35.035921+00:00
- actor: claude-code
  id: 01kztfannd1wpf3zxnw8h56mxy
  text: |-
    ### Research notes, before the first test

    - Canonical names in the tree: `DeepSeekV4Configuration`, `DeepSeekV4Model` (in `Libraries/MLXLLM/Models/DeepSeekV4.swift`), and `DeepSeekV4ChatEncoder` (in `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift`). The card predates the rename.
    - `ModelTypeRegistry` already has `contains(_:)` and an async `createModel`. `DeepSeekV4Model.init(_ configuration:)` fits the `create(...)` helper of `LLMModelFactory.swift`.
    - The write side gives the DSML grammar: `<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="...">`, `<｜DSML｜parameter name="..." string="true|false">value</｜DSML｜parameter>`. The delimiter is U+FF5C. Upstream `ml-explore/mlx-lm` PR 1337 confirms: `string="true"` keeps the value as text, `string="false"` decodes it as JSON, and one block can hold many invokes.
    - DeepSeek's own `encoding/encoding_dsv4.py` makes `thinking_mode` a REQUIRED parameter with no default, and it has a `parse_tool_calls` read side. Thus the ReasoningConfig row must pick a default deliberately and record why.
    - The tool-call streaming path (`ToolCallProcessor`) sends a full `<start>...<end>` block to `parse`, which returns ONE call; `parseEOS` recovers all calls. `MiniMaxM3ToolCallParser` is the precedent for a one-wrapper-many-invokes format; `KimiK2ToolCallParser` gives the file shape.
    - Wiring plan: the detection rule is the type-registry entry. `LLMModel` already carries per-model hooks (`messageGenerator(tokenizer:)`, `missingChatTemplateRefusal`); the encoder wiring follows that pattern. A wrapping tokenizer in `Tokenizer.swift` builds the prompt with the encoder; the raw-message-to-encoder-message mapping lives in `Chat.swift` beside the code that writes those dictionary keys; `ChatSession` reaches the encoder through `processor.prepare`, and a test must prove that path.
    - I checked the upstream sources the card names: neither `osaurus-ai/vmlx-swift-lm` nor the `scouzi1966/maclocal-api` patch set puts DeepSeek-V4 code in `Tokenizer.swift`, `Chat.swift`, or `ChatSession.swift`. The card's own provenance rule applies: treat the shared files as candidates and change only what a test demands.
  timestamp: 2026-08-12T07:52:27.053218+00:00
- actor: claude-code
  id: 01kztg33m5mdfm8hgsr56z2f5e
  text: |-
    ### Implementation record (2026-08-12)

    The work went in five TDD cycles. Each test failed first for the correct reason, then the code made it pass.

    **1. Type registry.** `"deepseek_v4": create(DeepSeekV4Configuration.self, DeepSeekV4Model.init)` in `LLMTypeRegistry.shared`. Tests: `DeepSeekV4RegistryTests` (contains + createModel with the synthetic checkpoint). The synthetic `config.json` moved to a shared fixture, `Tests/MLXLMTests/DeepSeekV4SyntheticCheckpoint.swift`, and `DeepSeekV4PromptFallbackTests` now reads it from there — one copy, two suites.

    **2. Model registry.** `LLMRegistry.deepseekV4Flash4bit` for `mlx-community/DeepSeek-V4-Flash-4bit`, in `all()`. Test in `LLMRegistryTests`.

    **3. Reasoning row.** `(.exact, "deepseek_v4", deepSeekV4ThinkConfig)`: `<think>`/`</think>`, strategy `.templateFlag(key: "thinking", defaultOn: true)`. The decision is recorded in the code comment on `deepSeekV4ThinkConfig`: the reference encoder takes a REQUIRED `thinking_mode` with two explicit values, thus `.alwaysOn` would wrongly reject a disable request; `defaultOn: true` keeps the family precedent of the V3/R1 rows while it adds the off switch. `ReasoningHeuristics` gained the `"deepseek-v4"` marker. Tests in `ReasoningConfigTests` and `ReasoningHeuristicsTests`.

    **4. DSML read side.** `ToolCallFormat.dsml` (raw value `dsml`), exact `deepseek_v4` row, and `Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift`. `parse` gives the first invoke of a block (the shape of `MiniMaxM3ToolCallParser`, the precedent for one wrapper with many invokes); `parseEOS` recovers a whole parallel round. `string="true"` keeps text; `string="false"` decodes JSON with `.fragmentsAllowed`, because the shared `deserialize` helper rejects bare scalars. Tests: `DSMLToolCallParserTests`, 8 tests, with the streaming `ToolCallProcessor` path included. Sizing: the parser is 154 lines WITH its documentation comments — at the card's approximate 150-line bound, thus no split task.

    **5. Encoder wiring.** The detection rule is the type-registry entry: it routes the checkpoint to `DeepSeekV4Model`, and a new `LLMModel` hook — `promptTokenizer(wrapping:)`, default identity — turns that routing into the encoder path. `DeepSeekV4Model` overrides the hook and returns the new `DeepSeekV4EncodingTokenizer` (in `Tokenizer.swift`), whose `applyChatTemplate` renders the conversation with `DeepSeekV4ChatEncoder` and reads `additionalContext["thinking"]` to pick the mode — the same key and default as the reasoning row. The raw-dictionary-to-encoder-message mapping (roles, `tool_calls`, `tool_call_id`, `reasoning_content`, tool specs onto the system turn) lives in `Chat.swift` beside the writers of those keys. `LLMModelFactory._load` installs the hook's result once, thus the processor, the `ModelContext` tokenizer and `ChatSession` all speak through it.

    **ChatSession.swift needed no edit, and that is deliberate.** `ChatSession` builds prompts only through `processor.prepare`, which reaches the wrapper. The card's own provenance rule — "treat the rest as candidates and change only what a test demands" — governs, and no test demands a `ChatSession.swift` change. The end-to-end proof is `DeepSeekV4EncoderWiringTests/chatSessionUsesTheEncoder`: a real (synthetic, 2-layer) `DeepSeekV4Model` runs a full `ChatSession.respond` round, and the recorded prompt is the encoder's output — begin-of-sentence marker, `<｜User｜>Hello`, assistant tail. I also verified the upstream sources the card names: neither `osaurus-ai/vmlx-swift-lm` nor the `maclocal-api` patch files put DeepSeek-V4 code into `Chat.swift`, `Tokenizer.swift` or `ChatSession.swift`. If the reviewer wants a literal `ChatSession.swift` call site anyway, say so and I will add one — but it would be dead code beside the processor path.

    **Guards kept.** The `missingChatTemplateRefusal` stays on `DeepSeekV4Model` and still throws on the non-encoder path (`DeepSeekV4PromptFallbackTests` stays green). Attribution: the new files are original work against public references, thus they carry the repository's standard header plus reference links, like every other parser; the `jhk0apk` ported-file header applies to transcriptions only (the encoder already carries it).

    **Test evidence.** `swift test --filter 'DeepSeekV4|DSML|LLMRegistryTests|ReasoningConfigTests|ReasoningHeuristicsTests|ToolCall|ToolTests|ChatSession|TokenizerTokenIdTests|TokenizerGenerationPromptTests|MessageGeneratorReasoningTests|UserInputTests|VLMRegistryTests|Ministral3RegistryTests'` — 255 tests in 18 suites (MLXLMTests) plus 28 tests in 5 suites (MLXFoundationModelsTests), 0 failures. `swift build` gives no compiler warning (the one `missing creator for mutated node` line is a pre-existing build-system message, present before this card's first edit).
  timestamp: 2026-08-12T08:05:47.781928+00:00
- actor: claude-code
  id: 01kztg3d5km3b1mv7vcpba2pnh
  text: |-
    ### implement — changed
    - evidence: 11 files — Libraries/MLXLLM/LLMModelFactory.swift, Libraries/MLXLLM/LLMModel.swift, Libraries/MLXLLM/Models/DeepSeekV4.swift, Libraries/MLXLMCommon/ReasoningConfig.swift, Libraries/MLXLMCommon/ReasoningHeuristics.swift, Libraries/MLXLMCommon/Tool/ToolCallFormat.swift, Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift (new), Libraries/MLXLMCommon/Tokenizer.swift, Libraries/MLXLMCommon/Chat.swift, Tests: DeepSeekV4RegistryTests.swift (new), DSMLToolCallParserTests.swift (new), DeepSeekV4EncoderWiringTests.swift (new), DeepSeekV4SyntheticCheckpoint.swift (new), plus edits to LLMRegistryTests.swift, ReasoningConfigTests.swift, ReasoningHeuristicsTests.swift, DeepSeekV4PromptFallbackTests.swift. Filter run: 255 + 28 tests, 0 failures, 0 compiler warnings. No task split — the parser stayed at the ~150-line bound.
    - next: /review
  timestamp: 2026-08-12T08:05:57.555802+00:00
- actor: claude-code
  id: 01kztgwvzywtmy33xtnspmdapz
  text: |-
    ### review — clean
    - evidence: engine op `review sha HEAD~1..HEAD` (commit fdfdecf). Counts: 8 files attempted, 5 findings, 5 confirmed, 1 refuted, 0 failed, 0 skipped. All 5 findings are on lines that fdfdecf did not change: LLMModel.swift:50, LLMModel.swift:81, Chat.swift:19, ToolCallFormat.swift:219, ToolCallFormat.swift:344. Zero findings on changed lines. Each untouched-line item is recorded out of scope in the dated section, per the standing rule of ^ag7ant0. No prior findings section was open. The recorded deviation is correct: ChatSession.swift needed no edit, because ChatSession reaches the encoder through processor.prepare, and DeepSeekV4EncoderWiringTests/chatSessionUsesTheEncoder shows the full path.
    - next: none. The task moved to done.
  timestamp: 2026-08-12T08:19:51.934745+00:00
- actor: claude-code
  id: 01kztgxp1v4cp8bwgx776t7syp
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 9 library files (1 new: DSMLToolCallParser.swift), 8 test files (4 new); registries, reasoning row, ToolCallFormat.dsml, promptTokenizer(wrapping:) encoder wiring; all test-first
    - test: green — swift test, exit 0, 246 XCTests + 845 Swift Testing tests, 0 failures
    - commit: fdfdecf
    - review: clean — 5 engine findings, 0 on changed lines; all 5 recorded out of scope per the standing rule from ^ag7ant0; task moved to done
  timestamp: 2026-08-12T08:20:18.619621+00:00
depends_on:
- 01KZGMVSEEHGCCG1W8CPWR8R3H
- 01KZGN95NRBQ3PEBHPN35AW7VY
position_column: done
position_ordinal: ee80
title: 'Wire deepseek_v4 into the registries: type, reasoning, tool format'
---
## What

Make `model_type == "deepseek_v4"` actually loadable end to end. Today it fails fast: `LLMTypeRegistry.shared` (`Libraries/MLXLLM/LLMModelFactory.swift:26-90`) has no `deepseek_v4` entry, so `ModelTypeRegistry.createModel` throws `ModelFactoryError.unsupportedModelType` at `Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift:29`.

Changes:

1. **`Libraries/MLXLLM/LLMModelFactory.swift`** — add to `LLMTypeRegistry.shared`, alongside the existing `"deepseek_v3"` at line 53:
   `"deepseek_v4": create(DeepseekV4Configuration.self, DeepseekV4Model.init),`
   (matches the reference's own registration at its `LLMModelFactory.swift:50`).
2. **`Libraries/MLXLLM/LLMModelFactory.swift`** — add an `LLMRegistry` `ModelConfiguration` for `mlx-community/DeepSeek-V4-Flash-4bit` and include it in `all()` (see how `deepseekR14bit` is declared at line 385 and listed at line 551).
3. **`Libraries/MLXLMCommon/ReasoningConfig.swift`** — add a `deepseek_v4` entry. The existing table already has `(.exact, "deepseek_v3", alwaysOnThinkConfig)` and `(.exact, "deepseek_r1", ...)` at lines 284-285. DSV4 has explicit `chat` vs `thinking` modes rather than always-on `<think>`, so pick the strategy deliberately and document why — do not just copy the V3 row.
4. **`Libraries/MLXLMCommon/ReasoningHeuristics.swift`** — extend `reasoningModelMarkers` (line 21) if the repo-id heuristic should recognize DSV4.
5. **`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`** — add a DSML case for DeepSeek-V4's native tool format and map `deepseek_v4` to it, plus a parser under `Libraries/MLXLMCommon/Tool/Parsers/`. Follow the shape of the existing `KimiK2ToolCallParser.swift`. Cross-reference `ml-explore/mlx-lm` PR 1337, which adds the equivalent `deepseek_dsml` parser upstream.

Sizing note: if the DSML tool parser turns out to be more than ~150 lines, split it into its own task rather than growing this one past the file/subtask limits.

## Added Work: wire the DeepSeek-V4 chat encoder into the chat path (moved from `^gbsaqc2` on 2026-08-11)

The encoder is already written and proved. `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`
gives the public type `DeepSeekV4ChatEncoder`. Its output is byte-identical to
DeepSeek's own Python, `encoding/encoding_dsv4.py`. But no code calls it.

Call it from these three files:

- `Libraries/MLXLMCommon/Chat.swift`
- `Libraries/MLXLMCommon/Tokenizer.swift`
- `Libraries/MLXLMCommon/ChatSession.swift`

**Why this item is on this card.** A caller must first know that the loaded
model is DeepSeek-V4. That model-detection rule is the work of this card, item 1
above. The card `^gbsaqc2` cannot make the rule, thus it cannot do the wiring.
`^gbsaqc2` blocks `^35aw7vy`, and `^35aw7vy` blocks this card. The item could
not stay on `^gbsaqc2`, because a card cannot wait for a card that waits for it.
The orchestrator moved the item here to break that circle. No work is dropped.

DeepSeek-V4 ships **no** `chat_template`, neither in `tokenizer_config.json` nor
in `tokenizer.json`. Thus the encoder is the only way to make a correct
DeepSeek-V4 prompt. A DeepSeek-V4 model that goes through the usual chat-template
path gets a wrong prompt.

- [x] `Chat.swift`, `Tokenizer.swift` and `ChatSession.swift` use
      `DeepSeekV4ChatEncoder` for a model that the detection rule of this card
      identifies as `deepseek_v4`. (Chat.swift maps the raw messages, Tokenizer.swift
      holds the encoding tokenizer, and ChatSession reaches the encoder through
      `processor.prepare` — proven end to end by
      `DeepSeekV4EncoderWiringTests/chatSessionUsesTheEncoder`. ChatSession.swift
      itself needed no edit; see the implement comment of 2026-08-12.)
- [x] Every other model keeps its present path. No behavior changes for a model
      that is not DeepSeek-V4.
- [x] Test: a `deepseek_v4` model gets the encoder output; a different model does
      not.

## Provenance
- Registration pattern: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/LLMModelFactory.swift:50` (MIT).
- Patch surface for shared files confirmed by `scouzi1966/maclocal-api` — `Scripts/apply-mlx-patches.sh`, which lists `LLMModelFactory.swift`, `BaseConfiguration.swift`, `Load.swift`, `Evaluate.swift`, `Tokenizer.swift`, `Chat.swift`, `Tool/ToolCallFormat.swift`, `KVCache.swift`, `SwitchLayers.swift` as the DSV4 modification set. Our research showed `BaseConfiguration`/`Load`/`SwitchLayers` need no change (see task `wkv5j6f`); treat the rest as candidates and change only what a test demands.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [x] `LLMTypeRegistry.shared.contains("deepseek_v4") == true`.
- [x] `ModelTypeRegistry.createModel(configuration:modelType:)` with the real DSV4 `config.json` returns a `DeepseekV4Model` instead of throwing.
- [x] `LLMRegistry` exposes a configuration for `mlx-community/DeepSeek-V4-Flash-4bit` and it appears in `all()`.
- [x] `ReasoningConfig.infer(from: "deepseek_v4")` returns a config, and the choice of prompt strategy is justified in a code comment.
- [x] `ToolCallFormat.infer(from: "deepseek_v4")` returns the DSML case, and the parser extracts a tool call from a fixture.
- [x] No existing registry test regresses.

## Tests

- [x] Extend `Tests/MLXLMTests/LLMRegistryTests.swift`: assert the new DSV4 configuration's id and default prompt.
- [x] New test in `Tests/MLXLMTests/DeepSeekV4RegistryTests.swift` (canonical spelling): `contains("deepseek_v4")` is true; `createModel` with the checked-in `config.json` fixture succeeds.
- [x] Extend `Tests/MLXLMTests/ReasoningConfigTests.swift` with a `deepseek_v4` case mirroring `inferDeepSeekV3TypeAloneIsAlwaysOn`.
- [x] New test: DSML parser extracts name plus arguments from a fixture tool-call string, and returns nothing for a non-tool string.
- [x] Run: `swift test --filter 'DeepSeekV4|DSML|LLMRegistryTests|ReasoningConfigTests|ToolCall'` — all pass (suite names use the canonical DeepSeek spelling).

## Workflow
- Use `/tdd` — the registry `contains`/`createModel` test fails today for the right reason; start there.
#deepseek-v4

## Review Findings (2026-08-12 03:13)

Scope: `HEAD~1..HEAD` (commit `fdfdecf`). The engine examines full files. The standing rule of `^ag7ant0` applies: only a finding on a changed line keeps the task in review. Each item below was compared with the diff of `fdfdecf`. All five items are on lines that `fdfdecf` did not change, thus each item is out of scope for this task.

- [x] `Libraries/MLXLLM/LLMModel.swift:50` — Magic numbers should be replaced by named constants. — Out of scope: untouched line. The line is in the pre-existing `prepare` step. The changed lines of this file are 24-36 and 90-97.
- [x] `Libraries/MLXLLM/LLMModel.swift:81` — public declarations should be documented. — Out of scope: untouched line. The declaration is the pre-existing `messageGenerator(tokenizer:)`. The new `promptTokenizer(wrapping:)` hook at lines 91-97 has documentation.
- [x] `Libraries/MLXLMCommon/Chat.swift:19` — type `Message` is a near-duplicate of `Chat` at Libraries/MLXLMCommon/Chat.swift:11 (414 tokens, 99% alike). — Out of scope: untouched line. `Chat.Message` is a pre-existing type. The changed lines of this file are 3-4 and 346-447.
- [x] `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:219` — Magic numbers should be replaced by named constants. — Out of scope: untouched line. The line is in the pre-existing `generateToolCallID`. The changed lines of this file are 157-161, 206-207 and 298-305.
- [x] `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:344` — Magic numbers should be replaced by named constants. — Out of scope: untouched line. The line is in the pre-existing `inferLlamaFormat`.