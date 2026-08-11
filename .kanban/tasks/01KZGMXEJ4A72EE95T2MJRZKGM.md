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
depends_on:
- 01KZGMVSEEHGCCG1W8CPWR8R3H
- 01KZGN95NRBQ3PEBHPN35AW7VY
position_column: todo
position_ordinal: '8980'
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

- [ ] `Chat.swift`, `Tokenizer.swift` and `ChatSession.swift` use
      `DeepSeekV4ChatEncoder` for a model that the detection rule of this card
      identifies as `deepseek_v4`.
- [ ] Every other model keeps its present path. No behavior changes for a model
      that is not DeepSeek-V4.
- [ ] Test: a `deepseek_v4` model gets the encoder output; a different model does
      not.

## Provenance
- Registration pattern: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLLM/LLMModelFactory.swift:50` (MIT).
- Patch surface for shared files confirmed by `scouzi1966/maclocal-api` — `Scripts/apply-mlx-patches.sh`, which lists `LLMModelFactory.swift`, `BaseConfiguration.swift`, `Load.swift`, `Evaluate.swift`, `Tokenizer.swift`, `Chat.swift`, `Tool/ToolCallFormat.swift`, `KVCache.swift`, `SwitchLayers.swift` as the DSV4 modification set. Our research showed `BaseConfiguration`/`Load`/`SwitchLayers` need no change (see task `wkv5j6f`); treat the rest as candidates and change only what a test demands.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `LLMTypeRegistry.shared.contains("deepseek_v4") == true`.
- [ ] `ModelTypeRegistry.createModel(configuration:modelType:)` with the real DSV4 `config.json` returns a `DeepseekV4Model` instead of throwing.
- [ ] `LLMRegistry` exposes a configuration for `mlx-community/DeepSeek-V4-Flash-4bit` and it appears in `all()`.
- [ ] `ReasoningConfig.infer(from: "deepseek_v4")` returns a config, and the choice of prompt strategy is justified in a code comment.
- [ ] `ToolCallFormat.infer(from: "deepseek_v4")` returns the DSML case, and the parser extracts a tool call from a fixture.
- [ ] No existing registry test regresses.

## Tests

- [ ] Extend `Tests/MLXLMTests/LLMRegistryTests.swift`: assert the new DSV4 configuration's id and default prompt.
- [ ] New test in `Tests/MLXLMTests/DeepseekV4RegistryTests.swift`: `contains("deepseek_v4")` is true; `createModel` with the checked-in `config.json` fixture succeeds.
- [ ] Extend `Tests/MLXLMTests/ReasoningConfigTests.swift` with a `deepseek_v4` case mirroring `inferDeepSeekV3TypeAloneIsAlwaysOn`.
- [ ] New test: DSML parser extracts name plus arguments from a fixture tool-call string, and returns nothing for a non-tool string.
- [ ] Run: `swift test --filter 'DeepseekV4Registry|LLMRegistryTests|ReasoningConfigTests|ToolCall'` — all pass.

## Workflow
- Use `/tdd` — the registry `contains`/`createModel` test fails today for the right reason; start there.
#deepseek-v4