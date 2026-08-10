---
assignees:
- claude-code
depends_on:
- 01KZGMN4FQKCVPAYJDJJHK0APK
position_column: todo
position_ordinal: '8880'
title: Port DeepseekV4ChatEncoder core rendering (no chat_template ships with DSV4)
---
## What

Create `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` with the **core prompt rendering** path. Tool-call/DSML encoding is split into a follow-on task (`<dsml-encode>`) because the reference file is 48961 bytes (~1200+ lines), well past the 500-line guideline.

**This is required, not optional**: DeepSeek-V4-Flash ships **no `chat_template` in `tokenizer_config.json`** — DeepSeek distributes prompt construction as a separate Python file (`encoding/encoding_dsv4.py`, ~744 LOC). Without this port the model cannot be prompted correctly at all.

Note: this task depends only on the licensing decision, not on the config port — it is tokenizer/chat plumbing and can proceed in parallel with the model work.

Port the non-tool portion of `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`. `osaurus-ai/vmlx-swift-lm` additionally ships a fallback Jinja template at `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja` — consider carrying that as a secondary path.

In scope here:
- `chat` vs `thinking` modes.
- `reasoning_effort=max` preface.
- `drop_earlier_reasoning` multi-turn rule. Its "forced off when tools are present" interaction belongs to the follow-on task, but leave the hook for it.
- `latest_reminder` role and `developer` role.
- Special-token handling: the DeepSeek user/assistant turn markers and `<think>` must be treated as specials by the streaming detokenizer, not split. EOS token id is **1**, decoding to the DeepSeek end-of-sentence token; verify `TokenizerBridge` picks that up from `eos_token_id` in `tokenizer_config.json` rather than assuming it.

Out of scope (follow-on task): DSML tool-call encoding, `tool_result` merging into user `contentBlocks`, `sort_tool_results_by_call_order`.

Wire into existing plumbing: `Libraries/MLXLMCommon/Chat.swift`, `Libraries/MLXLMCommon/Tokenizer.swift`, `Libraries/MLXLMCommon/ChatSession.swift`.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` (MIT; header attributes Osaurus AI).
- Fallback template: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`.
- Original Python source of truth: DeepSeek's `encoding/encoding_dsv4.py` shipped in the model repo.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` exists and renders: plain chat, thinking mode, `reasoning_effort=max`, and multi-turn with `drop_earlier_reasoning`.
- [ ] `developer` and `latest_reminder` roles render.
- [ ] Rendered output is **byte-identical** to the Python reference on every fixture — not "looks right".
- [ ] Turn markers and `<think>` survive detokenization as single special tokens.
- [ ] `eos_token_id == 1` resolves from `tokenizer_config.json`.
- [ ] A documented extension point exists for the tool-encoding follow-on.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4ChatEncoderTests.swift` with checked-in fixtures: input conversation JSON plus expected rendered prompt strings generated from DeepSeek's Python `encoding_dsv4.py`.
- [ ] Test: at least 6 fixture conversations covering each in-scope behavior render byte-identically.
- [ ] Test: `drop_earlier_reasoning` on a 3-turn conversation drops the expected earlier reasoning block.
- [ ] Test: `eos_token_id == 1` resolves from a fixture `tokenizer_config.json` and decodes to the DeepSeek end-of-sentence token.
- [ ] Test: turn markers and `<think>` round-trip through the detokenizer unsplit.
- [ ] Run: `swift test --filter DeepseekV4ChatEncoderTests` — all pass.

## Workflow
- Use `/tdd` — generate the Python-rendered expected strings first and assert byte equality; anything looser lets a subtle spacing bug through.
#deepseek-v4