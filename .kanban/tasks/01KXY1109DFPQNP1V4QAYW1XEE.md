---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kya9ynwqz7ekm0rpcaf511ck
  text: |-
    Milestone: MLXLMCommon + MLXFoundationModels wiring done and green.

    Verified ground truth (curl, not WebFetch summaries, which hallucinated the namespace token slightly differently each time):
    - Real chat_template.jinja from mlx-community/MiniMax-M3-4bit (fetched via HF resolve URL).
    - Real minimax_m3.py parser + tests from mlx-lm PR #1416 (fetched via patch-diff.githubusercontent.com raw diff).

    Confirmed: M3 tool-call format is `]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="...">]<]minimax[>[<key>value]<]minimax[>[</key>...]<]minimax[>[</invoke>]<]minimax[>[</tool_call>` — namespace token `]<]minimax[>[` prepended to every tag, arbitrary `<key>value</key>` children (mappings -> nested tags, iterables -> `<item>` children). Confirmed M3 reasoning is `<mm:think>`/`</mm:think>` gated by a template `thinking_mode` kwarg taking "enabled"/"disabled"/"adaptive" (NOT boolean) — pre-opens `<mm:think>` in the generation prompt only when forced "enabled", closes immediately when "disabled", no priming when adaptive/unset.

    Implemented:
    1. `Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM3ToolCallParser.swift` (new) — hand-rolled recursive XML-fragment parser (no XMLParser/XMLDocument dependency, matching M2/XMLFunctionParser conventions), ported from mlx-lm PR #1416's `minimax_m3.py` including the `__parse_error__` surfacing for malformed/non-object argument XML (rather than silently degrading to `{}`).
    2. `ToolCallFormat.swift` — new `.minimaxM3` case (rawValue `minimax_m3`), `createParser()` wiring, inference-table row (`.prefix, "minimax_m3"`) covering both `minimax_m3` and `minimax_m3_vl`.
    3. `ReasoningConfig.swift` — new `ReasoningPromptStrategy.templateStringFlag(key:onValue:offValue:defaultValue:)` case (M3's `thinking_mode` is a 3-valued string kwarg, not a boolean — `.templateFlag` couldn't represent it faithfully: sending a Bool would never match the template's string comparisons and thinking would silently always render as "adaptive" regardless of caller intent). Added `minimaxM3ThinkConfig` + inference-table row.
    4. `MLXLanguageModel.swift` — extended the two think-then-call gate switches (`makeThinkThenCallConfig`, `thinkThenCallPhase1Engages`) to treat `.templateStringFlag` identically to `.templateFlag` (Phase 1 always engages regardless of prompt priming — matches Qwen3's "model may choose to think" semantics, verified via `reasoningPrimedInside`'s generic literal-scan detection working correctly for M3's conditional priming without any special-casing).
    5. `TranscriptConverter.swift` — added `.minimaxM3` to `structuredToolCallFormats`.

    Tests (all green, `swift test` full suite passes — one Gemma4ChunkedPrefillTests flake observed on one run, unrelated to this change, not reproduced on 2 subsequent full runs):
    - ToolTests.swift: 9 new MiniMaxM3 parser tests (single/multi param, nested array-of-objects, conversational prefix, no-call, malformed XML, non-object args, ToolCallProcessor integration, EOS multi-invoke) + rawValue/inference assertions.
    - ReasoningConfigTests.swift: templateStringFlag unit tests (on/off/unspecified) + minimax_m3/minimax_m3_vl inference tests + explicit VLMModelFactory-contract-pinning tests (see below).
    - ThinkThenCallGateTests.swift: templateStringFlag coverage for both gates.
    - TranscriptConverterTests.swift: M3 mirror tests reusing `firstMiniMaxToolValidatorViolation` (M3's tool-role validator in chat_template.jinja is byte-identical logic to M2's).

    VLMModelFactory verification decision: `_load`'s reasoning-inference block (`if mutableConfiguration.reasoningConfig == nil { mutableConfiguration.reasoningConfig = ReasoningConfig.infer(from: baseConfig.modelType, modelID: configuration.name, configData: configData) }`) is a generic, model-type-agnostic passthrough — verified by reading the code, unchanged by this task. Per `ReasoningConfigResolutionTests.swift`'s own documented precedent ("Sites 4-5 ... are verified end-to-end by the on-device reasoning integration tests"), the `_load` glue itself is not synthetic-weights unit tested anywhere in this codebase (same for the earlier qwen3_5 VLM onboarding). Added `vlmModelFactoryResolvesMiniMaxM3VLReasoningRow` + `vlmModelFactoryNoMatchingRowResolvesNil` to ReasoningConfigTests.swift, explicitly documenting and pinning this contract at the `ReasoningConfig.infer` level (the actual logic `_load` delegates to) rather than attempting a full synthetic-checkpoint `_load()` round trip (would require fabricating on-disk safetensors matching MiniMaxM3Model's shapes — disproportionate for a two-line generic passthrough with no model-specific logic).

    Next: chat-template probe test (real M3 template via swift-jinja `Template` directly — found the `Jinja.Template(_:with:)` + `Value(any:)` API, avoids needing a full PreTrainedTokenizer/vocab) + gated tool-round-trip integration test mirroring MiniMaxM3CoherenceIntegrationTests.swift.
  timestamp: 2026-07-24T14:54:58.455013+00:00
- actor: claude-code
  id: 01kyaawj4898263f2g40cej5j3
  text: |-
    Milestone: IntegrationTesting pieces landed.

    1. `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MiniMaxM3ChatTemplateProbeTests.swift` (new) — renders the REAL M3 chat_template.jinja (checked in verbatim as a raw Swift string literal, fetched via curl from the mlx-community/MiniMax-M3-4bit HF repo) directly through swift-jinja's `Template(_:with:)` + `Value(any:)` API — no PreTrainedTokenizer/vocab needed, since applyChatTemplate is just Template.render under the hood. 4 tests, all run and PASS against the real template: system+user+tool-call+tool-result+assistant round renders the namespaced `<tool_call>`/`<invoke>` wrapper correctly; plain no-tools render works; `thinking_mode: "enabled"`/`"disabled"` correctly prime the generation prompt with `<mm:think>`/`</mm:think>`; `"adaptive"` primes nothing. This directly exercises and validates the exact contract `ReasoningConfig.minimaxM3ThinkConfig` and the `.minimaxM3` tool-call format depend on.

    2. `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/MiniMaxM3ToolCallingIntegrationTests.swift` (new) — gated real-weights tool round trip (toolCalls -> toolOutput -> response) mirroring `MiniMaxM3CoherenceIntegrationTests`'s exact gating convention (220GB memory floor, `MLX_MINIMAX_M3_CHECKPOINT` override, graceful skip on load failure). `xcodebuild build-for-testing` succeeds. Attempted an actual run (this sandbox has 512GB RAM, so the memory gate passes and it proceeds to load): after 180s only a lock file existed in the HF cache (`~/.cache/huggingface/hub/.locks/models--mlx-community--MiniMax-M3-4bit`), no snapshot/blob data — confirms the same ~120-214GB-checkpoint bandwidth impracticality documented by the prior task (^wz8y8qq's coherence test). Did NOT complete to a pass/fail verdict in this session, same as that precedent.

    Full `swift test` suite: green (335/335 in the MLXLMTests bundle, 80/80 MLXFoundationModelsTests, 282/282 MLXGuidedGenerationTests, 7/7 others — 2 fresh full runs, no failures). No files touched outside this task's scope (confirmed via git status: Qwen25VL.swift, Gemma4.swift, MiniMax.swift/M2 untouched).

    Adversarial double-check agent launched to review the diff before handoff.
  timestamp: 2026-07-24T15:11:17.640323+00:00
- actor: claude-code
  id: 01kyac15s9r8hmxvbxdytvsar6
  text: |-
    Adversarial double-check found a genuine bug in `MiniMaxM3ToolCallParser.parseElements`: a scalar argument value merely containing a stray `<`/`>` (e.g. `"3 < 5"`) was misread as a broken tag and flagged `__parse_error__` instead of round-tripping as plain text.

    Fixed: added `isPlausibleTagName(_:)` (tag-name candidates must look like real identifiers: start with a letter/underscore, then alphanumeric/`_`/`-`) and made two of the parser's failure branches lenient when no sibling element has parsed successfully yet in the current fragment (`results.isEmpty ? [] : nil` instead of always `nil`): (1) no closing `>` found at all for a `<`, and (2) the candidate tag name isn't a plausible identifier. This correctly handles natural-language comparisons like `"3 < 5"` and `"3 < 5 > 2"` as scalars.

    Deliberately did NOT extend the same leniency to "a plausible-looking tag name that never finds its matching close tag" (e.g. `"List<String>"`) — verified that upstream's own reference parser (`ET.fromstring` in Python) would ALSO raise `ParseError` for that case, since unescaped `<`/`>` forming what looks like real nested tag structure is genuinely invalid XML per spec; M3's chat template renders leaf values via unescaped `{{ val }}`, so this is a latent, pre-existing quirk in the upstream template/parser pairing, not something this Swift port should diverge from to appear more lenient than the reference it's verified against. Added a regression test covering both fixed cases (`testMiniMaxM3ParserStrayAngleBracketsInScalarValue` in ToolTests.swift). Verified the original "malformed" (`<command>...` unclosed) and "non-object" (`<item>a</item>`) tests still fail correctly (unchanged).

    Full `swift test` green (336/336 in the affected bundle) and `xcodebuild build-for-testing`/the chat-template probe test re-verified green after the fix.

    Other double-check findings: (2) mixed loose-text + child-tag fragments could silently drop text -- rated low priority/optional by the reviewer, not fixed (extremely low real-world likelihood given `to_xml`'s actual output shape, and explicitly marked optional). (3) VLMModelFactory verification test calls `ReasoningConfig.infer` directly rather than `_load` end-to-end -- already disclosed and justified in the prior comment (matches the qwen3_5 precedent; not a hidden gap). No scope creep: git status confirms Qwen25VL.swift, Gemma4.swift, MiniMax.swift (M2) untouched.

    Task remains in `doing`, ready for `/review`.
  timestamp: 2026-07-24T15:31:17.417220+00:00
depends_on:
- 01KXY0ZVCCPBKZ1ANETWZ8Y8QQ
position_column: doing
position_ordinal: '80'
title: 'MiniMax-M3: tool calling + reasoning wiring through FoundationModels'
---
## What

Wire M3 into the tool-calling and reasoning inference tables, to the same bar the M2 task (^9mv1q33) met:

1. **`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`**: add an inference-table row for model_type `minimax_m3_vl` (and the bare `minimax_m3` text variant id if mlx-vlm/model cards use it). FIRST verify against the real repo's `chat_template.jinja` (fetch from `mlx-community/MiniMax-M3-4bit`) whether M3 uses M2's tool-call format (`.minimaxM2`, parsed by `MiniMaxM2ToolCallParser`) or a new one; if new, add a parser following `Libraries/MLXLMCommon/Tool/Parsers/` conventions. Do not assume — the M2 task's failure mode was exactly a template/render mismatch. (See the ^a7swy35 folded findings below: the format was already verified to be NEW, not M2's.)
2. **`Libraries/MLXLMCommon/ReasoningConfig.swift`**: add an inference row for `minimax_m3_vl` (M3 is an interleaved-thinking model — see the ^a7swy35 folded findings below: the template was verified to use toggleable `<mm:think>`, NOT M2's always-on `<think>`).
3. **`Libraries/MLXVLM/VLMModelFactory.swift`** — the `ReasoningConfig.infer` wiring into `VLMModelFactory._load` was ALREADY IMPLEMENTED by task ^05zt40g (commit 5891f01, landed for the qwen work): the factory now infers and threads `reasoningConfig` into the VLM-built `ModelConfiguration`, mirroring the LLM factory. For this task, VERIFY the existing inference handles `minimax_m3_vl`'s new row correctly — the factory wiring itself is done; do not re-implement it.
4. **`Libraries/MLXFoundationModels/TranscriptConverter.swift`**: confirm the structured tool-call rendering path added for `.minimaxM2` covers M3's template validation (tool message must follow assistant `tool_calls`); extend if the format differs.

### Folded from ^a7swy35 (chain reconciliation 2026-07-22)

Verified format findings (against the checkpoint's `chat_template.jinja` and upstream mlx-lm PR #1416, which adds a separate `minimax_m3.py` parser) — M3's formats are NOT M2's:

- **Tool calls**: namespaced XML with parameters as arbitrary `<key>value</key>` children — unlike M2's `<parameter name="k">v</parameter>`. The M2 parser cannot be reused: add `Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM3ToolCallParser.swift` following the M2 parser + `ParserUtilities.swift` conventions, and a new `ToolCallFormat` case (e.g. `.minimaxM3`) registered for the `minimax_m3` / `minimax_m3_vl` model types. Take parser fixture strings verbatim from the chat template / PR #1416 test cases; cover multi-parameter and nested-value cases, and match the streaming/partial parse contract used by the M2 parser tests.
- **Reasoning tags**: the template uses toggleable `<mm:think>` — NOT M2's always-on `<think>` (`ReasoningConfig.swift:201-208`). Wire the `<mm:think>` delimiters with the correct toggle semantics as expressed by the template, not `.alwaysOn`.
- **MLXFoundationModels**: add the new case to `structuredToolCallFormats` in `Libraries/MLXFoundationModels/TranscriptConverter.swift:218` (currently `[.mistral, .minimaxM2]`) so M3 tool turns replay through the chat template's `tool_calls` branch.
- **Chat-template probe**: the M3 template uses macros and namespace-token concatenation that may stress swift-transformers' Jinja engine — add an `ApplyChatTemplateProbeTests`-style test that renders the real M3 chat template (checked-in fixture) with a system+user+tool conversation, catching engine gaps before the real-weights integration test.
- Keep all M2 behavior unchanged.

### Folded from ^b90razv (chain reconciliation 2026-07-22)

- The gated tool-round-trip integration test follows the same runner conventions as ^wz8y8qq's coherence test: it lives in `IntegrationTesting/` (Xcode project, run via `xcodebuild` — NOT `swift test`), is gated by the `DeviceTier.swift` convention, allows overriding the checkpoint source via environment variable, and skips gracefully (not fails) when the checkpoint is absent or memory is insufficient.

## Acceptance Criteria

- [ ] `ToolCallFormat.infer` and `ReasoningConfig.infer` return the verified format/config for model_type `minimax_m3_vl` (unit tests pin both)
- [ ] The existing `VLMModelFactory._load` reasoning inference (landed in commit 5891f01, task ^05zt40g) is VERIFIED to resolve `minimax_m3_vl`'s row into the VLM-built `ModelConfiguration.reasoningConfig` (unit test pins it; regression: a VLM model with no matching infer row still loads with `reasoningConfig == nil`)
- [ ] Multi-turn tool exchange renders through M3's real `chat_template.jinja` without TemplateException (template-validator mirror test, same approach as the M2 fix)
- [ ] One real tool-call round trip completes end to end on `mlx-community/MiniMax-M3-4bit` (gated integration test: `toolCalls → toolOutput → response`)
- [ ] Thinking output lands in `.reasoning` entries, never leaking into the text reply (assert in the integration test)
- [ ] M2/Mistral/Qwen/GLM rendering unchanged (existing regression tests stay green)

## Tests

- [ ] Extend `Tests/MLXLMTests/ToolTests.swift` + `Tests/MLXLMTests/ReasoningConfigTests.swift`: inference-table rows for `minimax_m3_vl`
- [ ] New/extended VLM factory test: reasoningConfig threading (no-match → nil; minimax_m3_vl → the M3 config)
- [ ] Extend `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift`: M3 multi-turn tool rendering (mirroring the existing minimax tests)
- [ ] Gated integration: tool round trip + think-leak assertion against real weights (FoundationModels multitool suite pattern)
- [ ] Run: `swift test` → green; integration case passes

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax #minimax-m3