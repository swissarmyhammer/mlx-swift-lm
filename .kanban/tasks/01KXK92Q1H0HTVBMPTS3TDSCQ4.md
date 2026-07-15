---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxkhdss7fkgyq3vczy0x65hc
  text: 'Picked up by /finish (scoped-batch). Previous task xw6t27b landed in done (clean re-review). Driving this via implement→test→commit→review. No prior attempts on record. Scope: derive the tool-calling structural tag from the inferred ToolCallFormat (wire GLM4 at minimum) instead of hardcoding Qwen''s <tool_call> wrapper; keep Qwen path byte-identical; all unit-testable, no real-model E2E required.'
  timestamp: 2026-07-15T18:43:02.055567+00:00
- actor: claude-code
  id: 01kxkhrsz27zs9yjepz2gjwanz
  text: |-
    Research done. Key findings:
    - FoundationModels tool-calling path (MLXLanguageModel.runToolCalling -> encodeToolCallingGrammar) uses a SYNTHETIC JSON envelope {name,arguments} for both generation constraint and history replay; it does NOT use ToolCallFormat.createParser(). Parse side is emitToolCallingEvent -> unwrapToolCallMarkers (hardcodes <tool_call>, trims whitespace) + JSON parse. So the only degree of freedom is the WRAPPER delimiters.
    - GLM-4.7 (glm4_moe -> .glm4) native wrapper is <tool_call>/</tool_call> with NO inner newlines (function content directly follows), confirmed by GLM4ToolCallParser fixtures in Tests/MLXLMTests/ToolTests.swift. Qwen uses <tool_call>\n...\n</tool_call>. That's the genuine, defensible difference.
    - Source of truth for format = context.configuration.toolCallFormat (mirrors parse-side container.configuration.toolCallFormat at MLXLanguageModel:1509 and Evaluate.swift). resolved/RespondSetup is a per-instance override held as a local, NOT the identity source.
    Plan: data-driven seam ToolCallStructuralTag{begin,end}.forFormat(ToolCallFormat?) in SchemaConverter (nil/.json/unmapped -> .qwen byte-identical; .glm4 -> no-newline wrapper). encodeToolCallingGrammar gains format: ToolCallFormat? = nil. Thread context.configuration.toolCallFormat at the call site. unwrapToolCallMarkers unaffected (trims whitespace). Round-trip integration test calls without format -> unchanged.
  timestamp: 2026-07-15T18:49:02.690079+00:00
- actor: claude-code
  id: 01kxkj1v2n067mybehepbame8m
  text: |-
    Implementation landed (TDD: wrote 4 failing tests -> RED via 11 compile errors -> implemented -> GREEN).

    Files changed:
    - Libraries/MLXFoundationModels/GuidedGeneration/SchemaConverter.swift: added nested `ToolCallStructuralTag {begin,end}` value type with static specs `.qwen` (<tool_call>\n / \n</tool_call>, the historical default) and `.glm4` (<tool_call> / </tool_call>, no inner newlines), plus `forFormat(ToolCallFormat?)` seam (.glm4 -> .glm4; everything else incl nil/.json -> .qwen). `encodeToolCallingGrammar` gains `format: ToolCallFormat? = nil` and reads begin/end from the selected tag. Added `import MLXLMCommon`.
    - Libraries/MLXFoundationModels/MLXLanguageModel.swift: runToolCalling now passes `format: context.configuration.toolCallFormat` (mirrors the parse side's source of truth) into encodeToolCallingGrammar.
    - Tests/MLXFoundationModelsTests/ToolCallingSchemaTests.swift: added `import MLXLMCommon`, 4 tests (seam per-format selection; nil default -> Qwen wrapper; .json wrapper == default; .glm4 -> newline-free wrapper) + wrappedArm helper.

    Design note: byte-identical AC2 is enforced at the wrapper level (begin/end strings), NOT full-JSON string equality -- JSONSerialization of [String:Any] has no stable key order, so a full-string == is not a valid invariant (it flaked nil-vs-nil). unwrapToolCallMarkers on the parse side trims whitespace so GLM4's newline-free wrapper round-trips fine.

    Results: swift build exit 0 (only pre-existing deprecation warning at MLXLanguageModel:946). ToolCallingSchemaTests 12/12 pass (4 new + 8 existing). MLXFoundationModelsTests 217/217 pass. ToolTests 64/64 pass. swift-format lint clean on SchemaConverter + test file; MLXLanguageModel warnings all pre-existing and away from my insertion. Formatter run in-place, no changes beyond my edits. Task left in doing.
  timestamp: 2026-07-15T18:53:58.741655+00:00
- actor: claude-code
  id: 01kxkjby9b1xxkkpy6d972pc0s
  text: |-
    Adversarial double-check (advisory) verdict: REVISE — but the sole finding was "no fresh test-run evidence in-session" (the critic did static-only verification and could not see this session's test runs). All substantive points PASSed with file:line evidence: AC1 tag-from-format, source-of-truth (context.configuration IS the same ModelConfiguration container.configuration reads — ModelContainer.configuration = context.read{$0.configuration}; factory infers+threads before context build), AC2 Qwen byte-identical at wrapper level (JSONSerialization key order genuinely unstable, so wrapper-string guard is the correct invariant), AC3 GLM4 matches GLM4ToolCallParser startTag/endTag and unwrapToolCallMarkers round-trips the newline-free wrapper, no other caller needs format, glm4Bare->.qwen fallback acceptable/no regression.

    Finding resolved: fresh run captured — swift test --filter MLXFoundationModelsTests => "Test run with 217 tests in 38 suites passed"; ToolCallingSchemaTests 12/12; ToolTests 64/64; swift build exit 0. AC4 green. Task left in doing for /review.
  timestamp: 2026-07-15T18:59:29.707038+00:00
- actor: claude-code
  id: 01kxkjd10deky60p5ve572trmr
  text: 'Implement landed green in `doing`. Seam: new data-driven `SchemaConverter.ToolCallStructuralTag { begin, end }` with `.qwen` (historical default, `<tool_call>\n`/`\n</tool_call>`) and `.glm4` (newline-free `<tool_call>`/`</tool_call>`, grounded in GLM4ToolCallParser.startTag/endTag) specs, plus `forFormat(_:)` mapping `.glm4 → .glm4`, everything else (incl. nil/.json) → `.qwen`. `encodeToolCallingGrammar` gained `format: ToolCallFormat? = nil`; `MLXLanguageModel.runToolCalling` threads `context.configuration.toolCallFormat` (same source the parse side reads). Adding a family = one case + one spec constant. Tests (model-free): seam mapping, default→Qwen exact wrapper, Qwen/.json regression guard, GLM4 wrapper. Verified: swift build clean; ToolCallingSchemaTests 12/12; MLXFoundationModelsTests 217/217; ToolTests 64/64. Note on AC2: byte-identical enforced at wrapper (begin/end) level, not full serialized-JSON equality (JSONSerialization key order is unstable). Proceeding to checkpoint commit + review.'
  timestamp: 2026-07-15T19:00:05.261318+00:00
position_column: doing
position_ordinal: '80'
title: Tool-calling structural tag should follow the inferred ToolCallFormat, not hardcode Qwen's <tool_call> wrapper
---
## What

`Libraries/MLXFoundationModels/GuidedGeneration/SchemaConverter.swift`'s `encodeToolCallingGrammar(tools:)` hardcodes Qwen's `<tool_call>\n…\n</tool_call>` structural tag (with a bare-JSON alternative) for every model, while the parse side already does the right thing per model: `LLMModelFactory` infers a `ToolCallFormat` from the config's `model_type` (`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift` — `glm4*` → `GLM4ToolCallParser`, `gemma4` → its parser, `lfm2*`, llama3 heuristics, etc.).

So a non-Qwen model's guided tool-calling decode is constrained toward a wrapper format it wasn't trained on, while its chat template presents tools in its own native format — a layering mismatch. The generation constraint should be derived from the same inferred `ToolCallFormat` the parser uses (each format contributing its own structural-tag spec), so both halves of the round trip agree with the model's training.

## Empirical context (2026-07-15, FoundationModelsMultitool gated suite, M3 Ultra)

Observed with `mlx-community/GLM-4.7-Flash-4bit` driven through `MLXLanguageModel`/`LanguageModelSession`: the model *functions* under the Qwen-framed constraint (the bare-JSON alternative gives it an escape hatch — 21 parseable tool calls in one scenario, 2/4 scenario pass rate), so this is a correctness-of-design issue rather than a hard breakage. But the mismatch plausibly costs reliability: the model is decoding against a foreign wrapper grammar its priors don't expect, and every non-Qwen model pays that tax. Qwen-family models (the currently-best-performing pins) are unaffected, which may itself be partially an artifact of this bias.

## Acceptance Criteria

- [ ] `encodeToolCallingGrammar` (or its caller in `MLXLanguageModel`'s tool-calling phase) selects the structural tag from the model's inferred `ToolCallFormat` instead of unconditionally using Qwen's `<tool_call>` wrapper.
- [ ] Qwen-family behavior is unchanged (regression-guard the existing structural tag for `ToolCallFormat` = default/Qwen).
- [ ] At least GLM4's format is wired (its parser already exists); other formats can follow the same seam incrementally.
- [ ] Existing tool-calling tests stay green.
