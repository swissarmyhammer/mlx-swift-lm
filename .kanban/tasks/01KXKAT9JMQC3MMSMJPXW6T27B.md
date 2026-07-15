---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxkb93rdj9k9yf9xrgnd7cvj
  text: Picked up by /finish (scoped-batch, no scope). Review column empty; first ready todo. Driving to done via implement→test→commit→review loop. No prior attempts on record.
  timestamp: 2026-07-15T16:55:36.973730+00:00
- actor: claude-code
  id: 01kxke1eqd7pxvpzf3s0mq41a6
  text: |-
    Root cause confirmed against the real Devstral chat_template.jinja (cached locally at models--mlx-community--Devstral-Small-2-24B-Instruct-2512-4bit). Its alternation validator counts every assistant message WITHOUT tool_calls toward strict user/assistant alternation. TranscriptConverter replayed a `.toolCalls` entry as verbatim assistant TEXT content (no tool_calls field), so a completed tool round rendered as user, assistant(toolcall-text), assistant(answer) -> two counted assistants in a row -> TemplateException "After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results."

    Fix (family-specific, non-Mistral rendering untouched):
    - TranscriptConverter.mlxMessages gains `toolCallFormat:` param. For `.mistral`, a `.toolCalls` entry now renders as a SINGLE assistant message carrying structured `tool_calls` (empty text content) via new `mistralToolCall(from:)` helper -> template excludes it from alternation AND renders `[TOOL_CALLS]name[ARGS]args`. All other formats keep the verbatim-envelope-per-call behavior byte-for-byte.
    - MLXLanguageModel.respond: vision gate switched to entriesWithImages(transcript) so it still fails fast before weight load; messages now built AFTER loadContainer, format-aware from container.configuration.toolCallFormat (documented source of truth).
    - MLXLanguageModel.populatePromptCacheChunks (prewarm): same format-aware rendering (emptiness is format-invariant).

    Tests (Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift, model-free): testMistralToolRoundRendersTemplateAcceptableAlternation, testVerbatimToolCallRenderingViolatesMistralAlternation (regression-motivation guard), testMistralMultipleToolCallsFoldIntoOneAssistantMessage, testNonMistralToolCallRenderingUnchanged (Qwen/GLM guard). Added a firstAlternationViolation helper mirroring the template's exact check. Watched them fail RED (violation at index 3, tool_calls nil) then GREEN.

    Results: `swift build` clean (only pre-existing line-946 deprecation warning). `swift test --filter TranscriptConverterTests` = 26/26 pass. `swift test --filter MLXFoundationModelsTests` = 213/213 pass. Formatted with swift-format.

    Real-template proof (no model weights): ran the actual chat_template.jinja via Jinja2 over both shapes -> OLD verbatim shape raises the EXACT task TemplateException; NEW structured shape renders cleanly (429 chars) with [TOOL_CALLS] and [TOOL_RESULTS].

    Acceptance #2 (real Devstral ~13GB GPU generation E2E) NOT run: gated integration suite requiring model download + Apple-Silicon GPU inference, not feasible in this environment. Criterion #1 (renders without TemplateException) is proven at the template layer end-to-end; #3 (non-Mistral unchanged) is unit-guarded.

    Discovered (out of scope, candidate follow-up): Mistral3MessageGenerator wraps tool-role content in the chunk-array form, so the template's `[TOOL_RESULTS]' + content|string` stringifies a list literal for tool results. Not a TemplateException (no throw) but suboptimal prose for the model; separate from this alternation fix.
  timestamp: 2026-07-15T17:43:51.789477+00:00
- actor: claude-code
  id: 01kxkfs8ykwdcp93e5yn5q2a7w
  text: |-
    Adversarial double-check (advisory) verdict: REVISE, with all three findings explicitly classified as "verification gaps against the stated intent, not defects" — it confirmed the code sound on all five focus areas (vision-gate fail-fast preserved; toolCallFormat source correct/non-nil for Devstral via VLMModelFactory; .mistral branch correct; no non-Mistral regression; no other production call site). Resolutions:

    Finding 3 [Low] (real fix applied): populatePromptCacheChunks previously loaded the container before the empty-messages guard, so an empty transcript would load weights then return. Restored the original early-return-before-load: emptiness is format-invariant, so it now gates on the cheap format-agnostic render BEFORE loadContainer(), then re-renders format-aware after load only when non-empty.

    Finding 1 [Medium] (doc + evidence): the model-free mirror can't apply the real Jinja template in-repo. Strengthened firstAlternationViolation's docstring to quote the real chat_template.jinja validator loop verbatim (the exact `message.tool_calls is not defined/none/length==0` condition and the exact raise_exception message) so a reviewer can audit fidelity, and documented that the mirror is validated out-of-band against the real template (my Jinja2 run: OLD verbatim shape raises exactly this exception; NEW structured shape renders cleanly). The full E2E leg (real tokenizer applying the template during Devstral generation) remains the gated ~13GB+GPU integration test — deferred, see prior comment.

    Finding 2 [Low] (already satisfied): the #available(macOS 27) guard is the file's established pattern. The assertions demonstrably executed on this host — the RED run's failures came from INSIDE the guarded tests (firstAlternationViolation returned 3; tool_calls was nil), which is only reachable past the #available guard. Host is macOS 27 (Darwin 27.0.0). No change needed.

    Re-verified after the two follow-up edits: `swift build` clean; `swift test --filter MLXFoundationModelsTests` = 213/213 pass (includes 26 TranscriptConverterTests). Diff remains focused (3 files, no unrelated formatter churn, no >100-col new lines). Not re-spawning double-check (bounded to one re-check; remaining changes are a comment-only docstring and a strictly-safer guard reordering). Leaving task green in doing for /review.
  timestamp: 2026-07-15T18:14:20.883345+00:00
- actor: claude-code
  id: 01kxkftwbs2b43cvy1mx7em05x
  text: 'Implement landed green in `doing`. Root cause: TranscriptConverter replayed a `.toolCalls` entry as verbatim assistant *text* (no tool_calls field), so a completed tool round rendered as user, assistant(text), assistant(answer) — two counted assistants → Mistral TemplateException. Fix: for `.mistral` format, render `.toolCalls` as a single assistant Chat.Message with empty content + structured `toolCalls:`; all other formats byte-identical. Also made MLXLanguageModel.respond()/populatePromptCacheChunks() format-aware from container.configuration.toolCallFormat. Verified: `swift build` clean; TranscriptConverterTests 26/26; MLXFoundationModelsTests 213/213. Real-model Devstral E2E (AC#2) deferred — ~13GB download + GPU not feasible here; instead proven at template layer by running the real cached chat_template.jinja (old shape raises the exact TemplateException, new shape renders cleanly). Proceeding to checkpoint commit + review.'
  timestamp: 2026-07-15T18:15:13.529057+00:00
- actor: claude-code
  id: 01kxkgy2gaysg94scptsscvabz
  text: |-
    Re-picked from review to doing to work the 2025-07-15 13:16 Review Findings (both in Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift). Root-fixed both:

    Finding 1 (doc summary line): reworked the `firstAlternationViolation` doc comment so the first line is a complete single sentence — `/// Mirrors the alternation validator in Mistral3's chat_template.jinja.` — followed by a blank `///` line then the elaboration; the verbatim ```jinja citation block is untouched. Fixed the SAME defect on its sibling `completedToolRoundEntries` doc comment (also opened with a fragment across lines): now `/// Builds a completed single-round tool exchange as transcript entries.` + blank + body. No other `///` doc comments in the file.

    Finding 2 (trivial per-call invariant): `testNonMistralToolCallRenderingUnchanged` now uses TWO tool calls in one entry ([get_weather, get_time]). Asserts `.json`/`.glm4` each produce TWO assistant messages (count==2, roles [.assistant,.assistant], contents == default, tool_calls nil, envelope names == [get_weather, get_time]) — genuinely proving one-message-per-call — and adds a contrast asserting `.mistral` folds the same two calls into ONE. Kept every prior assertion (strengthened, not weakened). No production code changed.

    Verified: `swift build` clean (only pre-existing docc/resource warnings); `swift test --filter TranscriptConverterTests` = 26/26 pass; `swift test --filter MLXFoundationModelsTests` = 213/213 pass. swift-format lint --strict clean; format --in-place produced no changes beyond the edits. Both Review Findings checkboxes flipped to [x]. Leaving green in doing for /review.
  timestamp: 2026-07-15T18:34:26.698720+00:00
position_column: doing
position_ordinal: '80'
title: Tool-calling transcript rendering violates Mistral3's strict-alternation chat template (Devstral unusable)
---
## What

Driving native tool calling through `MLXLanguageModel`/`LanguageModelSession` with a `mistral3`-family model throws before generation on every multi-turn tool exchange:

```
TemplateException: "After the optional system message, conversation roles must alternate user and assistant"
```

Reproduced 2026-07-15 with `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` (model_type `mistral3`, arch `Mistral3ForConditionalGeneration`) via FoundationModelsMultitool's gated suite (M3 Ultra): all 4 scenarios throw this from the template-rendering step — the model never gets to generate. The tool-calling loop's message sequence (user → assistant tool-call → tool result → assistant …) is rendered in a shape Mistral's strict-alternation Jinja template rejects.

The parse/infer side is already mistral3-aware (`ToolCallFormat.infer` matches the `mistral3` prefix; `MistralToolCallParser` exists), so this is specifically the *prompt-rendering* half: the message generator that builds the chat array from the FoundationModels transcript doesn't produce a sequence Mistral's template accepts (e.g. tool results likely need the `tool` role adjacent to the assistant tool-call turn in the exact shape the template's alternation check permits, or need folding into the surrounding turns the way Mistral's reference clients do).

## Why it matters

Devstral Small 2 is arguably the strongest open agentic-coding model runnable on Apple Silicon (68% SWE-bench Verified, purpose-trained for tool-driven work, 24B dense ~13GB at 4-bit, the most-downloaded Devstral MLX build) — and it is currently 100% unusable for native tool calling through this stack, failing before inference on the very workload it's best at.

## Acceptance Criteria

- [ ] A multi-turn tool-calling exchange (user → assistant tool call → tool output → assistant answer) renders through the mistral3 chat template without TemplateException.
- [ ] Verified end to end with Devstral-Small-2-24B-Instruct-2512-4bit generating at least one real tool-call round trip.
- [ ] Non-Mistral families' rendering unchanged (regression-guard Qwen/GLM paths).

## Review Findings (2026-07-15 13:16)

- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:326` — The first line of a documentation comment should be a single-sentence summary ending in a period. Line 326 is a sentence fragment that continues across multiple lines. Reorganize the first line so that it forms a complete, standalone sentence ending with a period before continuing elaboration, such as: '/// Mirrors Mistral3's strict user/assistant alternation validator from `chat_template.jinja`.' Then follow with elaboration after a blank `///` line.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:497` — The test claims to verify that `.json` and `.glm4` formats 'keep the historical rendering — one assistant message per call', but only tests with a single tool call. The 'one per call' invariant cannot be verified when there is only one call; it is satisfied trivially regardless of whether folding is applied. Extend the test to include a case with two tool calls (e.g., `[first, second]`) and verify that both `.json` and `.glm4` create two messages (one per call), proving they do not fold multiple calls into one message like the `.mistral` format does.