---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxkx0wr6kx1z7zznd11xy2h3
  text: Picked up by /finish scoped-batch orchestrator. Starting implement → test → commit → review loop.
  timestamp: 2026-07-15T22:05:42.022744+00:00
- actor: claude-code
  id: 01kxqwgx4tsezvx7txhzw88wwa
  text: 'Re-picked up by /finish (single-task mode): the 2026-07-15 22:05 pickup session died before doing any work — no milestones, no findings, progress 0.0. Restarting the implement → test → commit → review loop from scratch. Dispatching /implement.'
  timestamp: 2026-07-17T11:13:55.866051+00:00
- actor: claude-code
  id: 01kxqx0zgnn21r9hsejydk9pdq
  text: |-
    Research findings (session restart, starting from scratch):

    ROOT CAUSES identified so far:
    1. `ToolCallFormat.infer(from:)` (Libraries/MLXLMCommon/Tool/ToolCallFormat.swift) has NO branch for model_type `minimax` (verified: local cached config.json says model_type=minimax). So `container.configuration.toolCallFormat` resolves nil for MiniMax-M2 → TranscriptConverter renders tool-call turns as verbatim assistant text → MiniMax's template counts them as plain assistant turns → `last_tool_call.name = none` → raise_exception on the next tool-role message. The `.minimaxM2` format + MiniMaxM2ToolCallParser exist but are unreachable via inference.
    2. `TranscriptConverter.mlxMessages` structured-tool_calls branch is gated to `toolCallFormat == .mistral` only. MiniMax's template (read the real chat_template.jinja from the cached snapshot) requires assistant `tool_calls` metadata before any tool-role message — same class as mistral. Template iterates `tool_call.function.name` + `.arguments.items()`, so the existing structured ToolCall bridge (arguments parsed to object) is the right shape.
    3. `ReasoningConfig.infer` has no minimax branch → no reasoning config → `<think>` handling never engages. MiniMax's template PRE-OPENS `<think>\n` at the generation prompt (`']~b]ai' ~ '\n' ~ '<think>' ~ '\n'`), so output starts inside an open think block — `reasoningPrimedInside` machinery exists and handles exactly this, but only when a ReasoningConfig is present. This is the composeChain leak: model thinks, thinking streams as text.
    4. Suspected: tool-calling path's `makeThinkThenCallConfig` gates think-then-call to `.templateFlag` families; MiniMax is alwaysOn+tool-aware (template renders tools AND pre-opens think). Without phase 1, a constrained round that thinks first produces `thinking…</think><tool_call>{…}` which fails envelope parse → fallback text leak (composeChain 0-calls signature). Investigating gate options.

    Verification harness located: ../FoundationModelsMultitool gated suite (MULTITOOL_INTEGRATION=1, SearchThenCallTests, 4 scenarios), wired via temporary .package(path:) override + model pin swap per prior sessions (y4s0w2j comments describe the procedure).
  timestamp: 2026-07-17T11:22:42.581818+00:00
- actor: claude-code
  id: 01kxqy01j28qqa7p7xz7wbddj2
  text: |-
    Implementation landed (4 TDD cycles, each RED watched then GREEN):

    A) ToolCallFormat.infer: model_type "minimax" (exact match, dense-GLM4 style) → .minimaxM2. This is what makes container.configuration.toolCallFormat non-nil for MiniMax-M2 — it also activates the already-written MiniMaxM2ToolCallParser on the MLXLMCommon native path (Evaluate.swift defaults nil → .json, which is why composeChain parsed 0 calls).

    B) ReasoningConfig.infer: "minimax" → alwaysOn <think>/</think> (template pre-opens <think>\n; no thinking kwarg exists). Note: per validateReasoningCapability, an alwaysOn model now REQUIRES the developer to declare .reasoning — otherwise a typed unsupportedCapability error, not a silent think-leak. Harness implication recorded below.

    C) TranscriptConverter: structured-tool_calls replay generalized from `== .mistral` to a structuredToolCallFormats set {.mistral, .minimaxM2}; mistralToolCall renamed structuredToolCall. Validated the fix shape against the REAL chat_template.jinja out-of-band with python jinja2: verbatim shape raises the exact "Message has tool role, but there was no previous assistant message with a tool call!" exception; structured shape renders the native <minimax:tool_call><invoke …> replay (arguments must be an object — DefaultMessageGenerator already emits argumentsObject, so .items() works). Test mirrors the template's last_tool_call validator (same pattern as the mistral alternation mirror from cd52383).

    D) Think-then-call gate widened for the tool path: makeThinkThenCallConfig now also passes .alwaysOn configs (was templateFlag-only, on the assumption alwaysOn ⇒ tool-blind — false for MiniMax-M2, which is alwaysOn AND tool-aware); a NEW post-render gate thinkThenCallPhase1Engages runs Phase 1 for alwaysOn ONLY when the rendered prompt is primed inside an open think block (M2's is; a tool-blind R1/V3-style prompt is not → single-phase unchanged, avoiding Phase 1 burning the budget waiting for a </think> that never comes). primedInside is now computed once at the call site and passed into executeThinkThenCallPhase1. Without this, M2's constrained tool round mixes free-form thinking into the envelope buffer → parse fallback → leaked reasoning as the reply (the composeChain signature).

    Targeted suites green: ToolTests inference, ReasoningConfigTests 17/17, TranscriptConverterTests 29/29, ThinkThenCallGateTests 8/8. Full swift test next.
  timestamp: 2026-07-17T11:39:40.482375+00:00
- actor: claude-code
  id: 01kxr06j74p5ak2mby4k5cpkz2
  text: |-
    Verification setup + first live attempt: wired FoundationModelsMultitool to this working tree (.package(path:) override), swapped the generation pin to mlx-community/MiniMax-M2-4bit, and added .reasoning to CLIRunner.makeMLXLanguageModel capabilities (required: MiniMax's new alwaysOn ReasoningConfig makes validateReasoningCapability throw unsupportedCapability when .reasoning is undeclared — that is by design, refuse-not-leak). All three harness edits are temporary and will be reverted.

    First singleCallWeather run FAILED at the suite's 30-min .timeLimit — but a concurrent interactive terminal session (ttys003, started 06:54, `swift-test --no-parallel --filter SearchThenCallTests`) is running the FULL gated suite against the same 119GB model on this same GPU. Two simultaneous MiniMax-M2 loads/generations contend for the M3 Ultra; my run stalling to the time limit is expected under that contention, so this failure is NOT evidence against the fix. Waiting for the tty run to finish, then re-running scenarios one at a time.
  timestamp: 2026-07-17T12:18:11.300951+00:00
- actor: claude-code
  id: 01kxr7119gw9z6mjx0xs43h0vb
  text: |-
    REAL-HARDWARE VERIFICATION (M3 Ultra, real mlx-community/MiniMax-M2-4bit weights, local working tree wired into FoundationModelsMultitool via .package(path:)):

    Route taken: the gated SearchThenCallTests scenarios kept hitting their fixed 30-min .timeLimit — NOT a hang: sampling showed the process at ~100% CPU, 127GB resident, actively inside MLXLanguageModel.Executor.respond. M2 rounds are simply slow now that each round runs a full Phase-1 thinking pass (the pre-fix repro reached the render-back throw quickly precisely because it never thought). First attempt was also invalidated by a concurrent tty-session suite run on the same GPU. So verification switched to the package's own multitool-cli (production wiring: Router resolve → makeMLXLanguageModel → native LanguageModelSession), which has no time gate and prints the answer; temp diagnostics added to print the transcript entry sequence and a think-leak check.

    RUN 1 (composeChain demo prompt, full tool set): completed in ~30 min, clean text answer, no TemplateException, no think markers in reply. Reply was an announce-then-stop deflection (same stochastic behavior class y4s0w2j logged and accepted for GLM), and transcript wasn't printed yet, so ran again with diagnostics.

    RUN 2 (single-call weather prompt "How warm is it in Austin right now?", --direct, ~25 min):
      TRANSCRIPT: instructions -> prompt -> response -> reasoning -> toolCalls(runCode) -> toolOutput(runCode) -> response -> reasoning
      THINK-LEAK: false

    That transcript is the acceptance evidence:
    - AC1 PROVEN live: the toolOutput rendered back into the continuation prompt and generation proceeded to a post-tool response — the exact step that threw TemplateException "Message has tool role, but there was no previous assistant message with a tool call!" before the fix.
    - AC2 PROVEN: one real end-to-end tool round trip (toolCalls(runCode) → toolOutput(runCode) → response) on the real 119GB weights.
    - AC3 PROVEN for these runs: reasoning landed in .reasoning transcript entries; final reply contains no <think>/</think> (THINK-LEAK: false); both runs' replies are clean prose.
    - AC4: unit-level regression guards green (mistral render tests unchanged+passing, non-mistral verbatim-render test, templateFlag gate behavior unchanged, alwaysOn-non-primed single-phase unchanged, minimax inference is an exact match so no other model_type is affected).

    Honest caveat: reply QUALITY is deflection-prone (model announces instead of finishing with the fixture temperature) — a model-capability/prompting outcome outside this card's ACs, same as the accepted GLM repair deflection in y4s0w2j. Also noted: M2 rounds now cost minutes of thinking each; the gated suite's 30-min per-scenario limit is tight for this model (harness-side concern, not this repo's).

    Next: revert all FoundationModelsMultitool temp edits, then really-done (fresh full suite + adversarial double-check).
  timestamp: 2026-07-17T14:17:30.160935+00:00
- actor: claude-code
  id: 01kxr7nemcqa7vsxv2s5d42a7b
  text: |-
    really-done GREEN. Evidence:
    - Fresh full `swift test` on the current tree (which absorbed mid-session upstream commits up to 942d870): exit 0, zero failure lines. Swift Testing 576 passed (257+0+80+232+7), XCTest 200 passed, 0 failures — baseline 764 + 12 new tests (1 ReasoningConfig, 3 TranscriptConverter, 8 ThinkThenCallGate; the 2 new infer expectations live inside an existing ToolTests test). Only known pre-existing warnings (SwiftPM unhandled-file/docc noise, MLXLanguageModel deprecated-init).
    - Adversarial double-check verdict: PASS. Key confirmations: alwaysOn non-primed tool-path execution provably byte-identical to before (reasoningContext nil, phase2Input == toolAwareInput, budget unchanged); structuredToolCallFormats rewrite semantics-preserving for nil/other formats; all mlxMessages/infer call sites checked (ChatSession already replays structured toolCalls — no missed sibling); minimax exact-match precludes cross-family inference changes; every production edit has a revert-detecting test; new test file's compile gate confirmed live by suite count 221→232.
    - FoundationModelsMultitool repo restored to checked-in state (git status clean; all 5 temp edits reverted).

    All 4 acceptance criteria checked off on the card with evidence. Leaving task in `doing` for /review per the implement contract.
  timestamp: 2026-07-17T14:28:39.180663+00:00
- actor: claude-code
  id: 01kxr7y9necp10y3f377mya6kv
  text: 'Iteration 1: implement landed green in doing — four root causes fixed (ToolCallFormat.infer "minimax" → .minimaxM2 [parser existed but was unreachable]; ReasoningConfig alwaysOn think tags for minimax; TranscriptConverter structured tool_calls replay generalized to [.mistral, .minimaxM2]; think-then-call gate accepts .alwaysOn when the rendered prompt is primed inside an open think block). Real-hardware verification on the 119GB MiniMax-M2-4bit: full tool round trip with tool result rendering back cleanly (the step that threw TemplateException before), THINK-LEAK false; gated scenarios hit their 30-min time limit due to full thinking, verified via multitool-cli wiring instead. Independent /test from scratch: 776 tests, 0 failures (baseline 764 + 12 new). Dispatching /commit.'
  timestamp: 2026-07-17T14:33:29.006740+00:00
position_column: doing
position_ordinal: '80'
title: 'MiniMax-M2 tool-calling transcript rendering: tool-role message rendered without its preceding assistant tool-call turn'
---
## What

Driving native tool calling with `mlx-community/MiniMax-M2-4bit` (model_type `minimax`, arch `MiniMaxM2ForCausalLM`) through `MLXLanguageModel`/`LanguageModelSession` throws on multi-turn tool exchanges:

```
TemplateException: "Message has tool role, but there was no previous assistant message with a tool call!"
```

Reproduced 2026-07-15 (M3 Ultra, 512GB, revision `1fbeb5d`, FoundationModelsMultitool gated suite): 3 of 4 scenarios threw this the moment a tool result had to be rendered back — the assistant tool-call turn is evidently missing/misplaced in the chat array by the time MiniMax's Jinja template validates it. This is the same class of failure as the (fixed) Mistral strict-alternation issue — `cd52383` renders mistral tool-call turns as structured `tool_calls`; MiniMax's family appears to need the same structured rendering path (its template checks that a `tool` message follows an assistant message carrying `tool_calls`).

The 4th scenario (composeChain) rendered but made 0 parsed tool calls and its final reply read like leaked reasoning ("I need to find out what cities are on the user's trip and then check the current…") — worth checking whether M2's interleaved `<think>` output is being handled by the reasoning path or ending up in the text reply.

Note: MiniMax-M2 is one of only two families with a dedicated tool-call parser already wired (`MiniMaxM2ToolCallParser`) — the parse side is ready; it's the render side that blocks it. Weights are fully cached locally (~119GB under `~/.cache/huggingface/hub/models--mlx-community--MiniMax-M2-4bit`), so verification needs no re-download.

## Acceptance Criteria

- [x] A multi-turn tool-calling exchange renders through MiniMax-M2's chat template without TemplateException (assistant tool-call turn present before each tool-role message). (Unit: TranscriptConverter minimax tests + template-validator mirror, cross-checked against the real chat_template.jinja via jinja2; live: RUN 2 transcript rendered toolOutput back and continued.)
- [x] MiniMax-M2-4bit completes at least one real tool-call round trip end to end. (Real weights, M3 Ultra, multitool-cli: `toolCalls(runCode) -> toolOutput(runCode) -> response`.)
- [x] M2's thinking output does not leak into the final text reply. (ReasoningConfig alwaysOn + primed-inside think-then-call; live THINK-LEAK: false, reasoning landed in .reasoning entries.)
- [x] Mistral/Qwen/GLM rendering unchanged. (Regression tests green: mistral alternation suite, non-mistral verbatim rendering, templateFlag gate unchanged, alwaysOn-non-primed single-phase unchanged; minimax inference is exact-match.)
