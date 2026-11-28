---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8h2mptx8p1gkxm2f96regwp
  text: |-
    ### test — green
    - evidence: xctest MLXLMTests 2858 Swift Testing + 1261 XCTest pass (3 known issues); MLXFoundationModelsTests 776 pass; MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 pass; xcodebuild IntegrationTesting ReasoningEffortTemplateTests 4 tests pass (real Qwen3.8 and Qwen3.5-9B templates). ChatConventionsTests.budgetTransitionsAreTheOnlyFamilyDelta now also resets effortKey.
    - note: the Qwen3.8 template accepts only xhigh, medium, low. `.light`, `.moderate`, `.deep` name no level.
  timestamp: 2026-11-28T12:38:50.973566+00:00
position_column: done
position_ordinal: ffffff9880
title: Send a named reasoning effort level to the chat template (D218)
---
Brief: /Users/wballard/github/swissarmyhammer/.briefs/effort/mlx-swift-lm-reasoning-effort-value.md (from testudoapp-4f, decision D218).

## Problem
`MLXFoundationModels` reduces `contextOptions.reasoningLevel` to on or off and sends only `enable_thinking`. A named level (for example `.custom("low")`) cannot get to `reasoning_effort`.

## Plan
- Add `ReasoningConfig.effortKey: String?` (default nil). Set it to `reasoning_effort` in `QwenReasoningProtocol.qwen35` (Qwen3.8 reads it, Qwen3.5 ignores it) and in `InklingReasoningProtocol.frames`.
- `ThinkingRender.init(config:thinkingEnabled:effort:)`: when thinking is on and the protocol has an effort key, state the effort under that key.
- `MLXLanguageModel`: `reasoningEffort(for:)` gives the name of a `.custom` level (not `no_think`). `.light`, `.moderate`, `.deep` and no level give nil (no change).
- Send the effort on the plain path and the tool-aware path (the think-then-call phase uses the tool-aware prompt).

## Acceptance
- Qwen3.8 contract render with `.custom("low")`: "Reasoning effort is set to low". No level: the `xhigh` line.
- Qwen3.5 (no effort read): no change. Thinking off: no change.
- Tool path sends the same level.