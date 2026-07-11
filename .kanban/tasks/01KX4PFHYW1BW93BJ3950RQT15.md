---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: Label Chat.Message factory methods' first parameter (system/assistant/user/tool/calls) per fluent-usage convention
---
## What
Review of `2yyn7f7`'s commits (`7236bbd` doc sweep, `0ecab73` factory-body dedup) confirmed 5 findings, but all 5 are on signatures that neither commit touched — confirmed via `git diff HEAD~2..HEAD -- Libraries/MLXLMCommon/Chat.swift` showing zero changes to any `func system(`/`func assistant(`/`func user(`/`func tool(`/`func calls(` signature line. This is genuinely pre-existing public API, present since the file's original commit.

The project's fluent-usage validator flags:
- `Chat.Message.Tool.calls(_ calls: [ToolCall])` — unlabeled first param
- `Chat.Message.system(_ content: String, ...)` — unlabeled first param
- `Chat.Message.assistant(_ content: String, ...)` — unlabeled first param
- `Chat.Message.user(_ content: String, ...)` — unlabeled first param
- `Chat.Message.tool(_ content: String, ...)` — unlabeled first param

Rule cited: "Omit the first argument label only for value-preserving conversions." Constructing a `Message`/`Tool` isn't a value-preserving conversion, so per this rule each should be labeled (e.g. `system(content:)` instead of `system(_:)`).

## Why this is its own task, not folded into 2yyn7f7
This is a **breaking public API rename** (`MLXLMCommon` is a public library target), not a mechanical/internal change. A non-trivial number of call sites use the current unlabeled form across the repo, including at minimum:
- `Libraries/MLXFoundationModels/TranscriptConverter.swift`
- `Libraries/MLXFoundationModels/MLXLanguageModel.swift`
- `Libraries/MLXLMCommon/UserInput.swift`
- `Libraries/MLXLMCommon/ChatSession.swift`
- `Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift`
- `Tests/MLXLMTests/ChatSessionTests.swift`, `UserInputTests.swift`, `ToolCallIdTests.swift`
- Several `IntegrationTesting` test files

(exact count not yet fully enumerated — the task should start by precisely grepping for every real call site, excluding `.build`/vendored checkouts).

## Acceptance Criteria
- [ ] Decide/confirm this rename is actually desired (it's a source-breaking change for any external consumer of `MLXLMCommon`, not just this repo) — if there's a reason the unlabeled form was chosen deliberately (e.g. matching a common Swift convention for primary-content factory methods, similar to how `Message.tool(_:images:id:)` already exists this way), that should be weighed before mechanically applying the rule. Documenting a deliberate rationale (with a comment) is an acceptable alternative resolution to renaming, same as how `ModelCache.makeConstraint`'s duplication finding was resolved by documentation in `9jtbtkd`.
- [ ] If renaming: label each factory's first parameter (`content:`) and `Tool.calls`'s (`calls:`), update every real call site in the repo (excluding vendored `.build` checkouts), keep default values/other params unchanged.
- [ ] Build clean, full test suite green (MLXLMTests, MLXFoundationModelsTests, MLXGuidedGenerationTests, any others touched).
- [ ] A local review pass (`review sha` scoped to the commit) confirms zero remaining findings of this class.

## Tests
- [ ] No new tests needed if this is a pure rename — existing suites must stay green with updated call sites.