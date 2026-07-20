---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxzxca9ke8bbt6zqvc53c0t4
  text: |-
    Picked up. Research findings:

    1. Verified swift-transformers 1.3.3 (checkout at IntegrationTesting/.build/checkouts/swift-transformers, git describe = 1.3.3): the `Tokenizers.Tokenizer` PROTOCOL has a full requirement `applyChatTemplate(messages:chatTemplate:addGenerationPrompt:truncation:maxLength:tools:additionalContext:) throws -> [Int]` (Message = [String: any Sendable], ToolSpec = [String: any Sendable]). So the bridge can pass addGenerationPrompt + tools + additionalContext with nothing dropped (chatTemplate: nil, truncation: false, maxLength: nil).
    2. The main SwiftPM package does NOT depend on swift-transformers (by design — the macro emits source compiled in downstream apps). So the bridge's real render cannot be executed under `swift test`; it is verified textually via macro expansion, and the exact upstream overload was verified against the pinned checkout.
    3. Production conformers of MLXLMCommon.Tokenizer besides the macro bridge: only BenchmarkHelpers.NoOpTokenizer (plus ~20 test doubles). All keep compiling via the protocol-extension default returning nil.

    Plan: TDD — (a) expansion test for #adaptHuggingFaceTokenizer in LanguageModelMacroTests.swift incl. the new addGenerationPrompt method, (b) default-nil unit test in Tests/MLXLMTests, (c) ChatML-style contract test (no-genprompt render strict prefix of with-genprompt render) on a conformer implementing the new method. Then implement protocol + macro.
  timestamp: 2026-07-20T14:02:49.523610+00:00
- actor: claude-code
  id: 01kxzxys0sz9y19dgdzhvghrge
  text: |-
    Implementation landed via strict TDD (RED observed for both test groups before implementing):

    - Libraries/MLXLMCommon/Tokenizer.swift: new optional protocol requirement `applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:) throws -> [Int]?` + protocol-extension default returning nil. All existing conformers (macro bridge aside, only BenchmarkHelpers.NoOpTokenizer in production plus ~20 test doubles) compile unchanged and opt out.
    - Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift: TokenizerBridge now implements the method by delegating to swift-transformers' full protocol requirement (chatTemplate: nil, truncation: false, maxLength: nil — nothing from the MLX-side signature is dropped: tools + additionalContext forwarded). Same missingChatTemplate error mapping as the 3-arg method.
    - Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift: new TokenizerAdaptorMacroTests.testBridgeExpansion pins the full generated bridge (RED diff showed the only delta was the missing new method — formatting matched first try).
    - Tests/MLXLMTests/TokenizerGenerationPromptTests.swift: default-nil test (MinimalTokenizer, RED was "extra argument 'addGenerationPrompt' in call") + ChatML-style strict-prefix contract tests (no-genprompt render is strict prefix of primed render, lacks trailing `<|im_start|>assistant\n`; addGenerationPrompt:true reproduces the default render).

    Verification: full `swift test` exit 0, zero failures (XCTest bundles 2+197+6 tests; swift-testing runs 268+80+258+7). No new warnings (only pre-existing MLXVLM warnings, untouched files).

    Note on acceptance criterion 2: the macro-generated bridge cannot be *executed* under `swift test` (main package deliberately has no swift-transformers dependency — the macro emits source compiled downstream). Real-render verification = expansion test pinning the delegation text + manual verification that the 7-arg overload is a protocol requirement on Tokenizers.Tokenizer in the pinned 1.3.3 checkout; the ChatML strict-prefix semantics are unit-verified on a conformer implementing the method per the bridge contract.
  timestamp: 2026-07-20T14:12:54.425580+00:00
- actor: claude-code
  id: 01kxzyz4rhr4s1q8ppbfkvxm67
  text: |-
    really-done complete. Adversarial double-check verdict: REVISE with a single Medium finding — NOT about this task's code (all adversarial checks on the four changed files came back clean: no overload ambiguity, no pre-existing 4-arg conformers, generated bridge verified against the pinned swift-transformers 1.3.3 checkout API, expansion test pins the addGenerationPrompt forwarding, acceptance criteria 1+2 judged satisfied).

    The finding: during the review, the shared working tree was concurrently broken by task 01KXY0S30Q2VFX8STGNB3ZPF2Q's in-flight y→mainY rename in Libraries/MLXLMCommon/Evaluate.swift, making the earlier green `swift test` momentarily unreproducible, and creating a commit-scope hazard.

    Resolution:
    1. Waited for the concurrent edit to settle (diff stable at 28+/34- for 2+ min, `swift build` green), then re-ran full `swift test` fresh: exit=0, XCTest 203 tests (197+6) with 0 failures, swift-testing 613 tests (268+80+258+7) all passed — 816 total, zero failures. Green claim re-established on the current tree.
    2. Commit-scope hazard (for the orchestrator, which owns commits): this task's files are ONLY Libraries/MLXLMCommon/Tokenizer.swift, Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift, Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift, Tests/MLXLMTests/TokenizerGenerationPromptTests.swift (+ this task's .kanban files). The tree also carries other tasks' uncommitted changes (Evaluate.swift, MLXFoundationModels/*, PromptCache tests) — stage selectively, do not sweep.

    Task is green and left in `doing` for /review. Nothing committed, per instructions.
  timestamp: 2026-07-20T14:30:35.025730+00:00
- actor: claude-code
  id: 01kxzz1fxfq7b9kpn2vk5kx06a
  text: '/finish iteration 1: implement landed green in `doing` (swift test 816/816 — XCTest 203, swift-testing 613). Files: Libraries/MLXLMCommon/Tokenizer.swift (optional protocol method, default nil), Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift (bridge delegates to swift-transformers'' 7-arg protocol requirement — tools + additionalContext forwarded, nothing dropped), Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift (expansion pin), Tests/MLXLMTests/TokenizerGenerationPromptTests.swift (default-nil + ChatML strict-prefix contract). TDD RED→GREEN verified. Orchestrator note: tree carries a CONCURRENT session''s WIP (Evaluate.swift rename, MLXFoundationModels files) — checkpoint commit will stage only this task''s files. Next: /test verify → scoped /commit → /review HEAD~1..HEAD.'
  timestamp: 2026-07-20T14:31:51.983711+00:00
position_column: doing
position_ordinal: '8280'
title: 'Tokenizer: expose addGenerationPrompt:false chat-template render (prereq for stable-boundary hybrid checkpoints)'
---
## What

Prerequisite for the hybrid prompt-cache fix (see the dependent task "PromptCache: hybrid checkpoints must snapshot at the transcript-stable boundary"). The executor needs to render a message array **as past turns** — i.e. WITHOUT the template's generation-priming region — to compute which prefix of a prompt will re-render identically on later rounds.

1. `Libraries/MLXLMCommon/Tokenizer.swift`: add to the `Tokenizer` protocol an OPTIONAL capability:
   `func applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:) throws -> [Int]?`
   with a protocol-extension default returning `nil` (meaning "renderer cannot control the generation prompt") so every existing conformer (test tokenizers like `ByteTokenizer` in FMTestHelpers, ParoQuant's, macro-generated bridges) keeps compiling and simply opts out. Callers treat `nil` as "no stable boundary computable" and skip the new behavior — zero behavior change for opted-out tokenizers.
2. `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift` (`TokenizerAdaptorMacro`): implement the new method on the generated `TokenizerBridge` by delegating to swift-transformers' `applyChatTemplate` overload that takes `addGenerationPrompt` (pinned swift-transformers is 1.3.3 — verify the exact overload signature available on `Tokenizers.Tokenizer` there; if the full `(messages:addGenerationPrompt:tools:additionalContext:)` combination isn't available on the protocol, use the widest overload that is and document what's dropped). Update the macro's expansion test (`Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift`) accordingly.
3. Check whether any other concrete bridge implements `MLXLMCommon.Tokenizer` directly (repo grep: `HuggingFaceIntegrationMacros.swift` is the only production conformer besides test doubles) and retrofit if found.

## Acceptance Criteria

- [x] New protocol method with defaulted `nil` implementation; all existing conformers compile unchanged
- [x] Macro-generated bridge returns a real render with `addGenerationPrompt: false` — unit-verified: for a ChatML-style template, the no-genprompt render is a strict prefix of the with-genprompt render and lacks the trailing assistant header
- [x] `swift test` fully green (786-suite baseline), no warnings

## Tests

- [x] Extend `Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift` for the new bridge method's expansion
- [x] Unit test of the default-nil path on a minimal conformer
- [x] Run: `swift test` → green

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #qwen