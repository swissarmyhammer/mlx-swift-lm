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
- actor: claude-code
  id: 01ky01dy4rd3weh3rz5e21czw6
  text: |-
    /implement iteration 2: worked the 73-item Review Findings checklist — all boxes flipped to [x].

    Root fixes (TDD RED→GREEN observed for each behavior change):

    1. Libraries/MLXLMCommon/Tokenizer.swift — documented EVERY public declaration (all 8 protocol requirements + 3 token properties, all extension conveniences, TokenizerError enum/case/errorDescription, StreamingDetokenizer protocol + append, NaiveStreamingDetokenizer struct/init/append/next). Extracted private `tokenId(of:)` helper now shared by eosTokenId/unknownTokenId and the NEW `bosTokenId` (completes the Token→TokenId pattern). Added convenience overloads `applyChatTemplate(messages:addGenerationPrompt:)` and `applyChatTemplate(messages:tools:addGenerationPrompt:)` mirroring the 3-param conveniences.

    2. Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift — documented all six public macro structs and their expansion methods. LoadContainerMacro and LoadContextMacro now REQUIRE the labeled `configuration` argument (matching LanguageModelMacro); shared fileprivate `argument(_:)` + `progressHandlerArgument` helpers on FreestandingMacroExpansionSyntax kill the verbatim progress-extraction duplication; LanguageModelMacro reuses the same helper. Call-site audit: the public macro declarations in Libraries/MLXHuggingFace/Macros.swift already declare labeled `configuration:` and every doc/README example uses the label — zero call-site changes needed (repo-wide grep confirmed no positional usage).

    3. Tests: NEW Tests/MLXLMTests/TokenizerTokenIdTests.swift (bosTokenId RED was a compile error before implementation); 3 new convenience-overload tests in TokenizerGenerationPromptTests.swift; NEW LoadContainerMacroTests + LoadContextMacroTests in Tests/MLXHuggingFaceMacrosTests/LanguageModelMacroTests.swift (labeled-expansion pin, progressHandler-forwarding pin, positional-diagnoses test — the diagnoses tests were observed RED, "Expected 1 diagnostics but received 0", before the guard change).

    Verification: full `swift test` exit 0, zero failures — XCTest 203+70 assorted bundle tests incl. MLXHuggingFaceMacrosTests now 12 tests, swift-testing runs 273+80+258+7 (618) all passed. Forced recompile of all touched targets (content-neutral change + revert): zero compiler warnings. Validators (PostToolUse) clean throughout.

    Task left in `doing` for /review. Nothing committed (tree still carries concurrent WIP in Evaluate.swift + MLXFoundationModels — commit must stage selectively).
  timestamp: 2026-07-20T15:13:36.920128+00:00
position_column: review
position_ordinal: '80'
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

## Review Findings (2026-07-20 09:38)

- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:12` — Public struct DownloaderMacro lacks documentation. All public APIs require doc comments per Swift conventions. Add a doc comment explaining what DownloaderMacro does and when to use it.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:34` — Public struct TokenizerAdaptorMacro lacks documentation. All public APIs require doc comments per Swift conventions. Add a doc comment explaining what TokenizerAdaptorMacro does.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:35` — Public static method expansion in TokenizerAdaptorMacro lacks documentation. Add a doc comment explaining this macro's expansion behavior.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:48` — Public struct `TokenizerAdaptorMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment above the struct declaration explaining its purpose.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:49` — Public static method `expansion` on `TokenizerAdaptorMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what the macro expands to.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:96` — Public struct `TokenizerLoaderMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment above the struct declaration explaining its purpose.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:97` — Public static method `expansion` on `TokenizerLoaderMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what the macro expands to.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:103` — Public struct TokenizerLoaderMacro lacks documentation. Add a doc comment explaining what TokenizerLoaderMacro does.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:104` — Public static method expansion in TokenizerLoaderMacro lacks documentation. Add a doc comment explaining this macro's expansion behavior.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:114` — Public struct `LoadContainerMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment above the struct declaration explaining its purpose.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:115` — Public static method `expansion` on `LoadContainerMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what the macro expands to.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:119` — LoadContainerMacro extracts the first argument unconditionally as `configuration` (line 119), but LanguageModelMacro explicitly checks for the labeled 'configuration' argument using a helper function (line 225). This creates inconsistent behavior: LoadContainerMacro accepts any positional first argument, while LanguageModelMacro requires the label. Update LoadContainerMacro to explicitly check for the 'configuration' label:
```swift
guard let configuration = node.arguments.first(where: { $0.label?.text == "configuration" })?.expression else {
    throw MacroExpansionError.message(
        "#huggingFaceLoadModelContainer requires a configuration")
}
```.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:120` — LoadContainerMacro extracts the first argument unconditionally as `configuration` (line 120), but LanguageModelMacro explicitly checks for the labeled 'configuration' argument. This creates inconsistent behavior: LoadContainerMacro accepts any positional first argument, while LanguageModelMacro requires the label. Update LoadContainerMacro to explicitly check for the 'configuration' label:
```swift
guard let configuration = node.arguments.first(where: { $0.label?.text == "configuration" })?.expression else {
    throw MacroExpansionError.message(
        "#huggingFaceLoadModelContainer requires a configuration")
}
```.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:121` — Public struct LoadContainerMacro lacks documentation. Add a doc comment explaining what LoadContainerMacro does.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:122` — Public static method expansion in LoadContainerMacro lacks documentation. Add a doc comment explaining this macro's expansion behavior.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:133` — Public struct `LoadContextMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment above the struct declaration explaining its purpose.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:134` — LoadContextMacro extracts the first argument unconditionally as `configuration` (line 134), but LanguageModelMacro explicitly checks for the labeled 'configuration' argument. This creates inconsistent API contracts: LoadContextMacro accepts any positional first argument, while LanguageModelMacro requires the label. Update LoadContextMacro to explicitly check for the 'configuration' label:
```swift
guard let configuration = node.arguments.first(where: { $0.label?.text == "configuration" })?.expression else {
    throw MacroExpansionError.message(
        "#huggingFaceLoadModel requires a configuration")
}
```.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:134` — Public static method `expansion` on `LoadContextMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what the macro expands to.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:143` — Public struct LoadContextMacro lacks documentation. Add a doc comment explaining what LoadContextMacro does.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:144` — Public static method expansion in LoadContextMacro lacks documentation. Add a doc comment explaining this macro's expansion behavior.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:150` — LoadContextMacro extracts the first argument unconditionally as `configuration` (line 150), but LanguageModelMacro explicitly checks for the labeled 'configuration' argument. This creates inconsistent API contracts: LoadContextMacro accepts any positional first argument, while LanguageModelMacro requires the label. Update LoadContextMacro to explicitly check for the 'configuration' label:
```swift
guard let configuration = node.arguments.first(where: { $0.label?.text == "configuration" })?.expression else {
    throw MacroExpansionError.message(
        "#huggingFaceLoadModel requires a configuration")
}
```.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:153` — Public struct `LanguageModelMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment above the struct declaration explaining its purpose.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:154` — Public static method `expansion` on `LanguageModelMacro` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what the macro expands to.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:163` — Public struct LanguageModelMacro lacks documentation. Add a doc comment explaining what LanguageModelMacro does.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:164` — Public static method expansion in LanguageModelMacro lacks documentation. Add a doc comment explaining this macro's expansion behavior.
- [x] `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:168` — Progress handler extraction logic is verbatim duplicated in LoadContextMacro (line 197). The if-else block that extracts the progressHandler argument is identical; it should be extracted into a shared helper function to avoid drift between the two macros. Extract the progress extraction logic into a helper function that takes the node and returns the progress string, then call it from both LoadContainerMacro and LoadContextMacro.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:7` — Protocol requirement func encode lacks documentation. Protocol members must be documented per Swift conventions. Add a doc comment explaining what encode does, its parameters, and return value.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:7` — Public protocol requirement `encode(text:addSpecialTokens:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose, parameters, and return value.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:8` — Protocol requirement func decode lacks documentation. Add a doc comment explaining what decode does.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:8` — Public protocol requirement `decode(tokenIds:skipSpecialTokens:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose, parameters, and return value.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:9` — Protocol requirement func convertTokenToId lacks documentation. Add a doc comment explaining what this function does and when it returns nil.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:9` — Public protocol requirement `convertTokenToId(_:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose and parameters.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:10` — Protocol requirement func convertIdToToken lacks documentation. Add a doc comment explaining what this function does.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:10` — Public protocol requirement `convertIdToToken(_:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose and parameters.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:12` — Protocol requirement var bosToken lacks documentation. Add a doc comment explaining what this token represents.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:12` — Public protocol property `bosToken` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this property represents (beginning-of-sequence token).
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:13` — Protocol requirement var eosToken lacks documentation. Add a doc comment explaining what this token represents.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:13` — Public protocol property `eosToken` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this property represents (end-of-sequence token).
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:14` — Protocol requirement var unknownToken lacks documentation. Add a doc comment explaining what this token represents.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:14` — Public protocol property `unknownToken` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this property represents.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:16` — Protocol requirement func applyChatTemplate (3-parameter version) lacks documentation. This is a complex public API requiring usage clarity. Add a doc comment explaining the parameters, return value, and when it throws.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:16` — Public protocol requirement `applyChatTemplate(messages:tools:additionalContext:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose, parameters, and return value.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:50` — Public convenience method func encode in Tokenizer extension lacks documentation. Add a doc comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:51` — Property eosTokenId and unknownTokenId (line 56) have identical implementations differing only by variable name. Both follow the pattern: guard-let on a token property, then return convertTokenToId(token). This is one function with an argument waiting to be extracted. Extract a shared helper function like `private func getTokenId(for token: String?) -> Int?` and call it from both properties: `public var eosTokenId: Int? { getTokenId(for: eosToken) }`.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:54` — Public convenience method func decode in Tokenizer extension lacks documentation. Add a doc comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:54` — Public extension method `encode(text:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining this convenience overload with default parameter.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:58` — Public computed property eosTokenId in Tokenizer extension lacks documentation. Add a doc comment explaining what this property returns.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:58` — Public extension method `decode(tokenIds:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining this convenience overload with default parameter.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:62` — The protocol provides convenience properties `eosTokenId` and `unknownTokenId` (lines 53–62) that convert token strings to optional Int IDs, but there is no corresponding `bosTokenId` property. Since `bosToken` is a protocol property (line 11) and can be looked up via `convertTokenToId` exactly like the others, this breaks the established pattern of providing a TokenId convenience for each Token property. Add a `bosTokenId` convenience property to match the pattern:
```swift
public var bosTokenId: Int? {
    guard let bosToken else { return nil }
    return convertTokenToId(bosToken)
}
```.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:62` — Public extension property `eosTokenId` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this computed property represents (end-of-sequence token ID).
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:63` — Public computed property unknownTokenId in Tokenizer extension lacks documentation. Add a doc comment explaining what this property returns.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:67` — Public extension property `unknownTokenId` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this computed property represents.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:68` — Public convenience method func applyChatTemplate (1-parameter version) in Tokenizer extension lacks documentation. Add a doc comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:72` — Public extension method `applyChatTemplate(messages:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:73` — Public convenience method func applyChatTemplate (2-parameter version) in Tokenizer extension lacks documentation. Add a doc comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:74` — The 3-parameter `applyChatTemplate` method has convenience overloads (lines 60–72) that provide partial application, but the new 4-parameter version does not follow the same pattern. Callers must always supply all four parameters explicitly, even for common cases like rendering with just `messages` and `addGenerationPrompt`. Add at least one convenience overload for the 4-parameter method:
```swift
public func applyChatTemplate(
    messages: [[String: any Sendable]],
    addGenerationPrompt: Bool
) throws -> [Int]? {
    try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil, addGenerationPrompt: addGenerationPrompt)
}
```.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:78` — Public extension method `applyChatTemplate(messages:tools:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining this convenience overload.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:92` — Public enum case missingChatTemplate lacks documentation. Add a doc comment explaining when this error case occurs.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:94` — Public computed property errorDescription in TokenizerError enum lacks documentation. Add a doc comment explaining what this property returns.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:98` — Public enum `TokenizerError` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the error type and its cases.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:99` — Public enum case `missingChatTemplate` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining when this error occurs.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:100` — Public protocol StreamingDetokenizer lacks documentation. Add a doc comment explaining what this protocol does and its purpose.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:101` — Public property `errorDescription` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining what this property provides.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:104` — Public struct NaiveStreamingDetokenizer lacks documentation. Add a doc comment explaining what this implementation does and when to use it.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:109` — Public protocol `StreamingDetokenizer` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the protocol's purpose and usage.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:110` — Public initializer in NaiveStreamingDetokenizer lacks documentation. Add a doc comment explaining what the tokenizer parameter is.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:110` — Public protocol requirement `append(token:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose and parameters.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:113` — Public struct `NaiveStreamingDetokenizer` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the struct's purpose as a naive streaming detokenizer implementation.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:117` — Public mutating method append in NaiveStreamingDetokenizer lacks documentation. Add a doc comment explaining what this method does.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:119` — Public initializer lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the initializer's parameters.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:123` — Public method `append(token:)` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose and parameters.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:134` — Public mutating method next in NaiveStreamingDetokenizer lacks documentation. Add a doc comment explaining what this method returns and when it returns nil.
- [x] `Libraries/MLXLMCommon/Tokenizer.swift:138` — Public method `next()` lacks a documentation comment; every public declaration must carry a `///` doc comment. Add a `///` documentation comment explaining the method's purpose and return value.