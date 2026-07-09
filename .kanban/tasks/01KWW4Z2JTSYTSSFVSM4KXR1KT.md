---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx2qzgsatq00afjmhjmphe9x
  text: |-
    Implemented the context-size check. Summary:

    **Where the check hooks in** (`Libraries/MLXFoundationModels/MLXLanguageModel.swift`, `Executor`):
    - New private static `validateContextSize(tokenCount:contextLength:)` throws `LanguageModelError.contextSizeExceeded(LanguageModelError.ContextSizeExceeded(contextSize:tokenCount:debugDescription:))` when `tokenCount > contextLength`; no-op when `contextLength` is `nil`.
    - `dispatchGeneration`'s guided-generation branch validates `setup.input` and the plain text-generation branch validates `setup.effectiveInput`, both right after unwrapping and before calling `runGuidedGeneration`/`runTextGeneration`.
    - The tool-calling path re-tokenizes independently (`context.tokenizer.applyChatTemplate` with tool specs), so `runToolCalling` got a new `contextLength: Int?` parameter (threaded from `setup.contextLength`) and validates right after building `toolAwareInput`, before `prepareConstraintSetup` (grammar/xgrammar compilation) runs.
    - `RespondSetup` gained a `contextLength: Int?` field, populated in `prepareRespondSetup` from the same `BaseConfiguration` decode that already produced `modelType` (previously decoded and discarded everything but `modelType`; now the one decode serves both).

    **Determining the model's real context length**: `Libraries/MLXLMCommon/BaseConfiguration.swift` had no context-length field at all. Grepped every model family under `Libraries/MLXLLM/Models` and confirmed `max_position_embeddings` is the overwhelmingly dominant key (Llama, Qwen, Gemma, Phi, GLM, Olmo, Mistral, etc. all use it, typically as `Int?` since some omit it). Added `public var contextLength: Int?` to `BaseConfiguration`, keyed to `"max_position_embeddings"` in its `CodingKeys`. This is a distinct type from each model family's own per-architecture `Args` struct (which separately declares its own `maxPositionEmbeddings` CodingKey for RoPE sizing), so there's no collision — `BaseConfiguration` is only decoded for `model_type`/quantization/eos-tokens/context-length metadata, never for building the actual model layers.

    **Decision on unknown/absent context length**: when `config.json` doesn't expose `max_position_embeddings` (or the file/directory isn't resolvable, e.g. an `.id(...)`-only `ModelConfiguration` never downloaded to a local directory), `contextLength` is `nil` and the check is skipped entirely rather than guessing a default — a wrong guess could reject requests the model can actually serve.

    **Real SDK signature note**: the task description states `LanguageModelError.contextSizeExceeded(contextSize:tokenCount:debugDescription:)` as if it were a flat case, but the real os27 SDK (confirmed via the FoundationModels.swiftmodule swiftinterface) declares it as `case contextSizeExceeded(LanguageModelError.ContextSizeExceeded)` — a nested struct, same shape as the existing `unsupportedCapability(UnsupportedCapability)` pattern already used elsewhere in this file. Implemented against the real nested-struct signature.

    **Tests** (`Tests/MLXFoundationModelsTests/ContextSizeValidationTests.swift`, new file):
    1. `withinContextLengthReachesGeneration` — contextLength=100, fixed processor returns 5 tokens; expects the fake model's `prepare()` probe error (proving dispatch reached real generation, no regression).
    2. `exceedingContextLengthThrowsBeforeGeneration` — contextLength=4, fixed processor returns 10 tokens; expects `LanguageModelError.contextSizeExceeded` with `contextSize == 4`, `tokenCount == 10`, and asserts the generation-probe error was NOT thrown (i.e. `prepare()` was never reached).

    Both fake a real on-disk model directory with a `config.json` (so `context.configuration.modelDirectory` resolves and the decode has something to read), following the loud-throwing-model/fixed-tokenizer idiom from `EagerFallbackPrepareOrderingTests.swift`.

    **TDD verification**: temporarily neutered `validateContextSize` to `return` immediately, rebuilt, ran the suite — confirmed test 2 (RED) failed with "Generation was reached..." and test 1 still passed (no regression), then restored the real implementation and reran (GREEN) — both pass.

    **Full verification** (SAFE invocation only, no `swift test`, no `-XCTest` filtering):
    - `swift build`: clean.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - `xcrun xctest` per bundle, each under `timeout`:
      - CXGrammarTests: 7/7 passed
      - MLXGuidedGenerationTests: 62/62 passed
      - MLXFoundationModelsTests: 116/116 passed (includes the 2 new tests)
      - MLXLMTests: 245/245 passed

    Adversarial double-check agent dispatched for sign-off before handoff; task left in `doing` per scope instructions.
  timestamp: 2026-07-09T06:10:28.778149+00:00
- actor: claude-code
  id: 01kx2s81k5jg7jxgt2n31xhzq8
  text: |-
    Adversarial double-check (spawned per really-done) returned REVISE with a real high-severity bug — fixed, verified, re-tested:

    **Finding 1 (High, fixed)**: `dispatchGeneration`'s plain text-generation branch validated `setup.effectiveInput` (the baseline, non-reasoning-primed prompt), but `runTextGeneration` actually feeds `reasoningSetup.input` to generation whenever reasoning is active (`.reasoning` declared + resolved `reasoningConfig` present, no tools/schema). The reasoning-primed render (via `Self.preparedInput` with a non-nil `additionalContext`) is independently rendered and can legitimately have a different token count than the baseline — so an over-length reasoning prompt was never checked, and `contextSizeExceeded` would never fire on that path.

    Fix: in `dispatchGeneration`'s final `else` branch, compute `let promptToValidate = setup.reasoningSetup?.input ?? fallbackInput` and validate that instead of always validating `fallbackInput`.

    **Finding 2 (Moderate, fixed)**: the guided-generation (`setup.input`) and tool-calling (`runToolCalling`'s independent re-tokenized) call sites had zero test coverage, and the reasoning sub-case above was untested too -- exactly why the bug shipped unnoticed.

    Added 3 more tests to `ContextSizeValidationTests.swift` (5 total now):
    - `exceedingContextLengthViaReasoningPromptThrowsBeforeGeneration` — reproduces Finding 1 exactly: `capabilities: [.reasoning]`, a `ReasoningConfig(.templateFlag(...))` on the model's `ModelConfiguration`, and a new `ContextSizeReasoningAwareProcessor` that returns a small token count for the baseline render (no `additionalContext`) and a large, over-limit count for the reasoning render (`additionalContext` present) — verified this **failed** (RED) against the pre-fix code with exactly the expected "generation was reached" message, then verified it passes after the fix.
    - `exceedingContextLengthThrowsBeforeGenerationGuidedGeneration` — schema-present request, exercises the `setup.input`/guided-generation call site.
    - `exceedingContextLengthThrowsBeforeGenerationToolCalling` — tools-present request; since `runToolCalling` re-tokenizes independently via `context.tokenizer.applyChatTemplate` (not `context.processor`), added a `ContextSizeFixedLengthTokenizer` fake and a `tokenizer:` parameter on `makeContextSizeTestModel` to control that path's prompt length directly.

    Refactored `makeContextSizeTestModel` to take `processor`/`capabilities`/`reasoningConfig`/`tokenizer` parameters (all defaulted) instead of duplicating the directory/config.json/model-construction boilerplate per test, and extracted a shared `expectContextSizeExceeded` assertion helper.

    **Re-verified after the fix (all SAFE invocation, no `swift test`, no `-XCTest` filtering):**
    - `swift build`: clean.
    - `xcodebuild build-for-testing ...`: TEST BUILD SUCCEEDED.
    - `xcrun xctest` per bundle:
      - CXGrammarTests: 7/7 passed
      - MLXGuidedGenerationTests: 62/62 passed
      - MLXFoundationModelsTests: 119/119 passed (was 116; +3 new tests, all 5 context-size tests green)
      - MLXLMTests: 245/245 passed

    Task remains in `doing` for `/review`.
  timestamp: 2026-07-09T06:32:36.709299+00:00
- actor: claude-code
  id: 01kx2wp15mv5gpdzcs11h7c1pe
  text: |-
    Added the regression test for the Phase-2 re-validation fix and completed full verification.

    **Confirmed the fix is present and correct**: `Libraries/MLXFoundationModels/MLXLanguageModel.swift`'s `runToolCalling`, right after `phase2Input` is constructed and before `phase2MaxTokens`/`executeToolCallingPhase2`:
    ```swift
    try Self.validateContextSize(
        tokenCount: phase2Input.text.tokens.size, contextLength: contextLength)
    ```
    Matches the task description exactly (verified via `git diff`).

    **Investigation into constructing the test** (`Tests/MLXFoundationModelsTests/ContextSizeValidationTests.swift`):

    Studied `runToolCallReasoningPhase`/`ReasoningTokenCollector`/`ReasoningEventEmitter`/`NaiveStreamingDetokenizer` closely. Key findings that made a full end-to-end test tractable (no fallback to a narrower test needed):
    - `TokenIterator` only samples the LAST position's logits each forward call (`logits[..., -1, ...]`), so a fake model can return one-hot logits favoring a *planned* token sequence indexed by a running position counter -- exactly the `MockMainModel` idiom already used in `Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift`. No real weights needed.
    - `ReasoningEventEmitter`'s `primedInside` flag (from `reasoningPrimedInside`, which decodes the prompt's tail) lets Phase 1 start already "inside" a reasoning span (mirroring DeepSeek-R1-style prompts that prefill `<think>`) -- so the fake model only has to emit the CLOSING delimiter's bytes to close Phase 1 cleanly, not the opening one too. Made the prompt text literally `"<think>"` (unclosed) to trigger this.
    - Crucially, this test is the ONLY one in the file whose prompt passes the initial `toolAwareInput` check and so actually reaches `prepareConstraintSetup`'s real xgrammar/`GrammarConstraint` compilation (every other existing test in the file fails before that point). Reused the proven `<0xNN>` SentencePiece-style byte-fallback vocab convention from `EagerFallbackPrepareOrderingTests.EagerFallbackByteFallbackTokenizer` / `ToolCallingSchemaTests.makeByteTokenizer()` (new `ContextSizeThinkThenCallByteTokenizer`) so grammar compilation succeeds against a known-compatible vocab.
    - A model whose `prepare` throws immediately (like the existing `ContextSizeGenerationProbeModel`) can't be reused here since Phase 1 needs `prepare` to succeed and drive real token generation. Built a new fake, `ContextSizeReasoningCloseModel`, whose `prepare` succeeds exactly ONCE then throws a dedicated `ContextSizeReasoningPhase2ReachedError` on any later call -- since Phase 2 starts its own `TokenIterator` (a second `prepare` call), this cleanly proves "Phase 2 was reached" if the fix is missing/reverted, without risking an unbounded grammar-constrained generation loop hanging the test.

    **Test added**: `exceedingContextLengthAfterThinkThenCallPhase1ThrowsBeforePhase2` (in the same `ContextSizeValidationTests` suite). Setup: `capabilities: [.toolCalling, .reasoning]`, `.templateFlag` reasoning config, prompt = `"<think>"` (7 byte-tokens, within `contextLength: 10`), fake model emits the 8 bytes of `"</think>"` to close Phase 1 normally (not cut off) with `reasoningTokenIDs.count == 8`, so `phase2Input` = 7 + 8 = 15 tokens > `contextLength` 10. Asserts `LanguageModelError.contextSizeExceeded(contextSize: 10, tokenCount: 15)` fires, and explicitly fails if `ContextSizeReasoningPhase2ReachedError` is seen instead (proving Phase 2 was never reached).

    Extended (not replaced) the shared `makeContextSizeTestModel` helper with an additional `model:` factory-closure parameter (defaults to the existing `ContextSizeGenerationProbeModel`) -- factory rather than a plain instance because `LanguageModel` isn't `Sendable` and the model is constructed inside a `@Sendable load` closure.

    **RED/GREEN verification** (explicit, as required):
    - RED: commented out the `validateContextSize` call in `runToolCalling`, rebuilt (`TEST BUILD SUCCEEDED`), ran `MLXFoundationModelsTests` -- exactly 1 failure, the new test, failing precisely because `ContextSizeReasoningPhase2ReachedError` fired (`Issue recorded ... Phase 2 was reached even though Phase 1's reasoning tokens push the prompt over the model's context length`). All other 119 tests stayed green -- confirms the test is meaningful and isolated.
    - Restored the fix, verified via `git diff` that the file matches the original fix exactly (no stray edits).
    - GREEN: rebuilt, ran `MLXFoundationModelsTests` again -- 120/120 passed (was 119 + this 1 new test).

    **Full verification (SAFE invocation only, no `swift test`, no `-XCTest` filtering)**:
    - `swift build`: clean (only the pre-existing unrelated `init(capabilities:)` deprecation warning).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - `xcrun xctest <bundle>` per bundle, each wrapped in `timeout`:
      - CXGrammarTests: 7/7 passed
      - MLXGuidedGenerationTests: 62/62 passed
      - MLXFoundationModelsTests: 120/120 passed (119 + 1 new)
      - MLXLMTests: 245/245 passed

    No other files touched; did not commit (left for the orchestrator); task left in `doing`.
  timestamp: 2026-07-09T07:32:40.756878+00:00
position_column: done
position_ordinal: '8580'
title: Add context-window length checking before prefill, throwing LanguageModelError.contextSizeExceeded
---
Nothing in Libraries/MLXFoundationModels/MLXLanguageModel.swift currently checks the converted transcript's token count against the model's context length before running prefill/generation. A transcript that has grown too large for the model's context window has no dedicated failure path — it either crashes, silently truncates, or fails with an unrelated low-level error.\n\n**Confirmed real signature** (verified by reading Anthropic's own `ClaudeForFoundationModels` provider, which ships against the real os27 SDK — see `Sources/ClaudeForFoundationModels/ErrorMapper.swift` at https://github.com/anthropics/ClaudeForFoundationModels): the framework's real typed error is\n\n```swift\nLanguageModelError.contextSizeExceeded(contextSize: Int, tokenCount: Int, debugDescription: String)\n```\n\nAnthropic's HTTP-based provider can't recover exact numbers from the Messages API's error response, so it reports `contextSize: 0, tokenCount: 0` — a best-effort typed error with no real accounting. MLX runs its own tokenizer locally and can do strictly better: count the actual token length of the converted prompt and compare it against the model's real context window before prefill, so the thrown error carries accurate `contextSize`/`tokenCount` values.\n\nImplement the check (tokenize the converted prompt, compare against the model's context length) prior to invoking generation in the Executor's `respond()` path, and throw `LanguageModelError.contextSizeExceeded` with real counts when it's exceeded.\n\nAdd tests covering: a transcript within limits succeeds normally, and a transcript exceeding the model's context length throws `contextSizeExceeded` with accurate `contextSize`/`tokenCount` before any generation work is attempted.\n\n## Resolution (2026-07-08/09)\n\nCommit `a285796`: added `BaseConfiguration.contextLength` (from `max_position_embeddings`), `Executor.validateContextSize(tokenCount:contextLength:)`, wired into all dispatch paths (guided-generation, plain text, reasoning-primed text, tool-calling's initial `toolAwareInput`). 5 new tests in `ContextSizeValidationTests.swift`. 433 tests green, independently re-verified twice with a full trace of every generation path.\n\n## Review Findings (2026-07-09 01:38)\n\n- [x] Duplicated reasoning error message — **DEFERRED**, already tracked in `^9jtbtkd`, confirmed pre-existing/untouched by this task.\n- [x] `runReasoning()` 5-level nesting — **DEFERRED**, confirmed pre-existing/untouched by this task (noting here rather than a new tracking task since it's a single isolated instance).\n- [ ] **Real gap in this task's own feature**: the think-then-call tool-calling path validates the initial `toolAwareInput` but never re-validates `phase2Input` (`toolAwareTokens + reasoningTokenIDs`) after Phase 1 generates reasoning tokens — a prompt that started within the context window could exceed it after Phase 1, uncaught. **FIXING** — added `try Self.validateContextSize(tokenCount: phase2Input.text.tokens.size, contextLength: contextLength)` right after `phase2Input` is constructed in `runToolCalling`, before `executeToolCallingPhase2`. Needs a regression test (see below) and re-verification.\n- [x] `BaseConfiguration.swift` — `eosTokenIds` casing, missing doc on `QuantizationContainer`-adjacent init, `QuantizationContainer.init(from:)` nesting — **DEFERRED**, confirmed pre-existing/untouched by this task (this task only added the `contextLength` field/CodingKey case to this file).