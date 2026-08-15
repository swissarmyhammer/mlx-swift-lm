---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kwywp9jbkhdnbwfdh73ncp42
  text: |-
    Orchestrator intervention: the implement loop was genuinely stuck (not just slow) on two shell-level hangs plus one real test bug.

    1. Two long-running `xcrun xctest` invocations (2.9h and 1.85h) ran the ENTIRE unfiltered MLXFoundationModelsTests.xctest bundle instead of a filtered run — killed (command ids 17, 22).
    2. A follow-up `xcodebuild test-without-building` invocation (missing `-disableAutomaticPackageResolution -skipPackagePluginValidation`) hung 70+ min trying to resolve the package graph in a no-network sandbox — killed (command id 40).
    3. **Root cause of the actual test hang**: `Tests/MLXFoundationModelsTests/EagerFallbackPrepareOrderingTests.swift`'s `toolCallingRequestSkipsEagerFallbackRender()` created a `LanguageModelExecutorGenerationChannel` and called `executor.respond(streamingInto: channel)` with no concurrent reader draining the channel. `Executor.respond` (MLXLanguageModel.swift:992) unconditionally sends a metadata event on that channel before reaching either template-render throw point; with no consumer, `channel.send` suspends forever (matches the pattern in `FMTestHelpers.TestResponseStream`, which every test that gets this far already relies on for exactly this reason).

    Fix applied directly (small, additive, low-risk): added a background `Task` right after `let channel = ...` that does `for try await _ in channel {}` to drain it, with `defer { drainer.cancel() }`. Verified:
    - `swift test --filter EagerFallbackPrepareOrderingTests`: builds and runs in ~9s (fails only on the pre-existing, already-documented "swift test cannot load the MLX metallib in this no-network sandbox" limitation — see this task's own prior verification notes on ^q3ddgqy, confirmed pre-existing/unrelated).
    - `xcodebuild build-for-testing` (with the disable-resolution flags) + `xcrun xctest <bundle>` (the proven working pattern from ^q3ddgqy): **all 84 tests in 16 suites passed in 0.05s**, including the new test (0.012s).

    Also note: the file was already renamed from `WeatherArgs` to `EagerFallbackWeatherArgs` earlier in this session to resolve a genuine `@Generable` macro cross-file name-collision compile error (not an access-level issue as I first guessed) — that fix is correct and stands.

    Next: run really-done / full test suite via the xcodebuild+xcrun pattern, then commit and review as normal.
  timestamp: 2026-07-07T18:15:51.627535+00:00
- actor: claude-code
  id: 01kwyxddhrzvzz3hp3q1x5q903
  text: |-
    Re-verified from a fresh agent instance (no re-diagnosis needed, per orchestrator's briefing):
    - Confirmed working tree matches expected state: MLXLanguageModel.swift has the reordered fallback (eager prepare() moved after the tool-branch return), and EagerFallbackPrepareOrderingTests.swift has both fixes (EagerFallbackWeatherArgs rename, channel-drainer Task + defer { drainer.cancel() }).
    - xcodebuild build-for-testing (with -disableAutomaticPackageResolution -skipPackagePluginValidation): TEST BUILD SUCCEEDED.
    - xcrun xctest .../MLXFoundationModelsTests.xctest (unfiltered): 84 tests in 16 suites passed in 0.050s, including "Executor.respond eager fallback vs tool-branch dispatch ordering" (0.012s).
    - swift build: Build complete, no errors (1.66s incremental).

    Proceeding to commit and task-scoped review.
  timestamp: 2026-07-07T18:28:29.368224+00:00
- actor: claude-code
  id: 01kwyxvvcffaqe0gkefjc4w88t
  text: |-
    Committed (2b8b318, local only, not pushed) and moved to review. Attempted the requested task-mode review scoped to HEAD~1..HEAD via `{"op": "review sha", "sha": "HEAD~1..HEAD"}` per /finish's checkpoint-delta convention.

    Blocked -- not a code finding, a pipeline infrastructure limit:
    `Validator 'scope' error: file Libraries/MLXFoundationModels/MLXLanguageModel.swift inlines 133804 bytes, over the 131072-byte review batch_size; a file is never split across review batches`

    Confirmed this predates this task's diff entirely: `git show HEAD~1:Libraries/MLXFoundationModels/MLXLanguageModel.swift | wc -c` = 133269 bytes -- already over the 131072-byte (128 KiB) limit before this commit's +504 net bytes. Same error reproduces regardless of scope narrowing tried: `review sha HEAD~1..HEAD`, `review file <path>` on the production file directly, `backend: "local"`, and attempts to pass a larger `batch_size`/`maxBatchSize`/`max_batch_size` (none are accepted -- the limit isn't caller-configurable through this tool's exposed parameters). A file is never split across batches, so any scope that includes this file hits the same wall.

    Isolated what I could: `review file Tests/MLXFoundationModelsTests/EagerFallbackPrepareOrderingTests.swift` (the new test, small enough to inline) came back **clean: 0 findings** (15 rules attempted, 0 confirmed).

    This is out of this task's scope to fix -- splitting MLXLanguageModel.swift to fit under a review-tool batch-size ceiling is an unrelated, large structural refactor, not part of "reorder tool-branch dispatch before eager fallback." Leaving the task in `review` rather than forcing it to `done`, since I have no clean pass on the production file's diff from the review engine. A human (or a future task with a raised batch_size / file-split) needs to unblock review of this file going forward.
  timestamp: 2026-07-07T18:36:22.287046+00:00
- actor: claude-code
  id: 01kx18h40dg8m6tjf1x2t96qc5
  text: 'Resuming via /finish scoped-batch. Confirmed the prior fix (commit 2b8b318, described in the comments above) did not survive the branch''s later hard-reset to origin/mlx-foundationmodels: `Tests/MLXFoundationModelsTests/EagerFallbackPrepareOrderingTests.swift` no longer exists, and `prepareRespondSetup` (the function `respond()`''s logic was extracted into during ^q3ddgqy''s round-6 rework) still calls `context.processor.prepare(input: userInput)` unconditionally at its top (line ~1038), before any tool/schema branch check — i.e. the original bug is back. Re-implementing from scratch using this task''s own comment history as the blueprint: reorder the eager fallback-input preparation so it''s skipped/deferred when the tool-calling branch will be taken, add back a regression test proving a tool-role replayed message doesn''t force eager rendering through a template that might lack tool-role support, and re-verify against the current post-^q3ddgqy code shape (prepareRespondSetup/dispatchGeneration split, not the original monolithic respond()).'
  timestamp: 2026-07-08T16:21:13.869603+00:00
- actor: claude-code
  id: 01kx19p09yd5zkzpkjw3vek6yw
  text: |-
    Re-implemented against the current post-^q3ddgqy code shape (prepareRespondSetup/dispatchGeneration split, not the old monolithic respond()).

    Fix: `RespondSetup.input` and `.effectiveInput` changed from non-optional `LMInput` to `LMInput?`. In `prepareRespondSetup`, added `needsEagerInput = request.enabledToolDefinitions.isEmpty` and gated the `context.processor.prepare(input:)` call (the eager, non-tool-aware render) on it -- skipped entirely when tools are enabled, since `runToolCalling` never reads `setup.input`/`setup.effectiveInput` (only `setup.resolved`; it re-tokenizes independently via its own tool-aware `applyChatTemplate(tools:)` call). `suppressedInput`/`reasoningSetup` were already gated by the narrower `mayRunReasoningPath = enabledToolDefinitions.isEmpty && schema == nil`, a subset of `needsEagerInput`, so no double-gating conflict. `effectiveInput = suppressedInput ?? input` naturally stays nil exactly when tools are enabled (both operands nil), and is guaranteed non-nil on the guided-generation and plain-text branches (schema present or fully unconstrained, both requiring tools empty). In `dispatchGeneration`, the guided-generation and plain-text branches now `guard let`-unwrap `setup.input`/`setup.effectiveInput` with a `preconditionFailure` documenting the invariant (unreachable given the branch guards); the tool-calling branch is untouched.

    Regression test recreated at `Tests/MLXFoundationModelsTests/EagerFallbackPrepareOrderingTests.swift`:
    - `EagerFallbackWeatherArgs` (not `WeatherArgs`, which collides with `ToolCallingSchemaTests.swift`'s existing private `@Generable` type).
    - `EagerFallbackProbeProcessor` (`UserInputProcessor`) throws `EagerFallbackProbeError` iff handed a `.tool`-role message -- this simulates a toolCalling-capable model whose default template can't render tool-role content.
    - `EagerFallbackGenerationProbeModel` (`LanguageModel` + `KVCacheDimensionProvider`) throws `EagerFallbackGenerationProbeError` from `prepare()` immediately, before any `MLXArray` op -- lets the test prove dispatch reached real generation without needing actual model weights or vocab-sized logits.
    - `EagerFallbackByteFallbackTokenizer` mirrors `ToolCallingSchemaTests.makeByteTokenizer()`'s proven `<0xNN>` byte-fallback vocab shape so `runToolCalling`'s xgrammar constraint compile runs against a known-compatible vocab.
    - The `MLXLanguageModel`'s `load:` closure builds a `ModelContainer(context:)` directly (bypassing `LLMModelFactory`/download entirely) with all three fakes constructed fresh inside the closure (no outer capture, avoiding `@Sendable`-closure capture issues).
    - Transcript replays a `.toolCalls`/`.toolOutput` round (tool-role message in history) followed by a fresh prompt, with `enabledTools` non-empty, so `dispatchGeneration` takes the tool-calling branch.
    - Asserts `respond()` throws `EagerFallbackGenerationProbeError` (proof dispatch reached real generation) and explicitly fails the test (`Issue.record`) if it instead throws `EagerFallbackProbeError` (proof the eager fallback blocked it) or anything else unexpected.
    - Manually verified RED: temporarily reverted `needsEagerInput` to `true`, rebuilt, ran the test standalone -- it failed with exactly the expected "ordering fix regressed" issue in 0.011s. Restored the fix, rebuilt, reran -- passes in ~0.03s.

    Verification (SAFE pattern, no `-XCTest` filtering, no hangs):
    - `swift build`: clean, no new warnings.
    - `xcodebuild build-for-testing` (with `-disableAutomaticPackageResolution -skipPackagePluginValidation`): TEST BUILD SUCCEEDED.
    - `xcrun xctest` per bundle, each under `timeout`, unfiltered:
      - CXGrammarTests: 7 tests / 5 suites passed.
      - MLXGuidedGenerationTests: 62 tests / 13 suites passed.
      - MLXFoundationModelsTests: 84 tests / 16 suites passed, including the new "Executor.respond eager fallback vs tool-branch dispatch ordering" suite.
      - MLXLMTests: 244 tests / 19 suites passed.

    Diff scope: `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (52 insertions / 10 deletions) + the one new test file. Not committed -- leaving that to the orchestrator per task scope. Task left in `doing` for review.
  timestamp: 2026-07-08T16:41:22.494624+00:00
position_column: done
position_ordinal: '8280'
title: Executor.respond computes fallback `input` eagerly before tool-branch dispatch; throws if replayed .tool-role message hits a template without tool-role support
---
## Context\nFollow-up from ^q3ddgqy (multi-turn tool calling). `Executor.respond` in `Libraries/MLXFoundationModels/MLXLanguageModel.swift` computes `let input = try await context.processor.prepare(input: userInput)` unconditionally near the top of the perform closure, before deciding whether to take the tool-calling branch or the plain-text branch. `input` (via `effectiveInput = suppressedInput ?? input`) is only actually *used* in the non-tool `else` branch (`runTextGeneration(fallbackInput: effectiveInput, ...)`); the tool-calling branch re-tokenizes independently via `DefaultMessageGenerator().generate(messages:)` + a dedicated tool-aware `applyChatTemplate(tools:)` call.\n\nNow that `TranscriptConverter.mlxMessages(for:)` replays `.toolCalls`/`.toolOutput` transcript entries as assistant/tool-role `Chat.Message`s (^q3ddgqy), a continuation round's `messages` array can contain a `.tool`-role message. The eager, unconditional `prepare()` call renders ALL messages (including `.tool`-role ones) through the model's *default* (non-tool-aware, `tools: nil`) chat template. If a toolCalling-capable model's template lacks a `role == \"tool\"` branch (or raises on an unrecognized role, or requires `tools` to be defined whenever a tool message is present), this call throws — even though the tool-calling branch below would never have used its result.\n\n**Verified NOT currently at risk**: the repo's documented/tested toolCalling model family (Qwen2.5, `mlx-community/Qwen2.5-3B-Instruct-4bit`) handles `role == \"tool\"` in its cached `tokenizer_config.json` chat template unconditionally, independent of whether `tools` is defined. So today's default fixture is safe. The risk is latent, surfacing only for a future/added toolCalling-capable model whose template omits tool-role handling.\n\n## Suggested fix\nOne of:\n- Make the fallback `input`/`suppressedInput`/`reasoningSetup` computation lazy or gated on `request.enabledToolDefinitions.isEmpty`, since it's provably unused when the tool branch is taken.\n- Or add a fallback rendering for the eager path when the template lacks tool-role support (e.g. fold `.tool`-role content into a `.user`-role message for that pre-check only).\n- Or explicitly catch/handle a template-render failure at that call site and treat it as \"fall through to tool-aware rendering\" rather than a hard failure.\n\nNeeds an integration test on a non-Qwen toolCalling-capable model (or a synthetic/local tokenizer fixture with a tool-incapable template) proving the failure mode, then the fix, per TDD.\n\n## Source\nSurfaced by the double-check agent during ^q3ddgqy's adversarial review; logged as accepted risk there rather than fixed in-card (fixing requires reordering Executor.respond's existing control flow, unverifiable offline, and out of that card's stated scope).\n\n## Resolution (2026-07-08)\n\nA first attempt (commit `2b8b318`, described in earlier comments) was lost to an unrelated branch hard-reset before it was pushed. Re-implemented from scratch against the current code shape (the original monolithic `respond()` had since been split into `prepareRespondSetup`/`dispatchGeneration` by ^q3ddgqy's round-6 rework):\n\n- `RespondSetup.input`/`.effectiveInput` are now `LMInput?`, gated by `needsEagerInput = request.enabledToolDefinitions.isEmpty` in `prepareRespondSetup` — the eager default-template render is skipped entirely when tools are enabled.\n- `dispatchGeneration`'s guided-generation and text-generation branches (the only readers of these fields, both unreachable when tools are enabled since the tool-calling branch returns first) unwrap them via a shared `unwrapSetupField<T>(_:fieldName:contextPath:)` helper with a `preconditionFailure` documenting the now-unreachable-by-construction nil case.\n- Recreated the regression test `Tests/MLXFoundationModelsTests/EagerFallbackPrepareOrderingTests.swift` (fake throwing `UserInputProcessor`/`Tokenizer`/`LanguageModel`, no real model weights needed), manually verified RED (revert the gate → fails) then GREEN (restore → passes).\n- Full suite green throughout (396 tests, 0 failures), independently re-verified twice plus an adversarial double-check PASS.\n\n## Review Findings (2026-07-08 11:55)\n\n- [x] `MLXLanguageModel.swift:365` — claimed `tokenizer.eosTokenId` should be `eosToken`. **REJECTED as factually incorrect** — these are two distinct, both-real properties on the `Tokenizer` protocol (`eosToken: String?` vs `eosTokenId: Int?`, the latter numeric); the call site needs the numeric ID. Renaming would pass the wrong type. `eosTokenId` is also used identically in 60+ other files across the repo.\n- [x] `MLXLanguageModel.swift:841`/`854` — duplicated guard/`preconditionFailure` blocks in `dispatchGeneration`. **FIXED** — extracted shared `unwrapSetupField<T>(_:fieldName:contextPath:)` helper, committed `636c3be`.\n\n## Review Findings (2026-07-08 12:10)\n\nClean — 0 findings.