---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx5jfv25zx1163v17c2ngptp
  text: |-
    ## SDK verification (ground truth, per project convention)

    Checked the real macOS 27 SDK's `.swiftinterface` at
    `/Library/Developer/CommandLineTools/SDKs/MacOSX27.0.sdk/System/Library/Frameworks/FoundationModels.framework/Versions/A/Modules/FoundationModels.swiftmodule/arm64e-apple-macos.swiftinterface`.
    Confirmed `ContextOptions` really has both fields exactly as the task described:

    ```
    public struct ContextOptions : Swift.Sendable, Swift.Equatable {
      public var includeSchemaInPrompt: Swift.Bool?
      public var reasoningLevel: ContextOptions.ReasoningLevel?
      public init(includeSchemaInPrompt: Swift.Bool? = nil, reasoningLevel: ReasoningLevel? = nil)
      ...
    }
    ```

    Key discovery that shaped the nil-semantics decision: every schema-taking `LanguageModelSession.respond`/`streamResponse` overload (`respond(to:schema:...)`, `respond(generating:...)`, etc.) defaults its own `contextOptions:` parameter to `ContextOptions(includeSchemaInPrompt: true)` -- NOT the plain `ContextOptions()` (all-nil) default the bare initializer uses. That asymmetry is the field's real contract: going through the framework's own convenience API without overriding `contextOptions` gets `true` by default (the framework already assumes the schema is in the prompt). `nil` only reaches an executor when a caller has stepped off that convenience default. Documented this reasoning directly in `shouldInjectSchemaIntoPrompt`'s doc comment; chose to treat `nil` identically to `false` (conservative: "not confirmed already in the prompt").

    ## Investigation finding (premise correction)

    The task description assumed the adapter "already injects the schema, as it does today." Exhaustive `grep` across `Libraries/`, `Tests/`, `IntegrationTesting/` for `includeSchemaInPrompt`/`SchemaInPrompt`/`injectSchema`/`SchemaEnvelope` found zero hits anywhere (confirmed the task's stated "ZERO references" claim). But tracing further: the guided-generation path (`Executor.respond` -> `dispatchGeneration` -> `runGuidedGeneration`) uses `schemaJSON` **only** to compile the xgrammar constrained-decoding constraint (`prepareConstraintSetup`/`GrammarConstraint(jsonSchema:)`) -- it never touches prompt text. The rendered prompt is 100% `TranscriptConverter.mlxMessages(for: request.transcript)`, which has no schema-awareness at all. **There was no pre-existing schema-in-prompt injection to gate.** This matters for how "preserve current behavior for false/nil" had to be implemented (see below).

    ## Implementation

    - `MLXLanguageModel.Executor.shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: Bool?) -> Bool` -- `includeSchemaInPrompt != true`. Pure, SDK-grounded gate.
    - `MLXLanguageModel.Executor.guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:) -> [Chat.Message]` -- the prompt-assembly seam, wired into `prepareRespondSetup`'s new `RespondSetup.guidedInput` field, itself consumed by `dispatchGeneration`'s guided-generation branch (replacing the old direct use of `setup.input`).
    - Constrained decoding (`schemaJSON` -> xgrammar in `runGuidedGeneration`) is completely untouched -- the flag only ever gates prompt text, never the sampling constraint, per acceptance criteria.

    ## Design revision after adversarial review (important -- read before touching this code again)

    First pass had `guidedGenerationMessages` actually append a NEW system message (`"Respond with a JSON object matching this schema:\n..."`) whenever the flag was `false`/`nil` -- i.e. real new functionality, since there was nothing pre-existing to replicate. The `double-check` agent (via `really-done`) correctly flagged this as **REVISE, High severity**: this silently introduced a brand-new default-on behavior change (extra tokens + a second system message on every guided-generation call, since nobody sets this flag today) that directly contradicts the acceptance criterion's literal text: "with false/nil, current behavior is unchanged." Agreed with this finding -- reverted to a fully behavior-neutral design: **`guidedGenerationMessages` is a genuine no-op today in all three flag states** (true/false/nil all return `messages` unchanged, matching the real pre-existing behavior of "never inject anything"). The gate (`shouldInjectSchemaIntoPrompt`) and the seam wiring (`RespondSetup.guidedInput`, `dispatchGeneration`'s use of it) stay in place and are genuinely exercised in production -- they're just inert today, existing as the one correctly-gated seam any *future* adapter-side schema-in-prompt rendering must route through (so `true` can never end up with two copies once one exists). Second `double-check` pass after this revision: **PASS**, independently re-verified the no-op behavior, the wiring correctness, and re-ran the full suite green.

    ## RED/GREEN evidence

    - RED: `swift test --filter GuidedGenerationSchemaPromptTests` failed to compile (`type 'MLXLanguageModel.Executor' has no member 'guidedGenerationMessages'`/`'shouldInjectSchemaIntoPrompt'`) before any production code existed.
    - GREEN: after implementing, all 5 new tests in `Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift` pass -- asserting zero schema occurrences for true/false/nil, a never-duplicates-existing-schema-text regression case (loops all 3 flag values against a transcript that already embeds the schema), and the pure gate's boolean semantics.
    - Full suite: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` then unfiltered `xcrun xctest .../MLXFoundationModelsTests.xctest` (per this repo's known SwiftPM/Metal-bundling limitation with plain `swift test` -- confirmed reproducing that same limitation here too) -- **153 tests, 33 suites, all passed, zero failures**, including the pre-existing `ContextSizeValidationTests` guided-generation regression tests and the GPU-touching `warmUp` availability test.
    - `swift build` clean (only pre-existing, unrelated resource-bundling warnings for `Documentation.docc`/`README.md`).
    - Confirmed (via `git stash` + rebuild) that a separate `IntegrationTesting` xcodeproj build failure (`FMTestHelpers.swift`: `.updateUsage`/`#require` macro-resolution errors) is **pre-existing on the base commit**, unrelated to this change.

    Files touched: `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (gate + seam + wiring + docs), `Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift` (new).
  timestamp: 2026-07-10T08:32:15.429304+00:00
- actor: claude-code
  id: 01kx5m604z232zj2nbzys3st6b
  text: |-
    ## Strengthened the test to exercise the real seam (per verifier feedback)

    Investigated feasibility of the two preferred options before choosing:

    1. **Option 1 (drive through `prepareRespondSetup` directly)**: infeasible without a scope-creeping visibility change. `prepareRespondSetup` is `private func` on `Executor`; Swift `private` restricts access to the enclosing declaration/file regardless of `@testable import` (that only lifts the `internal`→cross-module boundary, not `private`). Loosening it to `internal` purely to satisfy a test would be an unjustified production-code visibility change for a method that's intentionally scoped tightly (unlike `guidedGenerationMessages`/`shouldInjectSchemaIntoPrompt`, which are already `internal` `static func` -- that's why the original isolated tests could call them at all). Did not make this change.
    2. **Option 2 (drive through `Executor.respond` with a fake tokenizer/processor/model)**: fully feasible and is what I implemented, following the exact pattern already established by `ContextSizeValidationTests.swift`/`EagerFallbackPrepareOrderingTests.swift` in this same test target (`makeMLXExecutor`, `makeExecutorRequest`, a temp-directory-backed `MLXLanguageModel` with a fake `UserInputProcessor`/`LanguageModel`/`Tokenizer`).

    ### What changed in `Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift`

    - Kept the original 5 tests (renamed suite to `GuidedGenerationSchemaPromptTests (isolated)`) as direct unit tests of the pure helpers -- legitimate in their own right, but rewrote the file's top doc comment so it no longer claims these exercise "the real seam wired into production `respond()`". It now explicitly says what they do and don't cover.
    - Added a new suite, `GuidedGenerationPromptSeamTests`, with 4 new tests that drive `Executor.respond(to:model:streamingInto:)` end-to-end:
      - Builds a real `Transcript` (optionally embedding the schema's own `SchemaConverter.encodeToJSON` rendering in the prompt text, simulating `includeSchemaInPrompt == true`'s real contract) and a real schema-guided request via `makeExecutorRequest`.
      - `messages` are rendered by the real `TranscriptConverter.mlxMessages`, not hand-built.
      - A new `GuidedPromptSeamCaptureProcessor` (a fake `UserInputProcessor`, same idiom as `CountingTokenizer` in `TokenizerBiasCacheTests.swift`) captures the actual `[Chat.Message]` array `prepareRespondSetup`'s guided-generation seam hands to `UserInputProcessor.prepare` -- the real tokenizable content, not a reimplemented check -- and returns a fixed over-long token count.
      - A `max_position_embeddings: 1` fake model config forces `dispatchGeneration`'s guided-generation branch to throw `contextSizeExceeded` immediately after the real seam runs, before any real generation, mirroring `ContextSizeValidationTests`' established technique for cutting off dispatch cheaply.
      - Assertions count schema occurrences in the captured messages for `includeSchemaInPrompt` == true/false/nil, both with and without the schema already embedded in the transcript.

    ### Verification that the new tests are not vacuous

    Temporarily (and only locally, never committed) injected a fake regression into `guidedGenerationMessages` (`return messages + [.system("Schema: \(schemaJSON)")]` on the "should inject" branch) to confirm the new seam-level tests actually fail when the wiring misbehaves, not just the isolated ones. Confirmed: the `false`/`nil` seam tests failed with `occurrences == 2` (expected 1) while the `true` and true-with-no-schema tests still correctly passed (unaffected by that branch) -- proving the new tests genuinely exercise and validate the real dispatch path, not merely the isolated helper. Reverted the injected regression immediately after confirming (production code is back to the exact pre-existing no-op state; verified via `git diff`/grep that no debug/regression code remains).

    ### Final verification
    - `swift build`: clean (only pre-existing, unrelated warnings: `Documentation.docc`/`README.md` resource bundling, and a pre-existing `LanguageModelCapabilities(capabilities:)` deprecation warning at `MLXLanguageModel.init` -- neither touched by this change).
    - `xcodebuild build-for-testing` (per this repo's SwiftPM/Metal-bundling limitation) then unfiltered `xcrun xctest MLXFoundationModelsTests.xctest`: **157 tests, 34 suites, all passed** (up from 153/33 -- net +4 tests, +1 suite, since 4 of the new tests are genuinely new coverage and the isolated suite's 5 pre-existing tests are unchanged in count).

    Left the task in `doing` per scope instructions -- no column move, no commit.
  timestamp: 2026-07-10T09:01:50.111681+00:00
- actor: claude-code
  id: 01kx5nedt2yd4ydt4hg67w6vga
  text: |-
    Fixed the review finding: extracted the schema-occurrence counting logic that was duplicated between `schemaOccurrences(in:)` (GuidedGenerationSchemaPromptTests) and the inline reduce in `schemaOccurrencesAtRealSeam` (GuidedGenerationPromptSeamTests) into a single file-level `private func countSchemaOccurrences(in messages: [Chat.Message], schemaJSON: String) -> Int` in Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift, placed after the shared private test-support types (GuidedPromptSeamCaptureProcessor etc.) and before the GuidedGenerationSchemaPromptTests suite doc comment.

    - `schemaOccurrences(in:)` now delegates: `countSchemaOccurrences(in: messages, schemaJSON: schemaJSON)`.
    - `schemaOccurrencesAtRealSeam` now returns `countSchemaOccurrences(in: captured, schemaJSON: schemaJSON)` directly instead of re-implementing the reduce inline.
    - Kept the existing `@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)` annotation on the new helper for consistency with the file's established pattern, even though `Chat.Message` itself isn't availability-gated.

    Verification:
    - `swift build` — clean, exit 0.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` — TEST BUILD SUCCEEDED.
    - Unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (wrapped in `timeout`, no `-XCTest` filter, no `tail` pipe): 157/157 tests passed across 34 suites, including both "Guided-generation prompt-assembly helpers honor ContextOptions.includeSchemaInPrompt (isolated)" and "Executor.respond guided-generation prompt seam honors ContextOptions.includeSchemaInPrompt" suites — the exact suites whose call sites now share the extracted helper.

    Checked off the finding in the task description. Leaving task in `review` per scope — no commit made (orchestrator's job).
  timestamp: 2026-07-10T09:23:54.818323+00:00
- actor: claude-code
  id: 01kx5q1aa2vve4k2kwqry2689q
  text: |-
    Fixed the 2026-07-10 04:24 review finding: renamed `schemaJSON` -> `schemaJson` throughout Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift (the file-level `countSchemaOccurrences` parameter, the `GuidedGenerationSchemaPromptTests.schemaJson` static constant, the `makeRequest`/`schemaOccurrencesAtRealSeam` local params/vars, all call sites, and the doc comments referencing them).

    Investigated first: `MLXLanguageModel.Executor.guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:)` in MLXLanguageModel.swift is a real production API whose external parameter label is literally `schemaJSON` (unrelated to this finding's scope -- out of bounds). Left those 4 call-site labels (`schemaJSON:`) untouched to avoid a compile break, but renamed the argument expressions passed to them (`Self.schemaJson`) since those refer to the test's own renamed static constant. No production file was touched.

    Verified: `swift build` clean; `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` succeeded; unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (captured to a file, not piped through tail) shows 157/157 tests passed in 34 suites, including both `GuidedGenerationSchemaPromptTests` and `GuidedGenerationPromptSeamTests`.
  timestamp: 2026-07-10T09:51:42.402235+00:00
- actor: claude-code
  id: 01kx61t2jpfjs1mss2hrnyctfd
  text: 'Commit 9f4c9b2 ("fix(mlx-fm): honor ContextOptions.includeSchemaInPrompt in guided generation") plus GuidedGenerationSchemaPromptTests landed — this task''s core is implemented. Before closing, resolve the review finding on ^d54djxh at MLXLanguageModel.swift:2353: the engine reports a guard-else at the schema-in-prompt seam whose branches both return `messages` unchanged, i.e. possibly a no-op conditional on one path. Verify against this task''s acceptance criteria (schema appears exactly once when true; unchanged behavior when false/nil; grammar constraint unaffected) and either fix the dead seam or document why the identical branches are intentional.'
  timestamp: 2026-07-10T12:59:59.446952+00:00
position_column: done
position_ordinal: '9380'
title: Honor ContextOptions.includeSchemaInPrompt in guided generation (protocol fidelity gap)
---
## What\nOS27 protocol gap found by auditing the SDK surface against the adapter: `FoundationModels.ContextOptions` has TWO fields — `reasoningLevel` (fully handled in MLXLanguageModel.swift) and `includeSchemaInPrompt: Bool?` (ZERO references in Libraries/, Tests/, IntegrationTesting/). The framework/app uses it to signal whether the generation schema is already embedded in the prompt; the executor must not make its own injection decision when told.\n\nIn Libraries/MLXFoundationModels/MLXLanguageModel.swift's guided-generation path: read `request.contextOptions.includeSchemaInPrompt` and gate the adapter's own schema-envelope injection accordingly — when the app says the schema is already in the prompt, do NOT inject a second copy (wasted context tokens + model confusion); when false/nil, preserve current behavior (adapter injects). Trace exactly where the schema text currently enters the prompt (SchemaConverter / guided prompt assembly) and thread the flag there. Document the chosen nil-semantics against Apple's docs for the field.\n\n## Acceptance Criteria\n- [x] With includeSchemaInPrompt == true, the tokenized prompt contains exactly one schema rendering (the transcript's own), and the adapter injects nothing extra — asserted at the prompt-construction seam\n- [x] With false/nil, current behavior is unchanged (regression-guarded by existing guided-generation tests)\n- [x] Constrained decoding (grammar) still applies in BOTH cases — the flag governs prompt text only, not the sampling constraint\n\n## Tests\n- [x] Unit tests at the prompt-assembly seam (Tests/MLXFoundationModelsTests): schema appears once vs twice across the flag's three values\n- [x] Existing guided-generation integration suites stay green\n- [x] `swift test --filter MLXFoundationModelsTests` zero failures\n\n## Workflow\n- Use `/tdd` — write failing tests first, then implement to make them pass.\n\n## Resolution notes (commit 9f4c9b2)\nInvestigation found the feared double-injection bug doesn't currently exist — `schemaJSON` only ever fed the xgrammar grammar constraint, never prompt text (confirmed via independent whole-repo search). Added `shouldInjectSchemaIntoPrompt`/`guidedGenerationMessages` helpers wired live into `prepareRespondSetup`/`dispatchGeneration` (confirmed executed on every real guided-generation call, not dead code) — currently a behaviorally-neutral no-op, but correctly-gated scaffolding for if/when schema-in-prompt injection is ever added. New `GuidedGenerationPromptSeamTests` suite drives `Executor.respond(...)` through the real production seam. Two independent verification rounds, including an independently-reproduced red→green regression-detection cycle (injected a real duplication bug, confirmed the new seam tests caught it with the exact predicted signature, reverted, confirmed 157/157 green).\n\n## Review Findings (2026-07-10 04:09) — fixed, commit 3d1c2ad\n- [x] Deduped schema-occurrence counting logic between `schemaOccurrences(in:)` and `schemaOccurrencesAtRealSeam` via shared `countSchemaOccurrences(in:schemaJSON:)` helper.\n\n## Review Findings (2026-07-10 04:24)\n\n- [x] `Tests/MLXFoundationModelsTests/GuidedGenerationSchemaPromptTests.swift` — `schemaJSON` (parameter names, local variables, a static constant) mixes case for the JSON acronym; should be `schemaJson` per this project's established convention. New test file added by this task's own commit — fair game to fix directly. Rename throughout: the `countSchemaOccurrences(in:schemaJSON:)` parameter (~line 86-88), all call sites (~103, 126), the static constant (~line 116), and the `schemaOccurrencesAtRealSeam` local (~171, 176). Fixed: renamed to `schemaJson` everywhere in the test file except the 4 call sites of the real production API `MLXLanguageModel.Executor.guidedGenerationMessages(from:schemaJSON:includeSchemaInPrompt:)`, whose external label is genuinely `schemaJSON` in production code and must not change. 157/157 tests green.