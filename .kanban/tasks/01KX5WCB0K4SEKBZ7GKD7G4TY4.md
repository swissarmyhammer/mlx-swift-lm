---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxaf8g1s94h9wpq6ghj3v2yd
  text: |-
    Implementation approach: per-target duplication of TestBootstrap.swift (not a shared SwiftPM target) — matches the existing MLXFoundationModelsTests precedent, and the bootstrap file is small/self-contained enough that a shared target would be over-abstraction for 3 call sites. Package.swift untouched.

    Added `Tests/MLXLMTests/TestBootstrap.swift` and `Tests/MLXGuidedGenerationTests/TestBootstrap.swift` (byte-identical `MetalLibraryTestBootstrap` enum, copied from the original in MLXFoundationModelsTests).

    Wired `_ = MetalLibraryTestBootstrap.ensureColocatedMetallib` into every GPU/MLXArray-touching test entry point in both targets, using whichever pattern matches the existing test code shape:
    - Swift Testing `@Suite`/struct suites → added/extended `init()`.
    - `XCTestCase` subclasses → added `override func setUp()` (or `override public func setUp()` for `public class`).
    - Free-standing top-level `@Test func` files (no enclosing struct) → inserted the call as the first statement of each GPU-touching function body: KVCacheTests.swift (all 34 functions — every one touches MLXArray), Gemma4EmitDrafterStateTests.swift (all 4), Gemma4AssistantDraftModelTests.swift (2 of 3 — skipped the config-decode-only test), MTPSpeculativeTokenIteratorTests.swift (8 of 9 — skipped the pure-state-dictionary no-MLXArray test), MTPDrafterModelTests.swift (4 of 8 — skipped 4 pure-Swift-state tests), SSMTests.swift (its 1 function).

    Determined which files/suites needed the call empirically (grepped for MLXArray construction/`.item(`/`.asArray(`/`eval(`/`MLXRandom` across both targets) rather than guessing, mirroring the ~9/34 selective ratio the original MLXFoundationModelsTests fix used rather than blanket-adding to every suite.

    Verification (all green):
    - `swift build` clean.
    - `swift test --filter MLXLMTests`: 245/245 tests, 19 suites, no metallib crash (previously aborted mid-run).
    - `swift test --filter MLXGuidedGenerationTests`: 63/63 tests, 14 suites, no crash (previously aborted).
    - `swift test --filter MLXFoundationModelsTests` (untouched): still 207/207 green — no regression.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS'`: TEST BUILD SUCCEEDED.
    - `xcrun xctest` on both `MLXLMTests.xctest` and `MLXGuidedGenerationTests.xctest`: 245/245 and 63/63 pass — the project's mandated CI-equivalent safe pattern shows no regression.
    - One numerical-tolerance flake seen once in Gemma4ChunkedPrefillTests ("Chunked and single-pass prefill agree on logits"), confirmed pre-existing/unrelated to this change — reran clean twice after (0 failures), both isolated and full-suite.

    Adversarial double-check dispatched to verify selectivity judgment calls and diff scope before closing out.
  timestamp: 2026-07-12T06:12:01.209436+00:00
- actor: claude-code
  id: 01kxahrzv580f89j967wdz68ez
  text: |-
    Adversarial double-check verdict: REVISE, but only on the accuracy of my own progress-comment counts above — no code defect found in the actual bootstrap wiring, no crash risk, no unrelated files touched, all three insertion patterns (init(), setUp(), first-statement) correctly applied and non-duplicated.

    Corrections to the earlier comment's selectivity counts:
    - `Tests/MLXLMTests/Gemma4AssistantDraftModelTests.swift` has 4 `@Test` functions, not 3. 2 are skipped: the config-decode-only test (as stated) AND `testGemma4AssistantDraftModelInstantiatesAndShape` (constructs a real `Gemma4AssistantDraftModel` but never calls `.item(`/`.asArray(`/`eval(` — MLX's lazy-eval model means array construction alone never dispatches a Metal kernel, so no crash risk; independently re-verified by the double-check agent). Correct count: 2 of 4.
    - `Tests/MLXLMTests/MTPDrafterModelTests.swift` has 7 `@Test` functions, not 8. 3 are skipped (`testMTPDrafterContextRoundtrip`, `testMTPDrafterContainerPerform`, `testMTPEmitFlagKeyDefaultsToFalse`), all confirmed to touch no MLXArray. Correct count: 4 of 7.

    Both corrections are arithmetic-only; the underlying selectivity judgment (which functions actually needed the bootstrap call) was already correct in the code itself. No further code changes needed. Task remains green per all verification evidence above (swift build, swift test on both targets, xcodebuild build-for-testing + xcrun xctest safe pattern, MLXFoundationModelsTests regression check). Leaving in `doing` for review.
  timestamp: 2026-07-12T06:55:58.821034+00:00
position_column: doing
position_ordinal: '80'
title: Apply the swift-test metallib bootstrap fix to MLXLMTests and MLXGuidedGenerationTests
---
## What
`23ff1zx` fixed `swift test`'s "Failed to load the default metallib" crash for `MLXFoundationModelsTests` by adding a once-only bootstrap (`Tests/MLXFoundationModelsTests/TestBootstrap.swift`, `MetalLibraryTestBootstrap.ensureColocatedMetallib`) that symlinks `mlx.metallib` next to the running test binary, satisfying mlx-swift's `Cmlx` loader's first probe path.

Confirmed during that task's verification: `MLXLMTests` and `MLXGuidedGenerationTests` still crash identically under plain `swift test` (untouched, out of scope for that task's title/acceptance criteria which scoped it to `MLXFoundationModelsTests`).

## Acceptance Criteria
- [ ] Apply the same (or a shared/extracted) bootstrap mechanism to `Tests/MLXLMTests` and `Tests/MLXGuidedGenerationTests` so `swift test --filter MLXLMTests` and `swift test --filter MLXGuidedGenerationTests` no longer abort on GPU eval.
- [ ] Consider extracting the bootstrap logic to a shared location (e.g. a small internal test-support target, or duplicated per-target static like the original if a shared target is overkill) rather than copy-pasting verbatim across 3 test targets.
- [ ] No regression to the existing `xcodebuild build-for-testing` + `xcrun xctest` safe pattern already used elsewhere in this project.

## Scope
`Tests/MLXLMTests/`, `Tests/MLXGuidedGenerationTests/`. Not urgent/blocking — this is a test-infrastructure convenience (enables plain `swift test` for these bundles), the project's mandated safe test-invocation pattern already works around this via `xcodebuild`/`xcrun xctest`.