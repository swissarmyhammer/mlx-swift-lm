---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx5v9qb6da9pmk5vs2zx369w
  text: |-
    Root-cause fix implemented (not CPU-pin fallback). Summary:

    INVESTIGATION
    - Reproduced the crash directly: wrote MetalLibraryBootstrapTests.swift (a GPU-device MLXArray slice/concat/compare test) WITHOUT any bootstrap first, ran `swift test --filter MetalLibraryBootstrapTests`. Confirmed it aborts mid-run: "MLX error: Failed to load the default metallib. library not found library not found library not found library not found  at .../mlx-c/mlx/c/stream.cpp:115", with no per-test summary line (SwiftPM just reports "Note: Some test targets reported failures"), exit code 1. This is the RED state.
    - Confirmed the probe order in `.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp`'s `load_default_library`: (1) `current_binary_dir()/mlx.metallib`, (2) `current_binary_dir()/Resources/mlx.metallib`, (3) SwiftPM bundle scan (mainBundle then `Bundle.allBundles`/`allFrameworks`, bundle name from `SWIFTPM_BUNDLE = "mlx-swift_Cmlx"`, defined in mlx-swift's Package.swift), (4) `current_binary_dir()/Resources/default.metallib`, (5) CWD-relative `default.metallib` (`METAL_PATH`).
    - `current_binary_dir()` (mlx/backend/common/utils.cpp) is a `dladdr` lookup on `&current_binary_dir` itself. Since `Cmlx` is a plain `Target.target` (statically linked, not a dylib/framework), this resolves to whatever Mach-O the code got linked into -- i.e. the actual running xctest binary at `<Target>.xctest/Contents/MacOS/<Target>`.
    - Confirmed via `find .build -iname "*.metallib"` that the built metallib lives at `<Target>.xctest/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` for each of MLXFoundationModelsTests.xctest, MLXGuidedGenerationTests.xctest, MLXLMTests.xctest (plus a top-level copy) -- two directory levels from `Contents/MacOS/`, matching the task's description of why every probe misses.

    IMPLEMENTATION
    - Tests/MLXFoundationModelsTests/TestBootstrap.swift (new): `enum MetalLibraryTestBootstrap` with a once-only `static let ensureColocatedMetallib: Void` that (a) gets the running test binary's directory via `Bundle(for: BundleAnchor.self).executableURL` (falls back to `bundleURL/Contents/MacOS` if `executableURL` is ever nil), (b) locates `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` -- first via the test bundle's own `Contents/Resources/` (fast path, matches what we found on disk), then falling back to scanning `Bundle.allBundles`/`Bundle.allFrameworks` (bundleURL and resourceURL) the same way mlx's own `load_swiftpm_library` does, for robustness across build layouts, (c) creates a `mlx.metallib` symlink next to the binary if one doesn't already exist (idempotent), satisfying probe #1. Best-effort: any failure just logs to stderr and lets the original mlx error surface as before.
    - Tests/MLXFoundationModelsTests/MetalLibraryBootstrapTests.swift (new): a `@Suite` whose `init()` triggers the bootstrap, with one `@Test` that builds two `MLXArray`s via `MLXArray(0..<12, [3,4])`/`MLXArray(12..<24, [3,4])`, slices both (`a[1..., 0..<2]`, `b[0..<2, 2...]`), concatenates along axis 0, calls `eval(combined)`, and asserts both shape and elementwise equality (`.== `/`.all().item()`) against the expected values -- proving real GPU-device eval, not just construction.
    - Tests/MLXFoundationModelsTests/PromptCacheTestSupport.swift: added a doc-comment note (no code change) explaining this file's fixtures don't evaluate MLXArrays today, and pointing future GPU-touching fixtures at the new bootstrap instead of CPU-pinning.
    - Updated memory note `swiftpm-test-gpu-metallib-limit` (and the MEMORY.md index line) to describe the fix instead of the old "filter suites + merge profraw manually" workaround.

    TEST EVIDENCE
    - RED confirmed as above.
    - GREEN: `swift test --filter MetalLibraryBootstrapTests` passes; verified the symlink actually gets created at `.build/out/Products/Debug/MLXFoundationModelsTests.xctest/Contents/MacOS/mlx.metallib` -> `.../Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`. Ran a second time without rebuilding to confirm idempotency (symlink already present, no-op, still passes, faster).
    - Full target: `swift build --build-tests` clean, then `swift test --filter MLXFoundationModelsTests` (fresh, symlink removed first) -> "Test run with 158 tests in 35 suites passed after 0.525 seconds" -- zero failures, full printed summary, no abort.
    - xcodebuild safe-pattern regression check: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` (TEST BUILD SUCCEEDED) then unfiltered `xcrun xctest <DerivedData path>/MLXFoundationModelsTests.xctest` -> also "158 tests in 35 suites passed". Bootstrap ran there too (symlink created harmlessly alongside whatever probe already worked) -- confirms idempotent/no-op-safe behavior under xcodebuild, per acceptance criteria.
    - `mcp__sah__review` flagged two naming-convention findings (`modelID`->`modelId` in PromptCacheTestSupport.swift, `symlinkURL`->`symlinkUrl` in TestBootstrap.swift). Declined both as false positives: `modelID` is pre-existing code in that file I didn't touch and matches the dominant convention across PromptCache/MLXFoundationModels (`PromptCache.resolve(modelID:)`, `.store(modelID:)`, etc.); `symlinkURL`/`metallibURL` match both this codebase's overwhelming `fooURL` convention (`configurationURL`, `weightsURL`, `destinationURL`, `cacheURL` elsewhere in Libraries/) and Apple's own Foundation API this code directly calls (`Bundle.bundleURL`, `.resourceURL`, `.executableURL`).

    OUT OF SCOPE / DISCOVERED
    - A full unfiltered `swift test` (whole package) still shows the SAME pre-existing metallib crash in MLXLMTests and MLXGuidedGenerationTests -- untouched by this task (title/acceptance criteria scope this fix to MLXFoundationModelsTests only). Those targets' xctest bundles have the identical layout issue and would likely benefit from the same bootstrap pattern; flagging as a candidate follow-up task, not fixed here.
    - No pre-existing GPU-touching unit tests were found disabled/CPU-pinned in Tests/MLXFoundationModelsTests to "re-enable" per the task's optional follow-up -- the memory note's prior "Pure-logic suites run fine" already reflected reality; the new MetalLibraryBootstrapTests.swift is the first GPU-eval test in this target.

    Adversarial double-check (via really-done's gate) launched; will report verdict when it returns. Leaving task in `doing` for `/review` per process.
  timestamp: 2026-07-10T11:06:12.198760+00:00
- actor: claude-code
  id: 01kx5wbc183j8x84anfm9fgf5g
  text: |-
    Adversarial double-check verdict: PASS. Independent re-verification performed by the reviewer (not just trusting my summary):
    - Re-read TestBootstrap.swift, MetalLibraryBootstrapTests.swift, the PromptCacheTestSupport.swift diff, and the memory note in full -- all accurate.
    - Manually recomputed the slice/concat/compare arithmetic in the GPU test independently and confirmed it passes for the right reason.
    - Cross-checked mlx-swift's actual device.cpp probe order/current_binary_dir() semantics against the bootstrap's assumptions -- correct.
    - Deleted the mlx.metallib symlink and reran `swift test --filter MetalLibraryBootstrapTests` independently: symlink regenerated, test passed, no stderr error.
    - Reran `swift test --filter MLXFoundationModelsTests` independently: 158 tests / 35 suites, clean pass, matching my claimed result.
    - Independently confirmed the underlying bug is real and still reproduces in the untouched MLXGuidedGenerationTests target -- validates the crash is genuine and that leaving other targets unfixed is a pre-existing, out-of-scope condition, not a regression.
    - Re-read this kanban task in full and confirmed all 4 acceptance criteria are satisfied; the "Follow-up on success" re-enable-GPU-tests line is advisory, not a listed acceptance criterion, and isn't gating this task.
    - Confirmed `URL`-suffixed and `modelID`-style identifiers both dominate Libraries/, validating the decision to decline the review tool's `symlinkUrl`/`modelId` renames.
    - Confirmed git status/diff shows only the intended files changed after its own experiments were cleaned up.

    One minor non-blocking observation (not a defect in this change, no action taken): Tests/MLXLMTests/LFM2BidirectionalTests.swift (untouched, out of scope) has a header comment claiming swift test "can't bundle mlx-swift's Metal library" -- this fix shows that's no longer categorically true, just still true for that untouched target. Worth revisiting if/when the same bootstrap pattern is applied there.

    Task is green and ready for /review. Leaving in `doing` per process (not moving columns).
  timestamp: 2026-07-10T11:24:34.728923+00:00
- actor: claude-code
  id: 01kx5x2en938dk993cfq3k71vv
  text: |-
    Follow-up fix per orchestrator directive: added stderr diagnostics at the two previously-silent guard-let failure points in TestBootstrap.swift's installSymlinkIfNeeded(), matching the existing createSymbolicLink-catch log style (FileHandle.standardError.write(Data(...utf8))):
    - currentTestBinaryDirectory() returning nil -> "MetalLibraryTestBootstrap: could not determine the running test binary's directory; GPU-device tests may crash with \"Failed to load the default metallib\"."
    - locateDefaultMetallib(...) returning nil -> "MetalLibraryTestBootstrap: could not locate mlx-swift_Cmlx.bundle/default.metallib; GPU-device tests may crash with \"Failed to load the default metallib\"."

    Both remain no-op/graceful-failure paths (still `return`, no throw, no behavior change) — purely added a diagnostic trail.

    VERIFICATION
    - `swift build` clean.
    - `swift build --build-tests` clean, then `swift test --filter MLXFoundationModelsTests` -> "Test run with 158 tests in 35 suites passed" (matches expected count, no new stderr output on the happy path).
    - xcodebuild safe pattern: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` -> TEST BUILD SUCCEEDED, then unfiltered `xcrun xctest .../MLXFoundationModelsTests.xctest` -> "Test run with 158 tests in 35 suites passed" (same count).
    - Simulated the currentTestBinaryDirectory() nil path directly: temporarily hardcoded it to `return nil`, rebuilt, ran `swift test --filter MetalLibraryBootstrapTests` -> confirmed the new stderr line fired verbatim ("MetalLibraryTestBootstrap: could not determine the running test binary's directory; GPU-device tests may crash with \"Failed to load the default metallib\"."), test still passed (a stale symlink from a prior run satisfied probe #1 despite the bootstrap taking the early-return path, proving the log fires independent of whether the crash actually reproduces). Reverted the temporary change immediately after, rebuilt, and reran `swift test --filter MLXFoundationModelsTests` to confirm back to clean 158/35 pass with no stray output.

    Leaving task in `doing` per process (not moving columns, not committing).
  timestamp: 2026-07-10T11:37:11.081072+00:00
- actor: claude-code
  id: 01kx5xtfqf6pab4q9ap6dxh9fc
  text: |-
    Addressed both review findings in Tests/MLXFoundationModelsTests/TestBootstrap.swift:

    1. Added a doc comment above `enum MetalLibraryTestBootstrap` explaining what it does, pointing to the file-header writeup for the root-cause detail.
    2. Extracted the duplicated stderr-logging blocks in the two guard-else paths of `installSymlinkIfNeeded()` into a shared `private static func logError(_ message: String)` helper (writes the message + trailing newline to stderr). Both call sites now just pass their message string; removed the now-redundant trailing `\n` from the string literals since the helper appends it.

    Verification:
    - `swift build` — clean, exit 0 (only pre-existing unrelated "unhandled resource" warnings).
    - `swift test --filter MLXFoundationModelsTests` — 158 tests / 35 suites passed.
    - Mandated safe pattern: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` (TEST BUILD SUCCEEDED) then unfiltered `xcrun xctest .../MLXFoundationModelsTests.xctest` under `timeout` — 158 tests / 35 suites passed, matching the swift test count.

    Left the task in `review` per scope discipline; not committing (orchestrator handles that).
  timestamp: 2026-07-10T11:50:18.607208+00:00
- actor: claude-code
  id: 01kx5y969wc3jv98kg4t3b3hgc
  text: |-
    Fixed the 2026-07-10 06:51 review finding: `ensureColocatedMetallib`'s catch handler in TestBootstrap.swift now calls the shared `logError("MetalLibraryTestBootstrap: \(error)")` helper instead of spelling out the stderr-logging pattern inline, matching `installSymlinkIfNeeded`'s two call sites.

    Verified:
    - `swift build` — clean (only pre-existing unrelated resource warnings).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` — TEST BUILD SUCCEEDED.
    - Unfiltered `xcrun xctest MLXFoundationModelsTests.xctest` (wrapped in timeout) — 158 tests in 35 suites passed.
    - `swift test --filter MLXFoundationModelsTests` — 158 tests in 35 suites passed, matching count.

    Task left in `review` per scope; no commit made (orchestrator handles that).
  timestamp: 2026-07-10T11:58:20.476072+00:00
depends_on:
- 01KX3MZDHS31HAGZ36N34EPSM4
position_column: done
position_ordinal: '9580'
title: Fix metallib loading under swift test (root fix; CPU-pin only as documented fallback)
---
## What\nRoot-cause fix, not a workaround. `swift test` crashes on any GPU eval (\"Failed to load the default metallib\") because of PATH RESOLUTION, not a missing capability: the built metallib exists at `.build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`, but mlx's loader (mlx-swift Source/Cmlx/mlx/mlx/backend/metal/device.cpp, load_default_library) probes (1) `current_binary_dir()/mlx.metallib`, (2) `Resources/mlx.metallib`, (3) the SwiftPM bundle, (4) framework Resources, (5) CWD-relative `default.metallib` — and under swift test the executing binary is inside `MLXFoundationModelsTests.xctest/Contents/MacOS/`, two levels away from the bundle, so all five miss. (xcodebuild test layouts satisfy the probes, which is why IntegrationTesting works.)\n\nImplement: a test-support bootstrap in Tests/MLXFoundationModelsTests (e.g. a once-only static in PromptCacheTestSupport.swift or a dedicated TestBootstrap.swift) that locates the Cmlx resource bundle (derive from `Bundle(for: <any test class>).bundlePath` → up to the Products dir → `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`; fall back to scanning `Bundle.allBundles`) and creates a symlink `mlx.metallib` beside the executing test binary (probe #1) if absent. Idempotent, no-op under xcodebuild where loading already works.\n\nFallback ONLY if the symlink approach proves infeasible (e.g. sandboxed test dirs): pin tensor-evaluating suites to the CPU device — and document why the root fix failed.\n\nFollow-up on success: update the memory note swiftpm-test-gpu-metallib-limit and re-enable/verify the previously-crashing GPU-touching unit tests (model warmUp/availability) under plain `swift test`.\n\n## Acceptance Criteria\n- [x] A test that slices, concatenates, and compares MLXArrays on the DEFAULT (GPU) device passes under plain `swift test`\n- [x] `swift test --filter MLXFoundationModelsTests` no longer aborts with the metallib error — the full target runs to a reported summary\n- [x] Bootstrap is idempotent and harmless under xcodebuild (IntegrationTesting scheme still passes)\n- [x] Memory note and PromptCacheTestSupport doc comments updated to describe the fix (not the workaround)\n\n## Tests\n- [x] New bootstrap-proving test in Tests/MLXFoundationModelsTests: GPU-device tensor eval succeeds\n- [x] `swift test --filter MLXFoundationModelsTests` zero failures with a printed summary (no process abort)\n\n## Workflow\n- Use `/tdd` — write failing tests first, then implement to make them pass.\n\n## Resolution notes (commit 3049af1)\nRoot fix implemented and independently verified across two full rounds (including reproducing the original crash directly and a simulated-failure test proving the diagnostic-logging addition actually fires). Follow-up tasks created: ^d7g4ty4 (apply same bootstrap to MLXLMTests/MLXGuidedGenerationTests) and ^sd05wkh (unrelated, larger modelID->modelId rename surfaced during a different task's review).\n\n## Review Findings (2026-07-10 06:38) — fixed, commit 84424b9\n- [x] Added doc comment to `MetalLibraryTestBootstrap` enum.\n- [x] Deduped stderr-logging blocks in `installSymlinkIfNeeded` via shared `logError` helper.\n\n## Review Findings (2026-07-10 06:51)\n\n- [x] `Tests/MLXFoundationModelsTests/TestBootstrap.swift:47` — `ensureColocatedMetallib`'s catch handler spells out the stderr-logging pattern inline instead of calling the `logError` helper (defined for `installSymlinkIfNeeded`'s two call sites, but not used here too). Call `logError(\"MetalLibraryTestBootstrap: \\(error)\")` instead of the inline write, for consistency throughout the file.