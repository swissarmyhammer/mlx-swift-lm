---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx4y8awax3jk26wngh6prpwv
  text: |-
    Completed the refactor of GuidedGenerationLoop.swift. Pure internal restructuring; public `run()` signature unchanged, no behavior change.

    **Extraction shape:**
    - `run()`'s main while-loop body (previously ~415 lines with 4-5-level nesting) is now a short orchestrator calling into:
      - `prefillAndGetLogits()` — the prompt-prefill + `model.prepare`/switch logic.
      - `LoopState` (private struct) — bundles `cache`/`modelState`/`logits`/`mask`/`maskArray`/`detokenizer`/`tokenCount`/`accumulatedText`/`whitespaceTracker` so helpers don't need ballooning parameter lists.
      - `applyBiasAndSample()` — zone-policy bias computation (normal/soft/hard budget zones + whitespace suppression) + sampling + whitespace-tracker update. This collapses the hard-zone branch that was nested 4-5 levels deep.
      - `emitToken()` — shared detokenize+accumulate+emit helper, reused by both the sampled-token yield and the FF-token emission loop. Preserves the original's exact `tokenCount` accounting: only incremented on the non-stopping path (matches the original's `tokenCount += 1` placement *after* the `if !emit(text) { break }`).
      - `processFastForwardTokens()` — FF-token emission loop + FF batch model-forward loop + mask advance. Collapses both previously deeply-nested FF blocks into one call. Early-return semantics preserved exactly: hitting `maxTokens` mid-batch or an `emit`-stop returns `true` immediately, skipping the model-forward/mask-advance work entirely (matching the original's `shouldStopAfterFF` short-circuit).
      - `advanceSingleSampledToken()` — the non-FF single-token forward-pass + mask-advance branch (the old `else` branch), kept separate from FF handling since it has none of the batch/emission concerns.
      - `advanceMaskOnCPUWhileGPURuns()` (already extracted pre-existing, from qawe2hb) — left untouched.
      - `logMaskSnapshot()`, `logStopReason()`, `logProgress()` — small logging helpers using the new named constants.

    **Other fixes:**
    - Deleted the unused `StepResult` enum (confirmed dead via repo-wide grep — no references anywhere).
    - Extracted `private static let logitRejectionPenalty: Float32 = -10000.0`, replacing both duplicate literals (the `eosPenalty` precompute and the hard-zone `which(...)` call).
    - Extracted `private static let logPrefix = "[GuidedGen]"` and `private static let stopReasonFragment = " Stop reason: "`, replacing all 6 `"[GuidedGen]"` log sites and all 3 `" Stop reason: "` sites. Verified byte-identical reconstructed log strings (spacing included).
    - Extracted `private static let attosecondsPerMillisecond: Int64 = 1_000_000_000_000_000`, replacing both duplicate elapsed-time divisions (periodic progress log + final stats log).
    - Fixed both `buffer.baseAddress!` force-unwraps (the diagnostic mask-snapshot closure and `buildMaskArray`) with safe `guard let`/`if let`, falling back to "no mask" (nil pointer / nil array) only in the genuinely-degenerate empty-bitmask-buffer case (`bitmaskWords == 0`, which only occurs for a zero-vocab tokenizer) — normal operation is unaffected.
    - Verified `quantizedKvStart` (lowercase v) casing is correct and consistent throughout `run()`'s own signature/call-sites, and confirmed the one call into `MLXLMCommon.maybeQuantizeKVCache(..., quantizedKVStart: quantizedKvStart)` correctly keeps the uppercase-V label since that's genuinely the external function's own parameter name (`Libraries/MLXLMCommon/KVCache.swift`). No changes needed there — verified current state already matched the expected end state.

    **Discovery (not acted on, out of scope):** In the fast-forward branch, the originally-sampled token that triggered the FF batch does not appear to get its own forward pass fed through the model — only the FF tokens themselves do (compare to the non-FF `else` branch, which explicitly feeds the sampled token). This looks suspicious for KV-cache correctness but is pre-existing behavior, unrelated to this cleanup task, and preserved as-is per the "no behavior change" directive. Flagging for a follow-up investigation task if warranted.

    **Verification:**
    - `swift build` (whole package): clean, no errors.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - `xcrun xctest MLXGuidedGenerationTests.xctest` (unfiltered, via `timeout`): 62/62 tests passed, 13 suites.
    - `xcrun xctest MLXFoundationModelsTests.xctest` (unfiltered, via `timeout`): 148/148 tests passed, 32 suites (this bundle exercises `GuidedGenerationLoop.run` via `MLXLanguageModel.swift`).
    - `MLXLMTests` doesn't reference `GuidedGenerationLoop` (confirmed via grep), so not re-run.
    - `mcp__sah__diagnostics check working`: 0 errors, 0 warnings.
    - Adversarial double-check agent: PASS (verified tokenCount accounting, FF short-circuit semantics, byte-identical log strings, setup ordering, force-unwrap safety, and quantizedKvStart casing).

    Task left in `doing` per process — ready for `/review`.
  timestamp: 2026-07-10T02:38:37.962244+00:00
- actor: claude-code
  id: 01kx51g2cq4gx2stty16sgxmc1
  text: |-
    Applied all 5 review findings to GuidedGenerationLoop.swift:

    1. Casing: renamed tokenID→tokenId, stopTokenIDs→stopTokenIds, whitespaceTokenIDs→whitespaceTokenIds (run()'s own parameter), buildStopTokenIDs→buildStopTokenIds (function + all call sites) throughout the file. Left quantizedKVStart/quantizedKvStart dual casing untouched per the task's explicit instruction (external maybeQuantizeKVCache parameter name). Also left WhitespaceRunTracker's own `whitespaceTokenIDs:`/`record(tokenID:)` parameter labels untouched since those are that type's real external parameter names (WhitespaceRunTracker.swift, out of scope) -- only the local variable/argument value passed to them was renamed. Updated the two external call sites this renaming breaks: Tests/MLXGuidedGenerationTests/StopTokenSourceTests.swift (4x buildStopTokenIDs→buildStopTokenIds) and Libraries/MLXFoundationModels/MLXLanguageModel.swift:2172 (whitespaceTokenIDs: label → whitespaceTokenIds:, argument value unchanged).
    2. Added `private static let millisecondsPerSecond: Int64 = 1000` and used it in the new durationToMilliseconds helper (see #3).
    3. Extracted `private static func durationToMilliseconds(_ duration: Duration) -> Int64`, called from both run()'s final-stats log and logProgress().
    4. Extracted `private static func updateMaskAfterForwardPass(state:kvBits:kvGroupSize:quantizedKvStart:constraint:vocabSize:logitDim:)` wrapping the advanceMaskOnCPUWhileGPURuns call; both processFastForwardTokens and advanceSingleSampledToken now call it instead of duplicating the tuple-destructuring call.
    5. Extracted `private static func cacheOrNil(_ cache: [KVCache]) -> [KVCache]?`; both forward-pass call sites (in processFastForwardTokens and advanceSingleSampledToken) use it instead of the duplicated ternary.

    Verification: `swift build` clean (only a pre-existing unrelated deprecation warning in MLXLanguageModel.swift:762). `xcodebuild build-for-testing` succeeded. `xcrun xctest` on both bundles (unfiltered): MLXGuidedGenerationTests 62/62 passed, MLXFoundationModelsTests 148/148 passed -- same counts as before. Task left in review per orchestrator instructions; not moving columns or committing.
  timestamp: 2026-07-10T03:35:17.143200+00:00
- actor: claude-code
  id: 01kx53nwjyt01at1nxz88pv642
  text: |-
    Verified and fixed the IntegrationTesting propagation gap for the buildStopTokenIds/whitespaceTokenIds rename.

    **What I found:** `IntegrationTesting/` (separate Xcode project, guarded by `#if FoundationModelsIntegration`, not part of `swift build`/`mlx-swift-lm-Package`) had 5 files still referencing the old names, all under `IntegrationTestingTests/MLXFoundationModelsIntegration/GuidedGeneration/`:
    - `StopTokenRegressionIntegrationTests.swift` — 6 calls to `GuidedGenerationLoop.buildStopTokenIDs(...)`, plus 3 stale references in a doc comment, a `@Test` name string, and a test func name (`buildStopTokenIDsIsDeterministic`) — all renamed to `buildStopTokenIds`.
    - `GenerableRoundTripTests.swift` — 1 call-site argument label `whitespaceTokenIDs:` → `whitespaceTokenIds:` (on the `GuidedGenerationLoop.run(...)` call).
    - `MultiModelGuidedGenerationTests.swift` — 2 call-site argument labels fixed (same pattern, two separate test functions).
    - `HardReserveStressTests.swift` — 1 call-site argument label fixed.
    - `GuidedGenerationTests.swift` — 2 call-site argument labels fixed.

    **Important nuance:** the local variable `whitespaceTokenIDs` produced by `let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(...)` was deliberately left unchanged — `WhitespaceTokenBias.compute`'s tuple label is still `tokenIDs` (not renamed), and the main package (`Libraries/MLXFoundationModels/MLXLanguageModel.swift`) follows the identical convention: local var / struct field stays `whitespaceTokenIDs`, only the `GuidedGenerationLoop.run(...)` argument label changes to `whitespaceTokenIds:`. I matched that exactly rather than renaming the local variables too.

    Also grepped the entire `IntegrationTesting/` tree (excluding `.build`) for bare `tokenID` and `stopTokenIDs` per the task — zero hits beyond what's already covered above, so no additional missed call sites.

    **Compile verification:** Built via `xcodebuild build-for-testing -project IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination "platform=macOS" -skipPackagePluginValidation` (needed `-skipPackagePluginValidation` to get past an unrelated CudaBuild plugin validation failure in this sandbox). All 5 edited files compiled cleanly (`builtin-SwiftPerFileCompile` succeeded for each, confirmed via build log). The overall test-build command still fails, but only due to a **pre-existing, unrelated** issue in `IntegrationTestingTests/MLXFoundationModelsIntegration/Support/FMTestHelpers.swift` (a file I did not touch) — errors like `type '_ErrorCodeProtocol' has no member 'updateUsage'` and `no macro named 'require'`, consistent with a FoundationModels/Testing SDK version mismatch in this sandbox (macOS 27 SDK), not caused by this rename. So: full project test-build could not be completed end-to-end in this sandbox, but the specific renamed files were compile-verified directly (not just source-reviewed).

    **Main package re-verification (per task step 5):** `swift build` clean (Build complete). `swift build --build-tests` clean. `xcrun xctest .build/out/Products/Debug/MLXGuidedGenerationTests.xctest` → 62/62 passed. `xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` → 148/148 passed. No regressions.

    Left the task in `review` as instructed — did not move columns, did not commit.
  timestamp: 2026-07-10T04:13:24.958899+00:00
- actor: claude-code
  id: 01kx5731dmrk5dm45jz7kdsvb5
  text: |-
    Completed round-3 triage's remaining "Fix directly" items.

    **1. Completed the whitespaceTokenIDs→whitespaceTokenIds rename in `Libraries/MLXFoundationModels/MLXLanguageModel.swift`:**
    - `TokenizerBias.whitespaceTokenIDs` struct field → `whitespaceTokenIds`.
    - `ConstraintSetup.whitespaceTokenIDs` struct field → `whitespaceTokenIds`.
    - `makeTokenizerBias`'s local `let (whitespace, whitespaceTokenIDs) = WhitespaceTokenBias.compute(...)` → renamed local to `whitespaceTokenIds`, updated the `TokenizerBias(...)` construction site to match.
    - `prepareConstraintSetup`'s `ConstraintSetup(..., whitespaceTokenIDs: bias.whitespaceTokenIDs)` → `whitespaceTokenIds: bias.whitespaceTokenIds`.
    - The `runGuidedGenerationLoop`/`GuidedGenerationLoop.run(...)` call site's mixed-casing `whitespaceTokenIds: setup.whitespaceTokenIDs` → `whitespaceTokenIds: setup.whitespaceTokenIds`.

    **2. Renamed the local-variable occurrences in all 4 IntegrationTesting test files** (`GenerableRoundTripTests.swift`, `GuidedGenerationTests.swift`, `HardReserveStressTests.swift`, `MultiModelGuidedGenerationTests.swift`): every `let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(...)` and its corresponding `whitespaceTokenIds: whitespaceTokenIDs` call-site argument, renamed to `whitespaceTokenIds` throughout (2 occurrences each in GenerableRoundTripTests/HardReserveStressTests, 4 each in GuidedGenerationTests/MultiModelGuidedGenerationTests).

    Left untouched, confirmed via grep + git diff: `Libraries/MLXGuidedGeneration/WhitespaceRunTracker.swift` (its own `whitespaceTokenIDs` property/init param, a separate public struct's API) and the `WhitespaceRunTracker(whitespaceTokenIDs: whitespaceTokenIds)` construction call inside `GuidedGenerationLoop.run()` — that call correctly passes the renamed local into `WhitespaceRunTracker`'s own unrenamed parameter label. Also left `WhitespaceTokenBias.compute`'s own return-tuple label (`tokenIDs`) untouched — only the local variables bound to it were renamed.

    **3. Flattened the 4-level nesting in `applyBiasAndSample`** (`Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`): extracted the hard-zone bias computation (`var hardBias = which(bias .> 0, Float32(0.0), logitRejectionPenalty)` + the `if let eosPenalty { hardBias = hardBias + eosPenalty }` addition) into a new `private static func hardZoneBias(bias: MLXArray, eosPenalty: MLXArray?) -> MLXArray` helper, called from the `if hardReserve > 0` branch as `activeBias = hardZoneBias(bias: bias, eosPenalty: eosPenalty)`. Pure extraction — the computed value and invocation point are unchanged; nesting inside `applyBiasAndSample` drops from 4 levels (needsApply→bias→hardReserve→eosPenalty) to 3 (needsApply→bias→hardReserve).

    **Verification:**
    - `swift build`: clean (only the pre-existing unrelated `LanguageModelCapabilities(capabilities:)` deprecation warning).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package ... -skipPackagePluginValidation`: TEST BUILD SUCCEEDED.
    - `xcrun xctest MLXGuidedGenerationTests.xctest` (unfiltered): 62/62 passed, 13 suites.
    - `xcrun xctest MLXFoundationModelsTests.xctest` (unfiltered): 148/148 passed, 32 suites.
    - `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -skipPackagePluginValidation`: all 4 edited files (`GenerableRoundTripTests.swift`, `GuidedGenerationTests.swift`, `HardReserveStressTests.swift`, `MultiModelGuidedGenerationTests.swift`) compiled cleanly (`builtin-SwiftPerFileCompile` succeeded for each); overall test-build still fails only on the pre-existing, unrelated `FMTestHelpers.swift` SDK-mismatch (macOS 27 SDK Testing-framework errors), as previously documented — not caused by this change.
    - Adversarial double-check agent launched to verify the diff scope and behavior-preservation of the extraction.

    Task left in `review` per orchestrator instructions — did not move columns, did not commit. All 3 round-3 "Fix directly" checkboxes now checked off.
  timestamp: 2026-07-10T05:13:01.620179+00:00
- actor: claude-code
  id: 01kx59nbzksp5te9f78xy222p1
  text: |-
    Round-4 "Fix directly" items completed in GuidedGenerationLoop.swift:

    1. Renamed `advanceMaskOnCPUWhileGPURuns` -> `advanceMaskOnCpuWhileGpuRuns` (function definition, its one call site in `updateMaskAfterForwardPass`, and the doc-comment reference to it in `updateMaskAfterForwardPass`'s own doc).
    2. Extracted the duplicated "compute mask + build its array" pattern into a new private helper `computeMaskAndArray(constraint:vocabSize:logitDim:) throws -> (MaskResult, MLXArray?)`, placed directly after `advanceMaskOnCpuWhileGpuRuns`. Both call sites now use it: the pre-loop initial-mask setup in `run()` (`let (initialMask, initialMaskArray) = try computeMaskAndArray(...)`) and inside `advanceMaskOnCpuWhileGpuRuns`'s CPU/GPU overlap window (`let (mask, maskArray) = try computeMaskAndArray(...)`). The KV-cache quantization call before, and the `asyncEval`/`eval` GPU-overlap calls around, the extracted pattern in `advanceMaskOnCpuWhileGpuRuns` were left untouched as instructed.

    No other files or lines touched -- everything else in the round-4 triage was left as rejected/deferred/dropped per the existing triage notes.

    Verification:
    - `swift build` — clean, exit 0 (only a pre-existing unrelated deprecation warning in MLXLanguageModel.swift).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` — TEST BUILD SUCCEEDED.
    - `xcrun xctest MLXGuidedGenerationTests.xctest` (unfiltered, under `timeout`) — 62/62 tests passed.
    - `xcrun xctest MLXFoundationModelsTests.xctest` (unfiltered, under `timeout`) — 148/148 tests passed.

    Task left in `review` (not moved). No commit made per instructions -- orchestrator handles that.
  timestamp: 2026-07-10T05:57:59.411100+00:00
- actor: claude-code
  id: 01kx5amc8s2wrsye7tb974q8jn
  text: |-
    Round 5 review (2026-07-10 01:08) on commit 953b54f returned 1 finding: `bitmaskToMLXArray` (line 939) should supposedly be `bitmaskToMlxArray`. REJECTED, not fixed — two independent reasons: (1) confirmed pre-existing via `git diff HEAD~1..HEAD`, zero matches, untouched by this commit; (2) the suggested rename is actually wrong, not just out of scope — `MLXArray` is the real imported Swift type from the MLX framework (used identically, capitalized exactly this way, in hundreds of signatures throughout this entire codebase), not a stylistic acronym like the project's own `Xg`/`tokenId` conventions. The function converts a bitmask TO an `MLXArray`, so its name correctly references the real type name; "fixing" it to `MlxArray` would make it inconsistent with every other reference to the actual type. Same class of false positive as the already-adjudicated `quantizedKVStart` cross-boundary case earlier in this task's history.

    This closes out tba2jnb: all 5 review rounds now resolved (rounds 2-4 fixed across commits b240f96/e691f11/953b54f, round 5's sole finding rejected as above). Moving to done.
  timestamp: 2026-07-10T06:14:55.513645+00:00
position_column: done
position_ordinal: '9180'
title: 'GuidedGenerationLoop.run(): reduce pre-existing length/complexity and finish quantizedKVStart casing cleanup'
---
Surfaced by review pressure on `qawe2hb` (KV-cache work), but this is genuinely pre-existing code qawe2hb only lightly touched.\n\n- `GuidedGenerationLoop.run()` spans ~415 lines of actual code, far over the 250-line threshold, interleaving prefill, sampling, fast-forward handling, mask computation, KV-cache quantization, GPU overlap scheduling, and budget-zone policy. Extract reusable subroutines: `prefillAndGetLogits()`, `applyBiasAndSample()`, `processFastForwardTokens()`, plus a state struct bundling `logits`/`mask`/`detokenizer`/`cache`/`modelState`/`tokenCount` to thread through helpers without ballooning parameter lists. This one extraction resolves essentially all of the specific deep-nesting citations too (zone-policy hard-reserve branch, FF-token emission loop, FF-token batch-finalization loop — all 4-5 levels deep, all inside `run()`'s main loop).\n- Unused `StepResult` enum (dead code, never instantiated or matched) — delete it.\n- Hardcoded `-10000.0` logit rejection penalty duplicated in (at least) two places — extract to a named `private static let logitRejectionPenalty: Float32 = -10000.0` constant.\n- Force-unwrap `buffer.baseAddress!` (recurs at 2+ call sites) — `baseAddress` is optional and nil-able for an empty buffer; replace with a `guard let` or document why non-nil is guaranteed here.\n- \\\"[GuidedGen]\\\" logging prefix and \\\" Stop reason: \\\" message-template fragment each repeated verbatim across 3-6 log statements — extract to named constants (`private static let logPrefix = \\\"[GuidedGen]\\\"`, similar for the stop-reason fragment) and reuse.\n- `1_000_000_000_000_000` (attoseconds-per-millisecond conversion factor) repeated in two elapsed-time calculations — extract to a named `private static let attosecondsPerMillisecond` constant.\n- `quantizedKVStart`/`quantizedKvStart` casing: this file's OWN signature/call-sites were already fixed (renamed to `quantizedKvStart`) as part of qawe2hb. The call site into `MLXLMCommon.maybeQuantizeKVCache(...)` correctly uses `quantizedKVStart:` (uppercase) because THAT function's own parameter (`Libraries/MLXLMCommon/KVCache.swift:2005`, unrelated/untouched) is genuinely named that way — repeatedly, incorrectly flagged by review as needing a rename across multiple rounds now; it does NOT need one, changing it would break the build.\n\nNot urgent/blocking — this is pre-existing complexity/cleanliness debt, not a correctness bug. Scope to `GuidedGenerationLoop.swift` only; don't let it balloon into a wider rewrite.\n\n## Rounds 2-3 (2026-07-09 21:49 / 23:19) — all fixed, see commits b240f96, e691f11\n\n- [x] tokenID/stopTokenIDs/whitespaceTokenIDs/buildStopTokenIDs casing sweep across GuidedGenerationLoop.swift + call sites.\n- [x] millisecondsPerSecond, durationToMilliseconds, updateMaskAfterForwardPass, cacheOrNil dedup helpers.\n- [x] Completed whitespaceTokenIds rename across MLXLanguageModel.swift's TokenizerBias/ConstraintSetup + 4 IntegrationTesting test files.\n- [x] Flattened applyBiasAndSample's 4-level hard-zone nesting via new hardZoneBias helper.\n\n## Review Findings (2026-07-10 00:23) — round 4 triage\n\n**Fix directly (in-scope, same file, low-risk, mechanical):**\n- [x] `advanceMaskOnCPUWhileGPURuns` (currently ~line 757) — CPU/GPU acronym casing, pre-existing (predates tba2jnb, from qawe2hb) but squarely in the same acronym-casing sweep this task has already been doing throughout this file. Rename to `advanceMaskOnCpuWhileGpuRuns` + update its call site.\n- [x] Duplicated 2-line \"compute mask + build its array\" pattern: the initial setup (`let initialMask = try constraint.computeMask(); let initialMaskArray = buildMaskArray(for: initialMask, vocabSize:, logitDim:)`, currently ~line 207-208) and the identical pattern inside `advanceMaskOnCPUWhileGPURuns`/`advanceMaskOnCpuWhileGpuRuns` (~line 776-777) — pre-existing, but a small mechanical extraction while already in this file. Extract a shared private helper (e.g. `computeMaskAndArray(constraint:vocabSize:logitDim:) throws -> (MaskResult, MLXArray?)`) and use at both sites.\n\n**Rejected — stale/incorrect citations or already-adjudicated correct code (recurring false positives across multiple rounds now):**\n- Finding at \"line 151\" claiming an uppercase `whitespaceTokenIDs:` argument label — factually wrong, line 151 is already `whitespaceTokenIds: Set<Int> = [],` (correct casing). No such uppercase argument label exists there.\n- Finding at \"line 382\" claiming a `tokenID:` argument label mismatch — factually wrong, line 382 is `switch try model.prepare(input, cache: cache, windowSize: 512) {`, nothing resembling this citation.\n- Finding at \"line 189\" claiming duplicated mask computation at 189-190/310-311 — wrong lines (189-190 is the stop-token EOS-penalty loop; 310-311 is unrelated `logitDim:`/`emit:` parameters); the REAL duplicate (see \"fix directly\" above) is at different, correct locations.\n- `WhitespaceRunTracker(whitespaceTokenIDs: whitespaceTokenIds)` construction call (line 171) and `state.whitespaceTracker.record(tokenID: tokenId)` (line ~475) — both correctly reference `WhitespaceRunTracker`'s own separate, pre-existing, out-of-scope public API; already adjudicated multiple rounds ago, not touching.\n- `quantizedKVStart:` argument label at the `maybeQuantizeKVCache(...)` call site — the same false positive rejected at least 3 times across this task's review history; `MLXLMCommon.maybeQuantizeKVCache`'s real parameter is genuinely named `quantizedKVStart` (uppercase), unrelated/untouched cross-library boundary. Renaming would break the build. Not touching.\n\n**Deferred to existing tracking task `12d8p71`** (same identifiers already covered there, confirmed pre-existing/untouched by any tba2jnb commit): additional `suppressedLoadIDs`/`generatedTokenIDs` occurrences in `MLXLanguageModel.swift`.\n\n**Dropped under the test-refactor exception** (confirmed pre-existing, predates this task entirely, same as round-3's identical class of finding): `collectText`/`transcript`/`assertValidJSON`/`sanitize` duplication and \"inline constraint setup ceremony\" findings across `GenerableRoundTripTests.swift`/`MultiModelGuidedGenerationTests.swift`/`GuidedGenerationTests.swift`/`HardReserveStressTests.swift`.