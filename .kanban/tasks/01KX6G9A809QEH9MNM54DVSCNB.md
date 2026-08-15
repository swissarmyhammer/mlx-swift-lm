---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx99nfe9gnj7r6a1cfwwq0ja
  text: |-
    Implemented with a correction to the task's stated premise about @testable import.

    Only `storePromptCache` was marked `private static func`. The other three stayed `internal`:

    - `resolvePromptCache` — called directly (not via a wrapper) from `Tests/MLXFoundationModelsTests/PromptCachePrewarmScenarioTests.swift` (`MLXLanguageModel.resolvePromptCache(...)`). Tried marking it `private`; `xcodebuild build-for-testing` failed with `'resolvePromptCache' is inaccessible due to 'private' protection level` at that call site. **Empirically, `@testable import` does NOT expose `private`/`fileprivate` declarations from a different file** — contrary to what the task description assumed. Reverted to `internal`.
    - `removePromptCache` — called directly from multiple files in the separate `IntegrationTestingTests` target (`PromptCacheEquivalenceTests.swift`, `PromptCachePrewarmTests.swift`, `PromptCacheMultimodalBoundaryTests.swift`, `PromptCacheGuidedRoundTripTests.swift`), all via `@testable import MLXFoundationModels`. `PromptCacheEquivalenceTests.swift` even has a comment explicitly noting it's "an internal (not public) entry point -- reachable here via `@testable import`". Given the confirmed same restriction (private breaks `@testable` cross-file access), left as `internal` without even needing to test-and-revert.
    - `isDownloadingInCache` — called from `Libraries/MLXFoundationModels/MLXLanguageModel+Availability.swift` (a different file, production code, same module: `Self.isDownloadingInCache(modelID:)` inside an `extension MLXLanguageModel`). `private` is file-scoped in Swift so this would break same-module production code. Left as `internal`.
    - `storePromptCache` — grep confirmed no call sites anywhere outside `MLXLanguageModel.swift` itself (no test, no other production file). Marked `private static func`. Verified clean via full build + build-for-testing + all 4 xctest bundles.

    Verification:
    - `swift build`: clean (only a pre-existing, unrelated `LanguageModelCapabilities(capabilities:)` deprecation warning, not introduced by this change).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: **TEST BUILD SUCCEEDED**.
    - `xcrun xctest` per bundle, unfiltered, under `timeout`:
      - MLXFoundationModelsTests: 203/203 passed
      - CXGrammarTests: 7/7 passed
      - MLXGuidedGenerationTests: 63/63 passed
      - MLXLMTests: 245/245 passed

    No behavior change — pure access-level tightening on one of the four functions; the other three left as-is with documented reasons. Task left in `doing` for review, not committed.
  timestamp: 2026-07-11T19:15:00.681361+00:00
- actor: claude-code
  id: 01kx9a647d14xm7ksa2hsdzd9d
  text: |-
    Committed at `53d0714da73a28056718f56e820b41df1df5bf11` (1 file, 1 insertion/1 deletion — `storePromptCache` → `private static func`).

    Review (`review sha HEAD~1..HEAD`) returned 4 findings, ALL REJECTED as stale/out-of-scope:
    - Confirmed via `git diff HEAD~1..HEAD --stat`: this commit touches exactly 1 line (`storePromptCache`'s access modifier). None of the 4 findings' cited lines (841, 1175, 1794, 2008) are anywhere near that line — all are pre-existing code this commit never modified.
    - Finding at line 1175 (\"mark `removePromptCache` private for consistency, it has no external call sites\") is additionally factually WRONG, not just out-of-scope: `grep -rl removePromptCache` confirms real external callers in `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/{PromptCacheEquivalenceTests,PromptCachePrewarmTests,PromptCacheMultimodalBoundaryTests,PromptCacheGuidedRoundTripTests}.swift` — exactly the cross-file `@testable import` callers this same task's own investigation (previous comment) already found and confirmed empirically that `private` would break (contrary to the task description's original assumption that `@testable import` exposes `private` across files — it does not). Marking it private would be a regression, not a cleanup.
    - The two nesting/complexity findings (lines 1794, 2008) are pre-existing debt in unrelated functions (`commitPromptCache`'s trim path, `runReasoning`'s token processing) — the latter is the same recurring nesting finding already rejected as stale on the `svkteh8` commit review, and both are already tracked by existing backlog tasks (`dfk2b64` for commitPromptCache complexity, and general reasoning-loop complexity elsewhere).

    Task's own scope (mark safe passthroughs private) is complete: 1 of 4 functions safely tightened, the other 3 correctly left broader with documented cross-file-caller evidence. Moving to done.
  timestamp: 2026-07-11T19:24:06.253829+00:00
position_column: done
position_ordinal: a980
title: Mark resolvePromptCache/storePromptCache/removePromptCache/isDownloadingInCache as private in MLXLanguageModel.swift
---
## What
Surfaced by review pressure on `cthbfmw`'s PromptCache cutover commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — that commit's only change to `MLXLanguageModel.swift` was deleting `setPromptCacheSlotLimit`; these 4 functions were untouched.

`resolvePromptCache`, `storePromptCache`, `removePromptCache`, `isDownloadingInCache` (static functions in `MLXLanguageModel.swift`) are internal implementation details not meant for cross-module use, but default to `internal` access rather than being explicitly marked `private`.

## Acceptance Criteria
- [ ] Mark each of the 4 functions `private static func ...` — verify first that nothing outside this file/type actually calls them (grep the whole repo) before narrowing access, since `internal` may be relied upon by something not yet checked.
- [ ] No behavior change — pure access-level tightening.
- [ ] Build clean, full test suite green.

## Scope
`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only. Not urgent/blocking — pre-existing cleanliness debt, not a correctness bug.