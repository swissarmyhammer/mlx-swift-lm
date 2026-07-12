---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxc158f0pwnvbq8gz7sz6wgk
  text: |-
    Located target function directly via `grep -n "func commitPromptCache"` per the task's guidance, but neither `commitPromptCache` overload actually matches the description (they use switch/guard, not do-catch with a specific caught exception type + conditional cache mutation). Searched for `catch <SpecificType>` patterns instead and found the real match: `runGuidedGenerationLoop` (private func on `Executor`), whose do-catch:
    - do block: calls `GuidedGenerationLoop.run`, then conditionally mutates `finalCache = result.cache` only `if slot.cache != nil`
    - catch block: catches the specific `GuidedGenerationError.incompleteOutput`

    Refactor: extracted a new private static helper `slotAdoptingResultCache(_ slot: PromptCacheSlot, resultCache: [KVCache]) -> PromptCacheSlot` that encapsulates the "only adopt the result's cache when the slot started with a real cache" guard + slot reconstruction. The main function now tracks `finalSlot` (init'd to `slot`) instead of a bare `finalCache`, replaces the nested-if with one call to the helper in the success branch, and passes `finalSlot` straight to `commitPromptCache` (removing the inline `PromptCacheSlot(...)` construction at the call site).

    Verified: `swift build` clean (only pre-existing unrelated deprecation warning), `swift test --filter MLXFoundationModelsTests` 207/207 passed, local `review working` pass returned 0 findings. Adversarial double-check dispatched to confirm behavior-equivalence across all 4 branches (slot.cache nil/non-nil x success/catch) before handoff.
  timestamp: 2026-07-12T20:44:03.936306+00:00
- actor: claude-code
  id: 01kxc1arsa1gxptam3ejzqvvza
  text: 'Adversarial double-check returned PASS: traced behavior equivalence across all 4 branches (slot.cache nil/non-nil x success/incompleteOutput-catch), confirmed scope is limited to this file/function, confirmed the new helper''s doc matches its code, and independently re-ran `swift build` (clean) and `swift test --filter MLXFoundationModelsTests` (207/207 passed). Leaving task in doing for /review per the implement skill.'
  timestamp: 2026-07-12T20:47:04.490849+00:00
position_column: doing
position_ordinal: '80'
title: Reduce commitPromptCache's do-catch/cache-management complexity
---
## What\nSurfaced by review pressure on a standalone `getOrCreateCached`/`Set()` fix commit, but confirmed genuinely pre-existing via `git diff HEAD~1..HEAD` — zero matches, untouched by that commit.\n\n`Libraries/MLXFoundationModels/MLXLanguageModel.swift` (~line 2855 as of the flagging commit — will have shifted): a function with complex do-catch error handling with specific exception-type catching, conditional cache-slot mutation, and nested if logic for cache management. The do block has conditional cache mutation and the catch block catches a specific exception type.\n\n## Acceptance Criteria\n- [x] Extract the cache management logic (slot initialization, mutation, validation) into a separate helper function, reducing conditional density in the main function.\n- [x] No behavior change — pure refactor.\n- [x] Build clean, full test suite green.\n- [x] A local review pass confirms zero remaining findings of this class.\n\n## Scope\n`Libraries/MLXFoundationModels/MLXLanguageModel.swift` only — likely `commitPromptCache` or a neighboring function; relocate by content (do-catch with cache-slot mutation + specific exception-type catch), not line number. Not urgent/blocking — pre-existing complexity debt, not a correctness bug.\n\n## Resolution\nThe actual match (by content, not the `commitPromptCache` overloads named in the description) was `runGuidedGenerationLoop`, a private func on `MLXLanguageModel.Executor`: its do-catch calls `GuidedGenerationLoop.run`, conditionally mutates a `finalCache` var only `if slot.cache != nil`, and catches the specific `GuidedGenerationError.incompleteOutput`. Extracted a new `private static func slotAdoptingResultCache(_ slot: PromptCacheSlot, resultCache: [KVCache]) -> PromptCacheSlot` helper encapsulating that guard + slot reconstruction; the main function now threads a `finalSlot` (init'd to `slot`) through the do-catch instead of a bare cache variable, calling the helper once in the success branch.