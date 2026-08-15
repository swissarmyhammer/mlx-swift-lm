---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx0y1e2g8qexp4qsapn8qvxk
  text: 'From q3ddgqy''s 2026-07-08 07:59 review round: 3 additional pre-existing, untouched-by-q3ddgqy functions in MLXLanguageModel.swift also have a `config: ReasoningConfig` parameter the reviewer flagged as inconsistent with the file''s `reasoningConfig` convention (established elsewhere, e.g. `runReasoning`): `preparedInput(messages:config:thinkingEnabled:processor:cannotDisableMessage:)`, `reasoningPrimedInside(input:config:tokenizer:)`, and `runToolCallReasoningPhase(input:config:primedInside:...)`. Same deferral rationale as the doc-comment gaps this task already tracks: these functions predate q3ddgqy entirely (confirmed via `git diff` across the whole task range — their signature lines never changed), so renaming is bundled into this task''s broader "pre-existing function cleanup" scope rather than fixed piecemeal under review pressure on an unrelated task.'
  timestamp: 2026-07-08T13:17:54.128784+00:00
- actor: claude-code
  id: 01kx0z4hnhkrtkfkbstfvtj2hc
  text: 'Retracting part of the prior comment: q3ddgqy''s round-8 review flagged the `config`→`reasoningConfig` rename in `preparedInput`/`runToolCallReasoningPhase` (and likely `reasoningPrimedInside`) as a genuine consistency defect *caused by* q3ddgqy''s own partial rename in the surrounding code (once neighboring tuple/parameter names became `reasoningConfig`, these pre-existing ones read as inconsistent by direct comparison, not just in isolation). q3ddgqy is fixing all three directly rather than deferring. This task (`9jtbtkd`) still owns the missing `- Throws:`/`- Returns:`/`- Parameter` doc-block gaps on `loadContainer()`, `preload()`, `isDownloadingInCache()`, `lastLoadErrorInCache()`, `init(configuration:)` — those remain deferred, unaffected by this retraction.'
  timestamp: 2026-07-08T13:37:04.689664+00:00
- actor: claude-code
  id: 01kx126yh6g3vvrasnbm27cv11
  text: 'Update from q3ddgqy round 10 (2026-07-08 09:19 review): `loadContainer()` and `init(configuration:)`''s missing doc sections recurred a second time under review pressure on this file, so q3ddgqy fixed both directly rather than deferring again (cheaper than perpetual re-deferral). Remove these two from this task''s scope. Still open here: `preload()` missing `- Throws:`, `isDownloadingInCache()`/`lastLoadErrorInCache()` missing `- Returns:` — these have not recurred since first deferred, so remain tracked in this task.'
  timestamp: 2026-07-08T14:30:49.126942+00:00
- actor: claude-code
  id: 01kx131csmzb9hwmfm03pwsqwy
  text: |-
    Update from q3ddgqy round 11 (2026-07-08 09:34 review): `preload()`'s missing `- Throws:` recurred a second time, so q3ddgqy is fixing it directly too (same reasoning as loadContainer/init(configuration:) last round) — remove it from this task's scope. `isDownloadingInCache()`/`lastLoadErrorInCache()` missing `- Returns:` have still only appeared once, remain here.

    New finding class this round (not previously seen): several pre-existing property/function doc comments on `MLXLanguageModel` (`weightsLocation`, the `modelID` computed property, `configurationResolver`, the `load` closure-type doc, `remove(modelID:)`) have a multi-sentence opening summary with no blank line separating the one-sentence summary from elaboration — a doc-style rule distinct from the missing-section gaps this task already tracks. Adding this to scope here since it's the same "pre-existing, untouched-by-q3ddgqy documentation debt" bucket.
  timestamp: 2026-07-08T14:45:15.700657+00:00
- actor: claude-code
  id: 01kx13w67ht90e6ny4fem2a7gk
  text: 'Update from q3ddgqy round 12 (2026-07-08 09:49 review): `ModelCache.makeConstraint`''s `switch kind { case .json: ...; case .structuralTag: ... }` was flagged as near-duplicate `GrammarConstraint` initialization. This function is pre-existing/untouched by q3ddgqy (confirmed via `git diff` across the whole task range). The duplication is very likely unavoidable — `GrammarConstraint`''s two initializers take different labeled parameters (`jsonSchema:` vs `structuralTag:`), an external API shape, and the finding itself hedges ("if not feasible due to API constraints, document why"). Adding to this task''s pre-existing-code bucket for a human to confirm/document rather than fixed under q3ddgqy''s review pressure.'
  timestamp: 2026-07-08T14:59:53.713803+00:00
- actor: claude-code
  id: 01kx219b40mffqv7qqc7mfhtna
  text: 'Update from qawe2hb round-3 review (2026-07-08 18:12): the error message string "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output." is duplicated verbatim at two call sites (the reasoning-suppression path and `validateReasoningCapability`''s throw). Confirmed pre-existing/untouched by qawe2hb (only line-shifted by earlier unrelated extractions). Extract to a private static constant and reuse both sites — in scope for this task''s "pre-existing MLXLanguageModel.swift cleanup" bucket.'
  timestamp: 2026-07-08T23:33:53.408801+00:00
- actor: claude-code
  id: 01kx3kswfa92t7nbpd6h44f29d
  text: 'Update from nzbwjgd round 3 review (2026-07-09 08:38): `prewarm()`''s doc comment says "spawns a detached warmup Task" but the code uses unstructured `Task { }` instead of `Task.detached { }`. Confirmed via git diff pre-existing/untouched by nzbwjgd''s commits. In scope for this task''s pre-existing-cleanup bucket.'
  timestamp: 2026-07-09T14:16:44.266257+00:00
- actor: claude-code
  id: 01kx3m9k2gmtfr62sp2eagma96
  text: |-
    Update from nzbwjgd round 4 review (2026-07-09 09:16), all confirmed pre-existing/untouched by nzbwjgd's commits:
    - `ModelCache.makeConstraint`'s json/structuralTag switch duplication — same instance already noted from qawe2hb's review (likely API-mandated, GrammarConstraint's two initializers take different labeled params).
    - `validateReasoningCapability`/`preparedInput` duplicated do-catch blocks calling `.additionalContext()` and catching `ReasoningError.cannotDisableReasoning` — candidate for a shared `validateThinkingConfiguration` helper.
    - Duplicated `storePromptCache` call across `.matches`/`.trimCacheByOne` switch cases.
    - 4-level nesting in a for/switch generation loop — candidate for `handleGenerationChunk()`/`handleGenerationInfo()` extraction.
  timestamp: 2026-07-09T14:25:18.928600+00:00
- actor: claude-code
  id: 01kx3qspgm6ecp400pmj83vq81
  text: |-
    Update from 64jm412 review (2026-07-09 10:15), both confirmed pre-existing/untouched by that commit:
    - Duplicated reasoning error message (recurring, same instance already tracked).
    - Hardcoded `tokenCount: 1` repeated in 3 places (`sendTextDelta`, `sendSegments`' reasoning case, the `.response` case) — all representing "a text delta counts as 1 token." Candidate for a named `textDeltaTokenCount` constant.
  timestamp: 2026-07-09T15:26:32.468451+00:00
- actor: claude-code
  id: 01kx3rd62x863pk1f1svnkxy2m
  text: 'Update from 64jm412 round 2 review (2026-07-09 10:26): `runReasoning()` (58 lines, for-await→switch→if-let nesting) — confirmed pre-existing/untouched. Candidate helper: `processGeneratedToken(_:emitter:detokenizer:reasoningTokenCount:)`.'
  timestamp: 2026-07-09T15:37:11.005865+00:00
- actor: claude-code
  id: 01kx3s1c48bedtctcrggw1x7vw
  text: 'Refined recommendation from 64jm412''s round 3 review: rather than just a message constant, extract a shared `private static func throwCannotDisableReasoningError(_ message: String) throws` helper covering the full catch-block pattern (catch ReasoningError.cannotDisableReasoning → throw LanguageModelError.unsupportedCapability(...)), used at both call sites in validateReasoningCapability and preparedInput (and the third site flagged in an earlier round).'
  timestamp: 2026-07-09T15:48:12.552653+00:00
- actor: claude-code
  id: 01kx3ses3cqq7jq8hhes896z51
  text: |-
    Update from 64jm412 round 4 review (2026-07-09 10:48), both confirmed pre-existing/untouched:
    - `Stream.gpu.synchronize()` duplicated verbatim in `respond()`'s `catch is CancellationError` and generic `catch` blocks — candidate for `defer { Stream.gpu.synchronize() }` wrapping both, or a shared helper.
    - Reasoning error-mapping duplication (recurring, same instance) — refined again: `static func throwReasoningCapabilityError(thinkingEnabled:debugDescription:) throws`.
  timestamp: 2026-07-09T15:55:31.820993+00:00
- actor: claude-code
  id: 01kx3tdxw3g8j2a6k6npv7tf0k
  text: |-
    Update from 64jm412 round 5 review (2026-07-09 10:55), all confirmed pre-existing/untouched by that commit (doc-only change):

    - 6 findings proposing `xgTokenizer(s)`/`hasCachedXgTokenizer`/`makeXgTokenizer` → `xGTokenizer(s)`/etc. (uppercase-both-letters acronym). **REJECTED** — this is the exact opposite of the deliberate `XG`→`Xg` rename established early in q3ddgqy's history (100+ existing uses of the `Xg` pattern throughout this file). Do not apply; same class as the already-rejected acronym-casing contradictions.
    - `respond()` cited as ~150 lines / 4+ nesting needing extraction into "channel metadata setup / container.perform closure / error-handling" helpers — this decomposition already happened (`prepareRespondSetup`/`dispatchGeneration` already extracted in earlier rounds); `respond()` itself is currently ~210 lines total (1013-1223), consistent with the previously-confirmed ~224-line/~145-code-line figure, under the 250-line threshold. Same pattern as prior stale/re-litigated "respond too long" findings in q3ddgqy's history.
    - `runUnconstrained`'s for-await-switch-case 4-level nesting, matching `runReasoning`'s already-tracked issue — same `processGeneratedToken`-style extraction would likely resolve both together.
    - Reasoning error-message duplication — recurring, same instance already tracked multiple times.
  timestamp: 2026-07-09T16:12:32.515287+00:00
- actor: claude-code
  id: 01kx4gq0nnwy6rr64q2gt6ptkk
  text: |-
    Finished this task, picking up from a prior implementer whose work was already 90%+ done when I started (verified via `git diff --stat`: 351 insertions/68 deletions already present against `Libraries/MLXFoundationModels/MLXLanguageModel.swift` before I touched anything). I read the whole 3182-line file in ~5 large chunks (not per-function windows) to build a map of what was already fixed vs. still open, then made only the remaining, verified-necessary edits.

    **Verified already done by the prior agent (no re-edit needed):**
    - Doc-comment sweep: ran a script that parses every `func`/`init` in the file, checks whether it has params/`throws`, and whether its preceding `///` block has a `- Parameter(s):`/`- Throws:` section. Zero gaps found except two (fixed below).
    - `isDownloadingInCache()`/`lastLoadErrorInCache()` `- Returns:` — already present.
    - Blank-line-after-summary formatting on the 5 named sites (`weightsLocation`, `modelID` computed property, `configurationResolver`, the `load`/`ContainerLoader` typealias doc, `remove(modelID:)`) — already has blank lines.
    - `ModelCache.makeConstraint`'s json/structuralTag switch — already has an explanatory comment documenting why the duplication is API-mandated (`GrammarConstraint`'s two initializers take different labeled params).
    - Shared reasoning-error-throw helper — already extracted as `additionalContextOrThrowCapabilityError(promptStrategy:thinkingEnabled:debugDescription:)`, used at all real throw call sites (`validateReasoningCapability`, both `preparedInput` call sites in `prepareRespondSetup`). Only one `throw LanguageModelError.unsupportedCapability(.reasoning)` site exists in the whole file now, inside that helper.
    - `prewarm()` doc vs. code — already uses `Task.detached { }`, matching its doc comment.
    - `storePromptCache`/`commitPromptCache` dedup across `.matches`/`.trimCacheByOne` — already a single `shouldStore`-gated call.
    - `runReasoning()` and `runUnconstrained()` nesting — already reduced via `processReasoningToken(...)` and `handleGenerationEvent(...)` extractions respectively.
    - `Stream.gpu.synchronize()` dedup in `respond()` — already a single call in one shared `catch` block (not duplicated across two catch blocks).
    - `textDeltaTokenCount` constant — already defined and used at the 3 sites named in the task (`sendTextDelta`, `sendSegments`'s `.reasoning`/`.response` cases).

    **What I actually changed (all verified with `swift build` after batching, plus a final full-suite test run):**
    1. Deduped the last remaining literal copy of "This model always reasons; .reasoning must be declared at MLXLanguageModel init to receive its output." in `prepareRespondSetup` (the suppressed-input call site) to use `Self.alwaysReasoningDebugDescription` instead of repeating the string.
    2. Added missing `- Throws:` blocks to two functions the doc-gap scan actually found: `executeToolCallingPhase2` and `runGuidedGenerationLoop` (both throw and forward/catch `GuidedGenerationLoop.run`'s errors, catching only `GuidedGenerationError.incompleteOutput`).
    3. Replaced a 4th, previously-missed hardcoded `tokenCount: 1` in `handleRealTool`'s `.appendArguments` call with the existing `textDeltaTokenCount` constant (whose doc comment already covers `.appendArguments`, not just `.appendText`).
    4. Ran the project's local `review` tool (`review file`) against the finished file and got two confirmed findings: `sendSegments`'s `.response` case duplicated `sendTextDelta`'s body verbatim. Fixed by calling `sendTextDelta` from that case instead.
    5. Re-ran review; it then flagged the sibling `.reasoning` case as the same pattern once `.response` was fixed. Added a new `sendReasoningDelta(_:entryID:channel:)` helper mirroring `sendTextDelta` and used it in the `.reasoning` case for symmetry.

    **Verification:**
    - `swift build` after each batch: clean, only the same pre-existing unrelated deprecation warning (`LanguageModelCapabilities(capabilities:)` at line 762).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` — **TEST BUILD SUCCEEDED**.
    - `xcrun xctest <DerivedData>/.../MLXFoundationModelsTests.xctest` (unfiltered, wrapped in `timeout`) — 147/147 tests passed across 31 suites, 0 failures, exit code 0 (test count grew from the original 83 baseline since other tasks landed tests in the interim; this is expected).
    - Adversarial double-check agent (via `really-done`'s gate) reviewed the 5 changes above against the diff and source independently: **PASS**, no discrepancies found (one cosmetic wording nit in my own narrative, no code issue).
    - Post-edit doc-gap script re-run: 0 missing `- Parameters:`/`- Throws:` blocks across the whole file.

    **Deliberately left out, with rationale:**
    - A broader "insert a blank line after every multi-sentence doc-comment summary in the file" pass. The accumulated review comment (q3ddgqy round 11) named 5 specific sites for this rule, all of which were already fixed. A wider sweep would touch ~15+ additional struct/property/type doc comments (e.g. `ConstraintKind`, `TokenizerBias`, `ModelCache`, `LoadTask`, etc.) never called out by any reviewer round and outside the task's stated acceptance criteria — applying it unprompted risked unbounded scope creep on a task that has already absorbed 6+ prior tasks' review debt.
    - The local `review` tool's newest round (after my last fix) surfaced 6 additional findings requesting deeper "cognitive complexity" extraction in `preparedInputMappingImageFailures`, `respond()`, `commitPromptCache`, `runReasoning`, `runToolCallReasoningPhase`, and `runGuidedGeneration` (e.g. extract `hasImageContent(_:)`, a `CacheAction` enum, a `ReasoningState` struct, `streamTextDeltasToChannel(_:entryID:)`, etc.). These are new, generic refactor opinions from re-running the reviewer, not items from the task's own accumulated comment history — and `respond()`'s length/nesting specifically was already investigated and confirmed stale/re-litigated in the 64jm412 round 5 comment above (~210 lines, under the 250-line threshold, already decomposed via `prepareRespondSetup`/`dispatchGeneration`). I did not chase these; doing so would restart an open-ended refactor loop this task's history explicitly flags as already resolved/rejected once.

    No behavior change anywhere in this pass — every edit is either a doc-comment addition/fix, a literal-to-constant substitution, or an inline-to-existing-helper-call substitution, all verified byte-for-byte equivalent in emitted behavior by the adversarial reviewer. Leaving the task in `doing` for `/review`, per process (not moving columns myself). No commit made — that's the orchestrator's job.
  timestamp: 2026-07-09T22:41:58.965399+00:00
- actor: claude-code
  id: 01kx4kcj36hq7g427gffv7cm3g
  text: |-
    Fixed the 2026-07-09 17:48 review finding: merged `sendTextDelta`/`sendReasoningDelta` into a single `private static func sendDelta(_ text: String, entryID: String, channel: LanguageModelExecutorGenerationChannel, isReasoning: Bool) async` in `Libraries/MLXFoundationModels/MLXLanguageModel.swift`. The merged function branches on `isReasoning` to send `.reasoning(...)` vs `.response(...)` with `.appendText(text, tokenCount: textDeltaTokenCount)`; full `///` doc comment with `- Parameters:` covering all four params. Updated all 6 call sites (the guided-generation forwarder loop, `handleGenerationEvent`'s `.chunk` case, both `sendSegments` cases (`.reasoning`/`.response`), `emitToolCallingEvent`'s malformed-output fallback, and `handleFinalAnswerTool`) to call `sendDelta(..., isReasoning:)` instead of the two separate functions. Deleted both old functions -- confirmed no remaining references via grep.

    Verified: `swift build` clean (same single pre-existing unrelated deprecation warning at line 762, no new warnings/errors). `xcodebuild build-for-testing` succeeded; unfiltered `xcrun xctest` on `MLXFoundationModelsTests.xctest` (wrapped in `timeout`, not piped through `tail`) reports "Test run with 147 tests in 31 suites passed" -- same 147/147 count as before, 0 failures.

    Left the task in `review` (no column move) and checked off the finding's checkbox. No commit made -- orchestrator's job.
  timestamp: 2026-07-09T23:28:42.086064+00:00
position_column: done
position_ordinal: 8d80
title: Add full Parameters/Throws doc blocks across MLXLanguageModel.swift's pre-existing functions
---
## What\n`Libraries/MLXFoundationModels/MLXLanguageModel.swift` has had a brief one-line `///` summary documentation style since its original commit (`134db41`) — functions carry a short description but omit `- Parameters:`/`- Throws:` blocks even when they take multiple parameters or throw. This predates and is unrelated to any of ^q3ddgqy's tool-calling/KV-cache work; it surfaced as review findings on that task's diff but the vast majority of flagged functions were never touched by that diff, so it's split out here as its own task rather than blocking tool-calling work on a full-file documentation pass.\n\nRepresentative sample of functions needing full `- Parameters:`/`- Throws:` blocks added (verified present as of commit `ebd7171` on `origin/mlx-foundationmodels`; there are ~30+ in total — sweep the whole file, this list is a non-exhaustive starting point, not a checklist ceiling):\n- `ModelCache.load(modelID:suppressDownloadingState:loader:)`\n- `ModelCache.isDownloading(modelID:)`, `.lastError(modelID:)`, `.remove(modelID:)`\n- `ModelCache.makeXgTokenizer(modelID:tokenizer:)`, `.hasCachedXgTokenizer(modelID:)`, `.makeTokenizerBias(modelID:tokenizer:)`, `.makeConstraint(modelID:kind:source:tokenizer:hostTokenizer:fastForward:)`\n- `MLXLanguageModel.loadContainer()`, `.loadContainer(suppressDownloadingState:)`, `.preload()`, static `makeXgTokenizer`/`makeTokenizerBias`/`makeConstraint`/`hasCachedXgTokenizer`/`isDownloadingInCache`/`lastLoadErrorInCache`\n- `Executor.warmUp()`, `.clampedTemperature(_:)`, `.samplingMode(from:)`, `.makeParameters(...)`, `.mapGrammarError(_:)`, `init(configuration:)`, `.respond(to:model:streamingInto:)`\n- `Executor.runUnconstrained(...)`, `.runTextGeneration(...)`, `.runReasoning(...)`, `.send(...)`, `.preparedInput(...)`, `.reasoningPrimedInside(...)`, `.runToolCallReasoningPhase(...)`, `.emitToolCallingEvent(...)`, `.unwrapToolCallMarkers(_:)`\n\nRule being applied (per this project's Swift documentation validator): every documented function with parameters gets a `- Parameters:` block (or inline `- Parameter x:` for single-parameter functions) enumerating each; every documented function that `throws` gets a `- Throws:` line.\n\n## Acceptance Criteria\n- [ ] Every `public`, `internal`, and `private` function/initializer in `MLXLanguageModel.swift` that already carries a `///` doc comment and has one or more parameters has a `- Parameters:` (or inline `- Parameter:`) block covering each parameter.\n- [ ] Every already-documented function/initializer that `throws` has a `- Throws:` section.\n- [ ] No behavior change — this is documentation-only.\n- [ ] A local review pass (`review sha` scoped to the commit) confirms zero remaining \"missing Parameters/Throws\" findings for this file.\n\n## Tests\n- [ ] No new automated test is needed for doc-comment-only changes; run the existing suite to confirm no accidental behavior change: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` then `xcrun xctest <DerivedData>/Build/Products/Debug/MLXFoundationModelsTests.xctest` — expect the existing 83/83 to still pass, zero new warnings.\n\n## Workflow\n- Mechanical documentation task — no TDD needed since there's no new behavior to test-first. Sweep the file function-by-function, verify the build/tests stay green throughout.\n\n## Review Findings (2026-07-09 17:48)\n\n- [x] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1849` — sendReasoningDelta is a near-verbatim duplicate of sendTextDelta, differing only in whether it sends .response or .reasoning. Two functions that differ only by a single value should be merged into one function with a parameter. Extract both into a single sendDelta(_:entryID:channel:isReasoning:) function that takes a Bool parameter to control which channel variant is sent. Update all call sites in sendSegments (lines 2013, 2015) and elsewhere to use the parameterized version instead of two separate functions.\n