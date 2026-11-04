---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzv3pbye859tdw9xeanw4
  text: 'Change after the PR 581 port (commit 843854ec): the MTP source is `SpeculativeDecodingConfig.Strategy.mtp(MTPDrafterContainer, blockSize)`. Add the deferred form to this enum.'
  timestamp: 2026-10-25T13:46:20.150521+00:00
- actor: claude-code
  id: 01m62gr18nqf4mhnfh128jftw7
  text: |-
    Research + design notes:
    - MLXLMCommon cannot import MLXLLM/MLXVLM, so the Qwen registrations cannot be called from Common. Design: `SpeculativeDecodingConfig.DrafterSource<Drafter>` (generic `.loaded`/`.deferred(bytes:loader)` with `loadedDrafter`, `estimatedBytes`, `load()`); `DraftModelSource` and `MTPDrafterSource` are typealiases of it, and `Strategy.mtp` now carries `MTPDrafterSource` (per the PR 581 comment, the deferred form lives in this enum).
    - Public API: Common `init(mtpDrafterBytes:blockSize:minimumDraftProbability:memoryPolicy:loadMTPDrafter:)` (scripted loader) and `init(mtpDrafterConfiguration:mtpDrafterBytes:from:using:factory:registerDrafterTypes:...)` (loads through `MTPDrafterModelFactory`); MLXLLM `init(qwen35TextMTPDrafter:...)` (registers `Qwen35TextMTPRegistration`, factory `.shared`) and MLXVLM `init(qwen35VLMMTPDrafter:...)` (registers `Qwen35VLMMTPRegistration`, factory `.visionLanguage`). A byte estimate is required, as for `draftModelBytes:` — nothing is downloaded before admission.
    - ChatSession: new `loadedMTPDrafter: SerialAccessContainer<MTPDrafterContainer?>` slot, seeded from an eager drafter, shared with forks; the deferred load runs inside `slot.update`, so concurrent forks still load once. `usesMTP` now comes from the strategy, not drafter presence; the pre-load memory check (`draftAdmissionBeforeLoading`, now fed by `estimatedDraftModelBytes` for MTP too) gates the load; fail throws before the load, fallback yields `mtpFallbackReason == .memoryBudgetExceeded` (helper takes an optional drafter). Eager MTP configs have no estimate, so their path is unchanged except one uncontended slot lock per turn.
    - Test filtering: `xcrun xctest -XCTest MLXLMTests.<SuiteName> .build/out/Products/Debug/MLXLMTests.xctest` runs only that Swift Testing suite (the XCTest side runs 0 tests) — a single suite does not need the full ~3.5 min run. `swift test --filter` still fails on the metallib.
    - Existing SpeculativeDecodingConfigTests matched `.mtp(let drafter, ...)`; updated to `.mtp(.loaded(let drafter), ...)`.
  timestamp: 2026-10-29T00:25:19.893488+00:00
- actor: claude-code
  id: 01m62gr7qzd5hyfn4stt2j8yem
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLLM/Qwen35TextMTPRegistration.swift, Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift, Tests/MLXLMTests/ChatSessionDeferredMTPDrafterTests.swift (new, 8.9 KB, 9 test cases), Tests/MLXLMTests/SpeculativeDecodingConfigTests.swift (pattern update). RED: new suite failed to compile on the missing API; GREEN: suite 6 tests / 9 cases pass. `swift build --build-tests`: 0 compiler warnings (only the build-system "missing creator for mutated node" note). xctest all 5 bundles pass: MLXLMTests 920 XCTest + 1972 Swift Testing (pre-existing benchmarkVerification skip), MLXFoundationModelsTests 376, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. Release xcodebuild (load ~5, Activity Monitor at 1 core as usual): qwen38FourBitMTPGeneration default blockSize 2 MTP 92.5/81.8/79.8 tok/s, plain ~59.3 (baseline 91/80/79); qwen38FourBitMTPSampledGeneration passed; qwen38FourBitMTPSystemPromptCache passed with no known issues, edit turn cached 2169 / prefilled 123 tok in 175 ms with MTP on, follow-ups prefill 25–38 tok.
    - next: /review
  timestamp: 2026-10-29T00:25:26.527958+00:00
- actor: claude-code
  id: 01m62gx8k1dpb5g0h0bqeyvqxe
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (c855ea1a) — 2 findings (2 confirmed, 1 refuted): Libraries/MLXLLM/Qwen35TextMTPRegistration.swift:69, Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift:61 (duplication/duplication)
    - next: implement the 2 findings in the 2026-10-28 19:26 Review Findings section, then re-review
  timestamp: 2026-10-29T00:28:11.233351+00:00
- actor: claude-code
  id: 01m62gxn4dwr307h00rsbwyeja
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 files (deferred MTP drafter in ChatSession, Qwen text/VLM shortcuts)
    - test: green — 5 xctest bundles; Release MTP 92.5/81.8/79.8 vs plain 59.3; edit turn 123 tokens
    - commit: c855ea1a
    - review: findings — Libraries/MLXLLM/Qwen35TextMTPRegistration.swift:69, Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift:61
  timestamp: 2026-10-29T00:28:24.077271+00:00
- actor: claude-code
  id: 01m62hczmzypgh5n9wf1708x3j
  text: |-
    Review-finding fix (duplication on the two Qwen MTP shortcuts):
    - Discovery: both shortcuts already forwarded to the shared MLXLMCommon init `init(mtpDrafterConfiguration:mtpDrafterBytes:from:using:factory:registerDrafterTypes:...)`. So "add a shared helper that takes the factory and the register function, and have both call it" was already the state. What repeated was the whole parameter list and its defaults (blockSize, minimumDraftProbability, memoryPolicy) in each shortcut. Another forwarding helper would still restate that list in both shortcuts.
    - Fix: new `public struct MTPDrafterRegistration: Sendable` in MLXLMCommon/MTPDrafterModelFactory.swift bundles `factory` + `registerDrafterTypes`. Its init defaults to `.shared` / `{}`. The Common init now takes `registration: MTPDrafterRegistration = MTPDrafterRegistration()` in place of the separate `factory:` / `registerDrafterTypes:` params. The parameter list now exists once.
    - MLXLLM now has `MTPDrafterRegistration.qwen35Text` (factory `.shared`, `Qwen35TextMTPRegistration.register`). MLXVLM now has `.qwen35VLM` (factory `.visionLanguage`, `Qwen35VLMMTPRegistration.register`). The `init(qwen35TextMTPDrafter:...)` and `init(qwen35VLMMTPDrafter:...)` shortcuts are deleted.
    - Call site: `SpeculativeDecodingConfig(mtpDrafterConfiguration: c, mtpDrafterBytes: n, from: d, using: t, registration: .qwen35Text)`.
    - Public API change: the two shortcut initializers are gone, and the Common init's `factory:` / `registerDrafterTypes:` labels became `registration:`. The only caller outside the library was ChatSessionDeferredMTPDrafterTests, which is updated.
    - TDD: updated the factory test to the new call first. RED was a compile error (`extra argument 'registration'`). GREEN after the change.
  timestamp: 2026-10-29T00:36:46.367875+00:00
- actor: claude-code
  id: 01m62hd3c6nzj9n6hw11dv8ngy
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift, Libraries/MLXLMCommon/ChatSession.swift, Libraries/MLXLLM/Qwen35TextMTPRegistration.swift, Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift, Tests/MLXLMTests/ChatSessionDeferredMTPDrafterTests.swift. `swift build --build-tests`: 0 compiler warnings (only the existing build-system "missing creator for mutated node" note). xctest results: ChatSessionDeferredMTPDrafterTests 6 tests pass; "Speculative decoding configuration" (SpeculativeDecodingConfigTests) 10 tests pass. Full bundles: MLXLMTests 920 XCTest + 1972 Swift Testing pass (the benchmarkVerification skip was already there). MLXFoundationModelsTests 376, MLXGuidedGenerationTests 70, CXGrammarTests 7 and MLXHuggingFaceMacrosTests 5 all pass. Both review findings are flipped to [x]. No decode-path change, so no Release benchmark.
    - next: /review
  timestamp: 2026-10-29T00:36:50.182714+00:00
- actor: claude-code
  id: 01m62herjc0tve5xh5hr9jg3mp
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (89c81045) — 0 findings, 0 confirmed, 0 refuted (7 attempted, 0 failed); the 2 prior items in `## Review Findings (2026-10-28 19:26)` are checked
    - next: task moved to done
  timestamp: 2026-10-29T00:37:44.652998+00:00
- actor: claude-code
  id: 01m62hf3z9a27kxp1sf4g0rqpj
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 5 files (MTPDrafterRegistration replaces the Qwen text/VLM shortcuts)
    - test: green — build 0 warnings; MLXLMTests 920+1972, FoundationModels 376, GuidedGeneration 70, CXGrammar 7, Macros 5
    - commit: 89c81045
    - review: clean — task moved to done
  timestamp: 2026-10-29T00:37:56.329887+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: fff680
title: 'MTP: deferred drafter loading in ChatSession through MTPDrafterModelFactory and the memory policy'
---
## What
After the ChatSession MTP task, a caller must load the MTP drafter itself (`MTPDrafterModelFactory.shared.load`, after `Qwen35TextMTPRegistration.register()`) and pass the model. The separate-draft-model path already supports deferred loading (`DraftModelSource.deferred(bytes:loader)`) and `SpeculativeDecodingMemoryPolicy` (ChatSession.swift ~43-140, ~1336-1345, ~1673-1711). Give MTP the same support.

- [x] Add a deferred MTP source, for example `SpeculativeDecodingConfig(mtpDrafterID: String, blockSize:memoryPolicy:)` or `(mtpConfiguration: ModelConfiguration, ...)`. It loads through `MTPDrafterModelFactory` (Libraries/MLXLMCommon/MTPDrafterModelFactory.swift) with the registrations of MLXLLM and MLXVLM Qwen35 MTP, only when speculation is admitted.
- [x] Report the drafter bytes (`estimatedDraftModelBytes`) to `SpeculativeDecodingMemoryPolicy`, so that `.recommendedWorkingSet` falls back to plain decode, and `.fail` throws `SpeculativeDecodingMemoryError`, as for draft models.
- [x] Keep the loaded drafter in `loadedDraftModel` (or a matching MTP slot), so that it loads one time for each session and its forks.

## Acceptance Criteria
- [x] A session configured with the MTP drafter ID loads the drafter on the first speculative turn and not before, and reuses it on later turns.
- [x] With a memory policy that does not admit the drafter, the session generates with plain decode (fallback action), or throws (fail action).

## Tests
- [x] Tests/MLXLMTests: ChatSession tests with a scripted loader closure that counts loads (one load across two turns and a fork), and memory-policy fallback and fail tests.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session

## Review Findings (2026-10-28 19:26)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M5QWTV4YK1C727JYYVGKDY9Y.jsonl` — no validator matches this file
> - `.kanban/tasks/01M5QWTV4YK1C727JYYVGKDY9Y.md` — no validator matches this file

- [x] `Libraries/MLXLLM/Qwen35TextMTPRegistration.swift:69` `duplication/duplication` — The text `SpeculativeDecodingConfig.init(qwen35TextMTPDrafter:...)` shortcut repeats the same argument-forwarding body as the VLM shortcut in Qwen35VLMMTPRegistration.swift. The two differ only by the factory, the registration function, and the label, so the logic that could drift (the argument list and the parameter defaults) exists twice. Fold the two shortcuts into one shared initializer that takes the factory and the registration function as parameters, as described for the VLM copy. Keep the text and VLM initializers as one-line forwards to it, or move the shared helper into a single file both registrations import.
- [x] `Libraries/MLXVLM/Qwen35VLMMTPRegistration.swift:61` `duplication/duplication` — The VLM `SpeculativeDecodingConfig.init(qwen35VLMMTPDrafter:...)` shortcut is a near-verbatim copy of the text shortcut `init(qwen35TextMTPDrafter:...)` in Qwen35TextMTPRegistration.swift. The two bodies differ only in the label, the factory (`.visionLanguage` vs `.shared`), and the registration function (`Qwen35VLMMTPRegistration.register` vs `Qwen35TextMTPRegistration.register`). Both copies are new in this change, so they will drift out of sync as the MTP initializer signature changes. Extract one internal helper that takes the factory and the registration function as parameters, for example `static func qwen35MTPDrafter(_ configuration:, mtpDrafterBytes:, from:, using:, factory:, register:, blockSize:, minimumDraftProbability:, memoryPolicy:)`. Have both public shortcuts call it with their own factory and register function. Alternatively, keep the public shortcuts as thin one-line forwards to a single shared initializer. Delete the duplicated bodies.
