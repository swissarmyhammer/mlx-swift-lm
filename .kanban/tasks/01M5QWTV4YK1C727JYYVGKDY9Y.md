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
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
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