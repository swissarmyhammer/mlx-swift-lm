---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx3c9b1zvhavy0jg1wxmdsst
  text: |-
    Investigated all 3 candidate cases. Implemented 1 of 3 (`.unsupportedTranscriptContent`); documented the other 2 as genuinely not applicable.

    **Important correction to the task's premise:** this machine DOES now have a real os27 SDK/Xcode available (`xcodebuild -version` reports Xcode 27.0, `sw_vers` reports macOS 27.0, and `xcode-select -p` points at the installed Xcode). I used it to get ground truth instead of guessing:
    - `swiftc -typecheck` with a deliberately-incomplete `switch` over `LanguageModelError` prints the compiler's own "add missing case" diagnostics, giving the full, real case list: `.contextSizeExceeded`, `.rateLimited`, `.guardrailViolation`, `.refusal`, `.unsupportedCapability`, `.unsupportedTranscriptContent`, `.unsupportedGenerationGuide`, `.unsupportedLanguageOrLocale`, `.timeout` (+ `@unknown default`). No `assetsUnavailable`/`decodingFailure`/`concurrentRequests` exist — confirms the task's caution against guessing those was correct.
    - `swift-demangle` over `FoundationModels.tbd`'s exported symbols gave the real associated-value shapes. Confirmed via compile probes (not just reading symbol names) that only the *smaller*-arity public inits are actually callable (the `underlyingErrors:`-inclusive inits are exported at the ABI level but not part of the public Swift API surface — attempting to call them gives "extra argument" errors):
      - `LanguageModelError.RateLimited(resetDate: Date?, debugDescription: String)`
      - `LanguageModelError.Timeout(debugDescription: String)`
      - `LanguageModelError.UnsupportedTranscriptContent(unsupportedContent: [Transcript.Entry], debugDescription: String)` — note `unsupportedContent` is `[Transcript.Entry]`, **not** segments or `[UserInput.Image]`.

    **1. `.rateLimited` — NOT implemented, genuinely not applicable.** Investigated concurrency handling: `PromptCache` (actor) and container loading both *queue/wait* for concurrent requests against the same model rather than *reject* them (see `PromptCache.swift` doc comments on slot pooling/actor serialization, and `ModelCache`'s per-model load de-dup). There is no code path anywhere in `MLXLanguageModel.swift`/`PromptCache.swift` that rejects a request because another is in flight. A fully local, in-process backend has no rate-limiter analog (no quota, no external service). Left unmapped per the task's own allowance.

    **2. `.timeout` — NOT implemented, genuinely not applicable.** Grepped for any timeout/deadline concept: `GenerationOptions` (confirmed via the same SDK-probing technique) exposes only `maximumResponseTokens`, `sampling`/`samplingMode`, `temperature`, `toolCallingMode` — no deadline field. The only cancellation mechanism anywhere in the generation loop is cooperative `Task.checkCancellation()`/`catch is CancellationError` (`MLXLanguageModel.swift` respond(), `GuidedGenerationLoop.swift`), which fires for *any* reason the enclosing `Task` is cancelled (caller-initiated cancel, test harness time limit, etc.) — not specifically because a time budget was exceeded. Remapping raw `CancellationError` to `.timeout` would misrepresent arbitrary cancellations as timeouts, which is exactly the "fake scenario" the task warns against. No existing deadline parameter to hook a real mechanism to, so per the task's guidance I did not invent a new timeout feature. Left unmapped.

    **3. `.unsupportedTranscriptContent` — IMPLEMENTED.** Traced the image path: `Transcript.ImageAttachment.ciImage` is a non-optional, non-throwing `CIImage` getter (confirmed via the SDK), and `UserInput.Image.asCIImage()` for our `.ciImage(...)` case just returns it directly — so the literal "CIImage decode failure" scenario the task description speculated about does **not** exist in this SDK; that door is already closed by the framework before we ever see the segment.

    The *real* generic-error leak is one level downstream: `Executor.prepareRespondSetup`/`Executor.preparedInput` call `context.processor.prepare(input:)` (the concrete per-model `UserInputProcessor`, e.g. a VLM's own image tensor prep). When a model's own image-specific processing fails (mismatched image count, unsupported resolution, resize/decode failure — `MLXVLM.VLMError` and its per-architecture siblings), that error previously propagated fully opaque/untyped to the FoundationModels caller.

    Deliberately did **not** add `import MLXVLM` to catch `VLMError` by type: `Package.swift` explicitly documents "the MLXFoundationModels product target deliberately does NOT depend on MLXVLM; runtime trampoline discovery is by design and unchanged" — adding that import would violate a documented architecture boundary. Instead, the mapping is by content-shape: added `TranscriptConverter.entriesWithImages(for:)` (new static func) and `MLXLanguageModel.Executor.preparedInputMappingImageFailures(processor:input:messages:transcriptEntries:)` (new static func), which wraps both production call sites of `processor.prepare(input:)`. When `prepare` throws while this round's messages carry image content, it remaps to `LanguageModelError.unsupportedTranscriptContent(unsupportedContent: <the transcript entries that carried images>, debugDescription: ...)`. Cancellation and already-typed `LanguageModelError`s pass through unchanged (never re-wrapped); a failure with NO image content in the round also passes through completely unchanged, so an unrelated tokenizer/config bug is never misrepresented as a vision problem.

    **Tests added** (`Tests/MLXFoundationModelsTests/UnsupportedTranscriptContentMappingTests.swift`, following the fake-model/fake-processor idiom from `EagerFallbackPrepareOrderingTests.swift`/`ContextSizeValidationTests.swift`):
    - `imageProcessingFailureMapsToTypedError` — a `UserInputProcessor` that throws a generic error only when handed image input; asserts the failure surfaces as `LanguageModelError.unsupportedTranscriptContent` with exactly 1 entry in `unsupportedContent`.
    - `textOnlyProcessingFailurePassesThroughUnchanged` — a processor that always throws; with a text-only (no image) transcript, asserts the ORIGINAL generic error type surfaces unchanged, proving the remap is scoped precisely to image-bearing rounds.

    **Verification:**
    - `swift build`: clean, exit 0. 14 pre-existing warnings unrelated to this change (verified via targeted grep — none touch `TranscriptConverter.swift`, the new `MLXLanguageModel.swift` code, or the new test file).
    - Mandated safe invocation: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` → `** TEST BUILD SUCCEEDED **`.
    - Unfiltered `xcrun xctest <bundle>` per bundle, each wrapped in `timeout`:
      - `CXGrammarTests.xctest`: 7/7 passed.
      - `MLXGuidedGenerationTests.xctest`: 62/62 passed.
      - `MLXFoundationModelsTests.xctest`: 139/139 passed (includes the 2 new tests, plus all pre-existing suites incl. `VisionCapabilityGateTests`, `TranscriptConverterTests`, `ContextSizeValidationTests`, `EagerFallbackPrepareOrderingTests`, `GrammarError typed mapping`).
      - `MLXLMTests.xctest`: 245/245 passed.

    Did not touch `.contextSizeExceeded` (task `4kxr1kt`'s territory) or the tool-calling path's `applyChatTemplate`-based tokenization (`runToolCalling` never threads images through `LMInput` at all today — a separate, pre-existing "no vision+tool-calling" limitation noted but out of scope here, not something this task asked to fix).

    Left in `doing` for review per process.
  timestamp: 2026-07-09T12:05:22.111524+00:00
- actor: claude-code
  id: 01kx3cxtdcck3nyv888c8d812k
  text: |-
    Adversarial double-check ran (per really-done) and returned REVISE with 2 findings. Addressed both:

    **Finding 1 (moderate): doc comment said "this round" but `messages` is actually the FULL transcript re-rendered every call** (the `LanguageModelExecutor` protocol has no session identity — every `respond()` receives the complete history again). So an unrelated failure on a later, text-only turn could still get remapped to `.unsupportedTranscriptContent` merely because an *earlier* turn carried an image.

    Investigated whether to narrow the check (e.g. only look at the newest transcript entry) and concluded that would make things *worse*, not better: a real VLM failure mode (`VLMError.singleImageAllowed`-shaped rejection when a second/earlier image is still in scope) would fire precisely when the newest turn has NO new image, so checking only the newest entry would cause us to miss genuine image-caused failures. Since the processor re-processes every image referenced across the whole rendered prompt on every call, "an image is present anywhere in what's being prepared" is the accurate correlate of "this failure could be image-related" — not a bug, but the necessary consequence of mapping by content-shape instead of catching `MLXVLM.VLMError` directly (which the architecture forbids, see task description/Package.swift).

    Fixed the doc comment on `preparedInputMappingImageFailures` to state this precisely (dropped the misleading "this round" framing, added an explicit paragraph naming the tradeoff and reasoning through the `singleImageAllowed` counterexample), and added a third test, `earlierTurnImageStillMapsLaterTextOnlyFailure`, that constructs a 3-entry transcript (image-bearing first turn, assistant response, text-only second turn) and asserts the second turn's unrelated failure still surfaces as `.unsupportedTranscriptContent` — turning the previously-undocumented behavior into an explicit, tested, intentional contract.

    **Finding 2 (minor/process): `Tests/MLXFoundationModelsTests/PromptCacheConcurrencyTests.swift` appeared in `git diff --stat` but wasn't disclosed.** Confirmed this is pre-existing uncommitted state that was already in the working tree before this session started (a `ConcurrencyProbeModel` → shared `PromptCacheProbeModel` test dedupe) — I never opened or edited that file. Disclosing explicitly here so the orchestrator doesn't attribute it to this task when committing; it's unrelated to `unsupportedTranscriptContent` and should be reviewed/committed separately (or was left mid-flight by a prior agent).

    **Re-verified after the fix:** `swift build` clean; mandated safe invocation (`xcodebuild build-for-testing ...` then unfiltered `xcrun xctest <bundle>` per bundle, each under `timeout`) — all 4 bundles green: CXGrammarTests 7/7, MLXGuidedGenerationTests 62/62, MLXFoundationModelsTests 141/141 (now 3 tests in the new suite), MLXLMTests 245/245.

    Task remains implemented as described in the prior comment (`.unsupportedTranscriptContent` mapped; `.rateLimited`/`.timeout` investigated and left unmapped as genuinely not applicable). Left in `doing` for review.
  timestamp: 2026-07-09T12:16:33.196494+00:00
position_column: done
position_ordinal: '8680'
title: Expand MLXLanguageModel's error taxonomy to cover confirmed LanguageModelError cases (rateLimited, timeout, unsupportedTranscriptContent)
---
Libraries/MLXFoundationModels/MLXLanguageModel.swift currently only ever throws `LanguageModelError.unsupportedCapability` and `.unsupportedGenerationGuide` (confirmed via grep — no other case is used anywhere in the file). Real FoundationModels callers pattern-match on the framework's typed `LanguageModelError` cases; failures outside these two categories leak out as opaque/untyped errors.\n\n## Resolution (2026-07-09)\n\nCommit `f402e64`: investigated 3 candidate cases. `.rateLimited`/`.timeout` genuinely don't apply to this fully-local backend (correct, intended outcome). `.unsupportedTranscriptContent` implemented for local image-processing failures. Commits `cdce44a`/`28286ce`: access-control fixes.\n\n## Review Findings, rounds 1-4 (2026-07-09 07:44 through 09:16)\n\nAll FIXED, or DEFERRED to `^9jtbtkd` (confirmed pre-existing/untouched debt), or resolved at the rule level (`reuse` validator updated directly).\n\n## Review Findings (2026-07-09 09:25) — round 5\n\n- [x] Duplicated reasoning error message (recurring) — **DEFERRED**, already tracked in `^9jtbtkd`, confirmed pre-existing/untouched by any of this task's commits.