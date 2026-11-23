---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m84ebpz5ej59ymnyjfjatryh
  text: |-
    Research: commit 3d1ee9eb (^xvwarz1) dropped the explicit `@unchecked Sendable` from MTPDrafterRegistry when it made the class final. Swift requires a subclass to restate an inherited `@unchecked Sendable` conformance. LLMRegistry and VLMRegistry both declare `AbstractModelRegistry, @unchecked Sendable` (left unchanged). AbstractModelRegistry's only mutable state is `private var registry`, which every accessor reads and writes inside `lock.withLock`. MTPDrafterRegistry adds only static members. Repo convention for the exception is `// swiftlint:disable:next no_unchecked_sendable <invariant>` directly above the declaration (for example ExecutorPromptCacheSlot and MTPDrafterFallbackLog).

    Before the fix: after `touch` on the file, `swift build --build-tests` warned `MTPDrafterModelFactory.swift:107:20: warning: class 'MTPDrafterRegistry' must restate inherited '@unchecked Sendable' conformance`.
    Fix: kept `final`, restored `@unchecked Sendable`, and added the swiftlint disable line with the invariant. Updated the doc comment so that it says the class restates the conformance.
    After the fix: after `touch` on the file, the build shows no MTPDrafterRegistry warning and no compiler warnings or errors. The only remaining line is SwiftPM's `missing creator for mutated node` note about the mlx-swift_Cmlx.bundle, which is a build-system message and not from the compiler. swift-format made no changes.
  timestamp: 2026-11-23T14:53:31.493926+00:00
- actor: claude-code
  id: 01m84ebrbcy2yewm55f7t3ch5a
  text: |-
    ### implement — changed
    - evidence: 1 file — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift. Ran `touch` and then `swift build --build-tests`: 0 MTPDrafterRegistry warnings (1 before), 0 compiler warnings, 0 errors.
    - next: review
  timestamp: 2026-11-23T14:53:32.908871+00:00
- actor: claude-code
  id: 01m84ez24b6zcc5qytxxc6rq3b
  text: |-
    ### test — green
    - evidence: swift build --build-tests (exit 0, no compiler warnings; only a build-system note "missing creator for mutated node ... mlx-swift_Cmlx.bundle/Contents/MacOS"); xcrun xctest on 5 bundles, all exit 0: MLXLMTests 1260 XCTest + 2604 Swift Testing passed, 3 known issues (expected), 0 failures; MLXGuidedGenerationTests 95 passed; MLXFoundationModelsTests 605 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest passed, 0 failures. 0 skipped. IntegrationTesting not run.
    - next: commit
  timestamp: 2026-11-23T15:04:05.515013+00:00
position_column: doing
position_ordinal: '80'
title: MTPDrafterRegistry must restate inherited @unchecked Sendable (build warning)
---
`swift build --build-tests` warns: `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift: warning: class 'MTPDrafterRegistry' must restate inherited '@unchecked Sendable' conformance`. Found while building for ^mga47hg, whose diff does not touch this file. Restate the conformance on the class so the build is warning-free.

#bug