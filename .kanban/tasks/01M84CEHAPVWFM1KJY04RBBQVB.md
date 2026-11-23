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
- actor: claude-code
  id: 01m84ezpeypyknj14amnn245k7
  text: |-
    ### commit — changed
    - evidence: 4f8d954b fix(mlxlmcommon): restate MTPDrafterRegistry's @unchecked Sendable (^4rbbqvb)
    - next: none
  timestamp: 2026-11-23T15:04:26.334356+00:00
- actor: claude-code
  id: 01m84f0y2mvya5n8zheny32h61
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (4f8d954b) — 1 finding (1 confirmed, 0 refuted) — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift:109
    - next: add a documentation comment for 'MTPDrafterRegistry' (code-hygiene/missing-docs-swift), then re-review
  timestamp: 2026-11-23T15:05:06.900555+00:00
- actor: claude-code
  id: 01m84f1a8t7ckdarxs0xmw63vb
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — MTPDrafterModelFactory.swift (restated @unchecked Sendable + swiftlint disable:next)
    - test: green — 5 bundles (2604/95/605/7/11)
    - commit: 4f8d954b
    - review: findings — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift:109 (missing-docs: MTPDrafterRegistry)
  timestamp: 2026-11-23T15:05:19.386569+00:00
- actor: claude-code
  id: 01m84f56eya804cadkpwsarrj2
  text: |-
    Finding fix (missing-docs-swift on MTPDrafterRegistry).

    Cause confirmed: the `// swiftlint:disable:next no_unchecked_sendable ...` line between the `///` block and the declaration detaches the doc comment. Reproduced the gate locally: `swift format lint --strict` with a config that turns on only `AllPublicDeclarationsHaveDocumentation`, file fed on stdin. Before the fix it reported the registry at 109, plus 8 more undocumented public members of `MTPDrafterModelFactory` in the same file: ContextType, ContainerType, shared, typeRegistry, modelRegistry, init, `_load(configuration:tokenizerLoader:)`, `_wrap(_:)`. Per "remove the cause from the whole file", all 9 are fixed.

    Repo pattern: every other `no_unchecked_sendable` exception uses `disable:next` below the `///` block. That is the same detaching shape. Those declarations pass only because none of them is `public`. No public `@unchecked Sendable` declaration in the repo pairs docs with the exception, so there was no precedent to copy.

    Change:
    - The exception moves onto the declaration line as `// swiftlint:disable:this no_unchecked_sendable <invariant>`, with the same invariant text. The `///` block now sits directly on the declaration. `disable:this` is a standard swiftlint directive.
    - Doc comments are added to the 8 factory members. typeRegistry and modelRegistry follow the LLMModelFactory/VLMModelFactory wording. The `_load` doc says `tokenizerLoader` is unused, which the body confirms.

    After the fix the gate reports 0 findings. swift-format format left the file unchanged.

    Note for the next agent: `swift format lint --configuration .swift-format` (lineLength 100) now reports `[EndOfLineComment] move end-of-line comment that exceeds the line length` on the declaration line. CI and pre-commit run only `swift-format format`, which accepts the line, and the sah swift gates drop layout tags. The fix the brief requires is a trailing directive with the full invariant, so it cannot avoid this note. The other way out would be a `// swiftlint:disable` / `// swiftlint:enable` pair placed above the `///` block and after the declaration. The `AlwaysUseLowerCamelCase` lint warnings on the checkpoint constants were there before this change. Stock `swiftlint` without the gate config does not know `no_unchecked_sendable`, as is true for every existing directive.
  timestamp: 2026-11-23T15:07:26.558883+00:00
- actor: claude-code
  id: 01m84f58nt2q085vx94pwcaj8m
  text: |-
    ### implement — changed
    - evidence: 1 file — Libraries/MLXLMCommon/MTPDrafterModelFactory.swift. missing-docs gate (`swift format lint --strict`, AllPublicDeclarationsHaveDocumentation only): 9 findings before, 0 after. Ran `touch` and then `swift build --build-tests`: MLXLMCommon recompiled, Build complete, 0 compiler warnings (none for MTPDrafterRegistry), 0 errors. The only line left is SwiftPM's `missing creator for mutated node` note. swift-format format made no changes. Test bundles not run (per instructions). Not committed.
    - next: review
  timestamp: 2026-11-23T15:07:28.826348+00:00
- actor: claude-code
  id: 01m84fnk4m6r1qtm97wjkjgaqs
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0, 0 compiler warnings (build was already up to date; only an llbuild note about a missing creator for the Cmlx bundle MacOS node). xcrun xctest on 5 bundles under bench.lock, all exit 0: MLXLMTests 2604 Swift Testing tests + 1260 XCTest, 0 failures, 3 known issues (expected); MLXGuidedGenerationTests 95; MLXFoundationModelsTests 605; CXGrammarTests 7; MLXHuggingFaceMacrosTests XCTest 0 failures. 0 skipped. IntegrationTesting not run.
    - next: commit
  timestamp: 2026-11-23T15:16:23.828273+00:00
- actor: claude-code
  id: 01m84fp4288q78m4r5q675hg0z
  text: |-
    ### commit — changed
    - evidence: 6b7db65d docs(mlxlmcommon): keep MTPDrafterRegistry's doc on its declaration (^4rbbqvb review)
    - next: none (local commit only, not pushed)
  timestamp: 2026-11-23T15:16:41.160722+00:00
- actor: claude-code
  id: 01m84fpyz7tj1h0726tchq94da
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (6b7db65d) — 0 findings, 0 confirmed, 0 refuted (7 validator passes, 0 failed); prior finding MTPDrafterModelFactory.swift:109 checked
    - next: none; card moved to done
  timestamp: 2026-11-23T15:17:08.711986+00:00
- actor: claude-code
  id: 01m84fq7btegqggkghw7p4823k
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — MTPDrafterModelFactory.swift (doc on declaration, disable:this, factory members documented)
    - test: green — 5 bundles (2604/95/605/7/11)
    - commit: 6b7db65d
    - review: clean — 0 findings
  timestamp: 2026-11-23T15:17:17.306086+00:00
position_column: done
position_ordinal: ffffde80
title: MTPDrafterRegistry must restate inherited @unchecked Sendable (build warning)
---
`swift build --build-tests` warns: `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift: warning: class 'MTPDrafterRegistry' must restate inherited '@unchecked Sendable' conformance`. Found while building for ^mga47hg, whose diff does not touch this file. Restate the conformance on the class so the build is warning-free.

#bug

## Review Findings (2026-11-23 09:04)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 1 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M842YQR2G89HR3CNSMGA47HG.jsonl` — no validator matches this file
> - `.kanban/tasks/01M842YQR2G89HR3CNSMGA47HG.md` — no validator matches this file
> - `.kanban/tasks/01M84CEHAPVWFM1KJY04RBBQVB.jsonl` — no validator matches this file
> - `.kanban/tasks/01M84CEHAPVWFM1KJY04RBBQVB.md` — no validator matches this file

- [x] `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift:109` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'MTPDrafterRegistry'.
