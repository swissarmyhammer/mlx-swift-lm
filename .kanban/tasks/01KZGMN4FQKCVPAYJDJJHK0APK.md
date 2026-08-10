---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kznsf6r5q7q4590cbbhe9wk4
  text: |
    ## Decision from the user (2026-08-10)

    The user made these decisions. Do not open them again.

    ### 1. Header block for each new DSV4 file

    Put this block at the top of every new DeepSeek-V4 file:

    ```
    // Copyright © 2026 Osaurus AI. All rights reserved.
    // SPDX-License-Identifier: MIT
    //
    // Ported from osaurus-ai/vmlx-swift-lm
    //   <source path> @ <sha>
    // Manual transcription; no git ancestry.
    ```

    Replace `<source path>` with the path of the source file. Replace `<sha>` with
    the commit SHA that you read. Give the attribution to Osaurus AI, not to
    scouzi1966.

    ### 2. NOTICE file

    Make a `THIRD-PARTY-NOTICES.md` file at the root of the repository. List these
    two reference repositories in it. Include the full MIT license text for each
    one:

    - `scouzi1966/mlx-swift-lm` (MIT)
    - `scouzi1966/maclocal-api` (MIT)

    ### 3. LICENSE file

    Do not change `LICENSE`. The per-file headers and the NOTICE file are
    sufficient.

    ### 4. Manual transcription

    The written decision must say that neither reference repository is a GitHub
    fork. There is no git ancestry to merge. Each port is a manual transcription,
    and a person must add the attribution by hand.
  timestamp: 2026-08-10T12:13:29.221766+00:00
- actor: claude-code
  id: 01kznsmt8vnbe9f501g6s03109
  text: |
    ## Research (implement step)

    I read the user decision comment. I implement it exactly.

    Repository facts I verified again:

    - `gh api repos/scouzi1966/mlx-swift-lm --jq '{fork,parent}'` gives
      `{"fork":false,"parent":null}`.
    - `gh api repos/scouzi1966/maclocal-api --jq '{fork,parent}'` gives
      `{"fork":false,"parent":null}`.
    - Both licenses are MIT. I read the true license text of each one:
      - `scouzi1966/mlx-swift-lm` keeps the upstream line
        `Copyright (c) 2024 ml-explore`.
      - `scouzi1966/maclocal-api` has `Copyright (c) 2025 MacLocalAPI Contributors`.

    Repository layout facts:

    - `CONTRIBUTING.md` is at the root. There is no `docs/` directory. I put the
      decision in `CONTRIBUTING.md`.
    - `LICENSE` stays as it is.
    - Tests use swift-testing (`import Testing`, `@Suite`, `@Test`).
    - `Tests/CXGrammarTests/CompilerTests.swift` shows the way to find a file from a
      test: `URL(fileURLWithPath: #filePath)` and then remove path components.
  timestamp: 2026-08-10T12:16:33.051623+00:00
- actor: claude-code
  id: 01kznstb9tpp386erhkzwmm9t7
  text: |
    ## Implementation

    I wrote the decision of the user, word for word. I did not change the decision.

    Files:

    - `CONTRIBUTING.md` — new section "Attribution for Ported Code". It has three
      parts: "There is no git ancestry", "Header block for each new ported file",
      and "Notice file". The header block is the block from the decision, without a
      change. The section says to give the copyright to Osaurus AI and not to
      `scouzi1966`. It also says not to change `LICENSE`.
    - `THIRD-PARTY-NOTICES.md` — new file at the root. It lists
      `scouzi1966/mlx-swift-lm` and `scouzi1966/maclocal-api`, and gives the full
      MIT license text of each one. I copied each license text from the repository
      itself with `gh api`, so the copyright lines are true:
      `Copyright (c) 2024 ml-explore` and
      `Copyright (c) 2025 MacLocalAPI Contributors`.
    - `Tests/MLXLMTests/LicenseNoticeTests.swift` — new test suite with three tests.

    I did not change any library source file.

    ## TDD

    I wrote the tests first and I saw them fail. The failure was correct: the
    notice file did not exist, and the test printed
    `Self.noticeFile.path → ".../THIRD-PARTY-NOTICES.md"` with
    `fileExists → false`. Then I made `THIRD-PARTY-NOTICES.md`, and the three tests
    passed.

    ## Evidence

    - `swift build` → `Build complete!`
    - `swift test --filter LicenseNoticeTests` → 3 tests in 1 suite passed, 0
      failures.
    - `swift-format lint --configuration .swift-format
      Tests/MLXLMTests/LicenseNoticeTests.swift` → exit 0.

    ### implement — changed
    - evidence: 3 files — CONTRIBUTING.md, THIRD-PARTY-NOTICES.md, Tests/MLXLMTests/LicenseNoticeTests.swift; `swift build` complete; `swift test --filter LicenseNoticeTests` 3/3 passed
    - next: /review
  timestamp: 2026-08-10T12:19:34.330791+00:00
- actor: claude-code
  id: 01kznt5s7mhvc5cgfedtx9ygfh
  text: |-
    ### commit — changed
    - evidence: 35ea7d0 docs: add attribution rules for ported DeepSeek-V4 code (also 8b4e319 chore(kanban): sync board state, a separate kanban-only commit)
    - next: none. The task record is complete.
  timestamp: 2026-08-10T12:25:49.044371+00:00
- actor: claude-code
  id: 01kznv94p1avtvdvswg84h50ks
  text: |
    ### review — findings
    - evidence: 7 findings — CONTRIBUTING.md:57, CONTRIBUTING.md:91, THIRD-PARTY-NOTICES.md:3, THIRD-PARTY-NOTICES.md:11, THIRD-PARTY-NOTICES.md:56, Tests/MLXLMTests/LicenseNoticeTests.swift:5, Tests/MLXLMTests/LicenseNoticeTests.swift:6
    - scope: `review sha 35ea7d0~1..35ea7d0`. The engine gave `Nothing in scope to review` (0 attempted), because no validator matches `*.md`. A control run on `4805454~1..4805454` attempted 9 pairs, thus the engine is healthy.
    - verified correct: all five parts of the decision are present; `LICENSE` did not change; both MIT copyright lines agree with the upstream LICENSE files, character for character.
    - next: correct the six ASD-STE100 items, and add the `osaurus-ai/vmlx-swift-lm` MIT text to the notice file. Then review again.
  timestamp: 2026-08-10T12:45:07.649450+00:00
- actor: claude-code
  id: 01kznvaj716ek9jb8t5w6aqx38
  text: |
    ### finish iteration 1 — findings
    - implement: changed — CONTRIBUTING.md, THIRD-PARTY-NOTICES.md, Tests/MLXLMTests/LicenseNoticeTests.swift
    - test: green — `swift test` exit 0; 199+13 XCTest (0 failures, 2 pre-existing skips) + 356/0/80/282/7 swift-testing, all passed
    - commit: 35ea7d0 (docs); 8b4e319 (separate kanban chore, out of scope)
    - review: findings — CONTRIBUTING.md:57, CONTRIBUTING.md:91, THIRD-PARTY-NOTICES.md:3, THIRD-PARTY-NOTICES.md:11, THIRD-PARTY-NOTICES.md:56, Tests/MLXLMTests/LicenseNoticeTests.swift:5, Tests/MLXLMTests/LicenseNoticeTests.swift:6
    - note: the review engine attempted 0 files — no validator in the fleet has a match glob for `*.md`. All 7 findings come from manual checks. A control run on 4805454 attempted 9 pairs, so the engine is healthy.
    - next: implement — fix all 7 findings
  timestamp: 2026-08-10T12:45:54.273788+00:00
- actor: claude-code
  id: 01kznvmyy5e3997mk3nqbs3c5f
  text: |
    ## Second implement pass — all 7 review findings

    ### 1. The licensing gap (`THIRD-PARTY-NOTICES.md:11`)

    I read the true LICENSE of `osaurus-ai/vmlx-swift-lm` with
    `gh api repos/osaurus-ai/vmlx-swift-lm/license`. I transcribed it. I did not
    invent it. I then made the proof: I put the fenced block of the new section into
    a file and compared it with the upstream text. `diff` gives no difference —
    byte-identical, the two copyright lines included:

    - `Copyright (c) 2024 ml-explore`
    - `Copyright (c) 2026 Osaurus contributors`

    The new section is first in the notice file, because each ported file comes
    from that project. The two `scouzi1966` entries stay as the user decided. This
    addition does not change that decision.

    ### The two names for the same copyright

    `Osaurus AI` and `Osaurus contributors` are both correct, and each one is a true
    transcription:

    - The header of the source file says `Osaurus AI`. The header block in
      `CONTRIBUTING.md` keeps that name.
    - The LICENSE says `Osaurus contributors`. `THIRD-PARTY-NOTICES.md` keeps that
      name.

    A later reader could see this as an error and "correct" one name into the other.
    To stop that, I made the distinction explicit in both files:

    - `CONTRIBUTING.md`, after the header block: a new paragraph that says to keep
      `Osaurus AI` there, and why the notice file uses a different name.
    - `THIRD-PARTY-NOTICES.md`, in the new section: a subsection
      "Two different names, and both are correct".

    Both say: **Do not change one name into the other.**

    ### 2. The six ASD-STE100 items

    I corrected the cause in each whole file, not only the line in the finding. A
    grep for `sometimes|carry|product|cover|reason|goes away` over the three files
    now gives no result.

    | Word | Where | Correction |
    |---|---|---|
    | `Sometimes` | `CONTRIBUTING.md` | The sentence starts with `This section is the decision about how to attribute code that we port...`. The condition replaces the adverb. |
    | `carry` | `CONTRIBUTING.md` | `have`, which the notice file already uses for the same statement. |
    | `product` | `THIRD-PARTY-NOTICES.md` | `The `mlx-swift-lm` repository contains...` — the name of the repository, as the dictionary help says. |
    | `For this reason` | `THIRD-PARTY-NOTICES.md` | `Thus`. |
    | `covered by` | `LicenseNoticeTests.swift` header | Active voice: `per-file headers and one notice file ... give the attribution for ported DeepSeek-V4 code`. |
    | `goes away` | `LicenseNoticeTests.swift` header | `If that notice file does not exist`. |

    `cover` had a second occurrence the finding did not name — the doc comment of
    `thirdPartyNoticesNamesBothReferenceRepositories` said "the MIT license text
    that covers it". That test is now two tests, and neither doc comment uses the
    verb.

    I did not touch `thus` or `sufficient`. The reviewer verified both as approved
    (pages 2-1-T6 and 2-1-S28). I did not add `therefore`. I did not change the
    verbatim MIT blocks; they are quotations.

    ### 3. TDD on the notice-file test

    `osaurus-ai/vmlx-swift-lm` was **already** in the notice file, in the prose at
    line 56. A test of `contains("osaurus-ai/vmlx-swift-lm")` would have passed
    before the fix and proved nothing. So the tests assert the two things the
    finding really asks for:

    - `thirdPartyNoticesHasOneSectionForEachRepository` — the `## ` heading of each
      of the three repositories.
    - `thirdPartyNoticesHasTheCopyrightLineOfEachRepository` — the copyright line of
      each repository, and `MIT License`.

    I saw both fail first, for the correct cause: `## osaurus-ai/vmlx-swift-lm` and
    `Copyright (c) 2026 Osaurus contributors` were absent. Then I wrote the notice
    section, and all four tests passed.

    ### Evidence

    - `diff` of the new MIT block against the upstream LICENSE → `IDENTICAL`
    - `swift build` → `Build complete!`
    - `swift test --filter LicenseNoticeTests` → 4 tests in 1 suite passed, 0 failures
    - `swift-format lint --configuration .swift-format Tests/MLXLMTests/LicenseNoticeTests.swift` → exit 0
    - grep for the six not-approved words over the three files → no result
  timestamp: 2026-08-10T12:51:34.981193+00:00
- actor: claude-code
  id: 01kznvn4qgf8c70qpm75cxpbf9
  text: |
    ### implement — changed
    - evidence: 3 files — CONTRIBUTING.md, THIRD-PARTY-NOTICES.md, Tests/MLXLMTests/LicenseNoticeTests.swift. All 7 review findings closed. `swift build` complete; `swift test --filter LicenseNoticeTests` 4/4 passed; `swift-format lint` exit 0; new MIT block byte-identical to the upstream LICENSE.
    - next: /review
  timestamp: 2026-08-10T12:51:40.912834+00:00
position_column: doing
position_ordinal: '80'
title: Decide attribution/licensing approach for ported DeepSeek-V4 code
---
## What

Before any bulk porting of DeepSeek-V4 code, settle how we attribute it. This blocks every other DSV4 task because it determines the file header every new file must carry.

Provenance chain established during research:

- `scouzi1966/mlx-swift-lm` @ `main` — MIT (verified via `gh api repos/scouzi1966/mlx-swift-lm/license` → `MIT`). Primary Swift reference; same directory layout as us.
- Its `Libraries/MLXLLM/Models/DeepseekV4.swift` header reads verbatim:
  `// Copyright © 2026 Osaurus AI. All rights reserved.` / `// SPDX-License-Identifier: MIT`
  So scouzi1966's copy **derives from `osaurus-ai/vmlx-swift-lm`**. Attribution belongs to Osaurus AI, not scouzi1966.
- `scouzi1966/maclocal-api` — MIT (verified). Integration reference only (`Scripts/apply-mlx-patches.sh`).
- Our own files carry `// Copyright © 2025 Apple Inc.` (see `Libraries/MLXLLM/Models/DeepseekV3.swift:1`), and this repo descends from `ml-explore/mlx-swift-examples` (MIT).

Note: neither reference repo is a GitHub fork (`fork=false`, `parent=none` on both), so there is no git ancestry to merge against — every port is a manual transcription and must be attributed by hand.

Decide and document:
1. The exact header block for new DSV4 files (retain Osaurus AI copyright + SPDX-License-Identifier: MIT, plus a line naming the source repo, path, and the commit SHA read).
2. Whether to add a `THIRD-PARTY-NOTICES` / `NOTICE` file at repo root, or rely on per-file headers alone.
3. Whether `LICENSE` needs an addendum.

Write the decision into `CONTRIBUTING.md` (or `docs/` if that reads better) so later DSV4 tasks can cite it rather than re-litigate.

## Acceptance Criteria

- [x] A written decision exists in a committed file naming: the header template, the attribution target (Osaurus AI), and the source repo + path + SHA convention.
- [x] If a NOTICE/THIRD-PARTY-NOTICES file is chosen, it exists and lists both reference repos with their MIT text.
- [x] The decision explicitly covers the "manual transcription, no git ancestry" situation.
- [x] No source-code changes in this task — documentation only.

## Tests

- [x] `swift build` still succeeds (proves no accidental source edits): `swift build 2>&1 | tail -5`
- [x] Add a test asserting the license/notice file exists and is non-empty if one was created, in `Tests/MLXLMTests/LicenseNoticeTests.swift`; run `swift test --filter LicenseNoticeTests`

## Workflow
- Documentation-only task; `/tdd` does not apply beyond the file-existence test.
#deepseek-v4

## Review Findings (2026-08-10 07:40)

Scope: `35ea7d0~1..35ea7d0`. The change is documentation only.

Engine result: `Nothing in scope to review` (0 findings, 0 files attempted). No
validator in the fleet has a match glob for `*.md`, thus `CONTRIBUTING.md` and
`THIRD-PARTY-NOTICES.md` got no automatic review. The engine is healthy: a
control run on `4805454~1..4805454` attempted 9 pairs and gave 57 findings. The
items below come from the checks that the engine cannot do.

The verbatim MIT license text in `THIRD-PARTY-NOTICES.md` is a quotation. Do not
change it. The STE rules below do not apply to it.

### Correct. No work is necessary

- The header block has the Osaurus AI copyright, `SPDX-License-Identifier: MIT`,
  the `Ported from osaurus-ai/vmlx-swift-lm` and `<source path> @ <sha>` lines,
  and `Manual transcription; no git ancestry.` (`CONTRIBUTING.md:76`)
- The decision gives the copyright to Osaurus AI, and not to scouzi1966.
  (`CONTRIBUTING.md:90`)
- `THIRD-PARTY-NOTICES.md` is at the root of the repository. It lists both
  reference repositories with the full MIT text.
- `LICENSE` did not change. The commit changed only three files.
  (`CONTRIBUTING.md:103` also tells the writer not to change `LICENSE`.)
- The decision says that neither reference repository is a GitHub fork, that
  there is no git ancestry, and that a person transcribes and attributes each
  port by hand. (`CONTRIBUTING.md:61`, `THIRD-PARTY-NOTICES.md:15`)
- The two MIT copyright lines are correct transcriptions, and not invented.
  `Copyright (c) 2024 ml-explore` and `Copyright (c) 2025 MacLocalAPI
  Contributors` agree with the upstream LICENSE files, character for character.
  The MIT body for `scouzi1966/mlx-swift-lm` is byte-identical to upstream
  (md5 `fd75e95d656151ace036b5ae71caa83f`). The body for
  `scouzi1966/maclocal-api` differs from upstream by one trailing newline only.

### Open items

- [x] `CONTRIBUTING.md:57` — `Sometimes` is not in the ASD-STE100 Issue 9
      dictionary. The alphabetical run is `SOME (adj)`, `SOME (pron)`,
      `SOMETHING (pron)`, `soon (adv)` (page 2-1-S17). Rule 1.1 permits only
      dictionary words, technical nouns, and technical verbs. Write the sentence
      again with a condition, or with `SOME` and a noun.
- [x] `CONTRIBUTING.md:91` — `carry` is a not-approved verb. The dictionary
      approved alternative is `TRANSMIT (v)`, which does not agree with this
      meaning. Use `have`, which `THIRD-PARTY-NOTICES.md:49` already uses for
      the same statement about the same header.
- [x] `THIRD-PARTY-NOTICES.md:3` — `product` is a not-approved noun (page
      2-1-P). The dictionary help says to use the name of the product. Name this
      repository instead of `This product`.
- [x] `THIRD-PARTY-NOTICES.md:11` — The notice file lists only the two
      scouzi1966 reference repositories, but `CONTRIBUTING.md:79` tells the
      writer that each ported file comes from `osaurus-ai/vmlx-swift-lm`, and
      `CONTRIBUTING.md:90` gives the copyright to Osaurus AI. The MIT text of
      `osaurus-ai/vmlx-swift-lm` is not in the notice file. That repository is
      public, and its LICENSE has two copyright lines: `Copyright (c) 2024
      ml-explore` and `Copyright (c) 2026 Osaurus contributors`. Add an entry
      for `osaurus-ai/vmlx-swift-lm` with its full MIT text, because the MIT
      license makes the copyright notice necessary for the code we use.
- [x] `THIRD-PARTY-NOTICES.md:56` — `For this reason` uses the not-approved noun
      `reason (n)` (page 2-1-R3), whose approved alternatives are `CAUSE (n)`
      and `BECAUSE OF (prep)`. Use `Thus`, `As a result`, or `Because of`.
      Note that `THUS (adv)` is approved (page 2-1-T6), thus
      `CONTRIBUTING.md:67` is correct as written.
- [x] `Tests/MLXLMTests/LicenseNoticeTests.swift:5` — `covered by` uses the
      not-approved verb `cover (v)`, whose approved alternative is
      `INCLUDE (v)`. The clause is also passive. Write it again in the active
      voice with an approved verb.
- [x] `Tests/MLXLMTests/LicenseNoticeTests.swift:6` — `goes away` is a phrasal
      verb, and it breaks Rule 9.3 ("When you use two words together, do not
      make phrasal verbs"). `GO (v)` has the restricted approved meaning "To
      move to or from something", which this sentence does not use. Write "if a
      person removes that notice file", or "if that notice file does not exist".