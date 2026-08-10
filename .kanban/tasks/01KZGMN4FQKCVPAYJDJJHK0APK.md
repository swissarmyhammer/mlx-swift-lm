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
- actor: claude-code
  id: 01kznw8zbhbmgghjj0c8batr43
  text: |-
    ### review — findings
    - evidence: round 2, scope `d647975~1..d647975`. Engine gave 0 findings and attempted 0 files. Hand review gives 1 new finding — THIRD-PARTY-NOTICES.md:55. All 7 round 1 items are correct now.
    - next: correct the over-broad statement about the source file headers in `osaurus-ai/vmlx-swift-lm`, then run review again.
  timestamp: 2026-08-10T13:02:30.769303+00:00
- actor: claude-code
  id: 01kznw9wvj85zbbpgq69h9jk9k
  text: |
    ### finish iteration 2 — findings
    - implement: changed — all 7 round-1 findings closed; added osaurus-ai/vmlx-swift-lm MIT block, byte-identical to upstream (md5 89e08e092e9bed6dea46d52063d03b0d, 1107 bytes)
    - test: green — `swift test` exit 0; XCTest 199+13, 0 failures, 2 pre-existing skips; swift-testing 357/80/282/7, 0 failures. The +1 delta (356 to 357) is this task's own new test.
    - commit: d647975 fix(license): close all review findings for third-party notices
    - review: findings — THIRD-PARTY-NOTICES.md:55 (1 new). All 7 round-1 findings independently verified as truly fixed, not only checked off.
    - note: the review engine again attempted 0 files. It has no `*.md` validator glob, and this round it also did not attempt the Swift test file. Every check in rounds 1 and 2 is a manual check.
    - next: implement — correct the overstatement at THIRD-PARTY-NOTICES.md:55
  timestamp: 2026-08-10T13:03:00.978540+00:00
- actor: claude-code
  id: 01kznwnv64evnmzm7k6g90p4fx
  text: |
    ## Third implement pass — round 2 finding

    ### The facts I verified first

    I read the true first line of each file with `gh api ... | base64 -d | head -1`.
    I did not trust memory.

    `osaurus-ai/vmlx-swift-lm`, `Libraries/MLXLLM/Models`:

    | File | First line |
    |---|---|
    | DeepseekV4.swift | `// Copyright © 2026 Osaurus AI. All rights reserved.` |
    | DeepseekV4Compressor.swift | `// Copyright © 2026 Osaurus AI. All rights reserved.` |
    | DeepseekV4Configuration.swift | `// Copyright © 2026 Osaurus AI. All rights reserved.` |
    | DeepseekV4JANGTQ.swift | `// Copyright © 2026 Osaurus AI. All rights reserved.` |
    | DeepseekV4MathHelpers.swift | `// Copyright © 2026 Osaurus AI. All rights reserved.` |
    | DeepseekV3.swift | `// Copyright © 2025 Apple Inc.` |
    | Llama.swift | `// Copyright © 2024 Apple Inc.` |

    So each DeepSeek-V4 file has the Osaurus AI name, and the other files do not.
    The sentence the finding names was not true. The corrected sentence is true.

    I also checked `scouzi1966/mlx-swift-lm`: its four `DeepseekV4*.swift` files all
    have the Osaurus AI header, and `DeepseekV3.swift` has the Apple Inc. header.
    Thus `CONTRIBUTING.md` is correct when it says the DeepSeek-V4 files there have
    that header.

    ### The cause, removed from both files

    The cause is one statement pattern: prose that gives the name `Osaurus AI` to
    more files than the DeepSeek-V4 files. The finding names one line. The new test
    found two more lines with the same cause.

    `THIRD-PARTY-NOTICES.md`:

    1. "The header of each source file in that project gives the name `Osaurus AI`"
       → "The header of each **DeepSeek-V4** source file in that project gives the
       name `Osaurus AI`". I also added the two counter-examples that make the
       limit clear: the headers of `DeepseekV3.swift` and `Llama.swift` give the
       name `Apple Inc.`
    2. "...because that is the name in the header of the source file" → "...in the
       header of each **DeepSeek-V4** source file".
    3. "Thus the ported files in this repository give the copyright to Osaurus AI"
       → "Thus the ported **DeepSeek-V4** files in this repository...".

    `CONTRIBUTING.md`:

    4. Heading "Header block for each new ported file" → "...each new ported
       **DeepSeek-V4** file". This also gives back the exact scope of the user
       decision, which says "every new DeepSeek-V4 file".
    5. "Put this block at the top of each new file you port" → "each new
       **DeepSeek-V4** file you port".
    6. "Give the copyright to Osaurus AI" → "Give the copyright of each ported
       **DeepSeek-V4** file to Osaurus AI".
    7. "It is a true transcription of the header of the source file" → "...of each
       **DeepSeek-V4** source file".
    8. "This block is only for a file you port" → "This block is only for a
       DeepSeek-V4 file that you port from `osaurus-ai/vmlx-swift-lm`", with the
       consequence: a file you port from a different source keeps the name in the
       header of that source file.

    I did not change `THIRD-PARTY-NOTICES.md:100` ("Some DeepSeek-V4 files in that
    repository have this header"). The review calls that form correct, and it is
    true. It is about a different repository from the sentence at line 55, thus the
    two quantifiers do not need to be the same word. Both statements now limit the
    name to the DeepSeek-V4 files, which is what the finding asks for.

    I did not change the header block template. `Osaurus AI` stays there. I did not
    change the verbatim MIT blocks.

    ### The test

    A test can assert the corrected claim, so I wrote one. It is not a check of one
    string. It is the invariant behind the finding:

    `thirdPartyNoticesLimitsTheOsaurusAiNameToDeepSeekV4Files` reads the notice
    file, removes the fenced blocks (they are quotations, not statements of this
    repository), splits the prose into sentences, and asserts that each sentence
    that gives the name `Osaurus AI` also names `DeepSeek-V4`.

    TDD: I wrote the test first and I saw it fail. It failed for the correct cause,
    and it found three sentences, not one:

    ```
    ✘ ... The header of each source file in that project gives the name `Osaurus AI`
    ✘ ... The header block in `CONTRIBUTING.md` keeps `Osaurus AI`, because that is the name in the header of the source file
    ✘ ... Thus the ported files in this repository give the copyright to Osaurus AI
    ```

    The test found the two lines the finding did not name. Then I corrected the
    prose, and all five tests passed.

    I did not write the same test for `CONTRIBUTING.md`. That file gives an
    instruction about the block ("Keep the name `Osaurus AI` in the block above"),
    and the same rule would give a false failure on an instruction that is not a
    statement about which files have the header. The finding gives the cause for
    the notice file: "A notice file is a legal record, thus each statement in it
    must be true."

    ### Evidence

    - `swift build` → `Build complete!`
    - `swift test --filter LicenseNoticeTests` → 5 tests in 1 suite passed, 0 failures
    - `swift-format lint --configuration .swift-format Tests/MLXLMTests/LicenseNoticeTests.swift` → exit 0
    - grep for `sometimes|carry|product|cover|for this reason|goes away|therefore`
      over the three files → no result
    - `thus` and `sufficient` stay in both Markdown files
  timestamp: 2026-08-10T13:09:32.484090+00:00
- actor: claude-code
  id: 01kznwp2rh2pqqqhwb11cs09e3
  text: |
    ### implement — changed
    - evidence: 3 files — CONTRIBUTING.md, THIRD-PARTY-NOTICES.md, Tests/MLXLMTests/LicenseNoticeTests.swift. The round 2 finding (THIRD-PARTY-NOTICES.md:55) is closed, and the same cause is removed from 7 more lines in the two Markdown files. `swift build` complete; `swift test --filter LicenseNoticeTests` 5/5 passed; `swift-format lint` exit 0.
    - next: /review
  timestamp: 2026-08-10T13:09:40.241880+00:00
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

## Review Findings (2026-08-10 08:01)

Scope: `d647975~1..d647975`. Round 2. The change is documentation and one test
file.

Engine result: `Nothing in scope to review` (0 findings, 0 files attempted).
The engine did not attempt the two Markdown files, because no validator has a
match glob for `*.md`. The engine also did not attempt
`Tests/MLXLMTests/LicenseNoticeTests.swift`. All the checks below are by hand.

The verbatim MIT license text in `THIRD-PARTY-NOTICES.md` is a quotation. Do not
change it. The STE rules below do not apply to it.

### The round 1 items are correct now

- `CONTRIBUTING.md:57` — `Sometimes` is gone. The sentence is now "This section
  is the decision about how to attribute code that we port into this repository
  from a different project."
- `CONTRIBUTING.md:90` — `carry` became `have`.
- `THIRD-PARTY-NOTICES.md:3` — `This product` became
  "The `mlx-swift-lm` repository".
- `THIRD-PARTY-NOTICES.md:24` — The new section for `osaurus-ai/vmlx-swift-lm`
  is a faithful copy of the upstream LICENSE. I read the upstream file with
  `gh api repos/osaurus-ai/vmlx-swift-lm/contents/LICENSE`. The block at
  `THIRD-PARTY-NOTICES.md:29-50` and the upstream file are byte-identical: both
  are 1107 bytes with md5 `89e08e092e9bed6dea46d52063d03b0d`, and `diff` shows
  no difference. Both copyright lines are correct:
  `Copyright (c) 2024 ml-explore` and `Copyright (c) 2026 Osaurus contributors`.
- `THIRD-PARTY-NOTICES.md:103` — `For this reason` became `Thus`.
- `Tests/MLXLMTests/LicenseNoticeTests.swift:5` — `covered by` became the active
  clause "give the attribution for ported DeepSeek-V4 code".
- `Tests/MLXLMTests/LicenseNoticeTests.swift:6` — `goes away` became
  "does not exist".
- The six not-approved words `sometimes`, `carry`, `product`, `cover`,
  `for this reason`, and `goes away` are not in any of the three files. A
  case-insensitive search of all three files finds none of them.
- `thus` and `sufficient` stay, and this is correct. `THUS (adv)` is approved
  (page 2-1-T6) and `SUFFICIENT (adj)` is approved (page 2-1-S28). The
  not-approved word `therefore` is not in any of the three files.

### The tests are correct

- The four tests pass: `swift test --filter LicenseNoticeTests` gives
  "Test run with 4 tests in 1 suite passed".
- The new assertions are not vacuous. Each of the three heading strings
  `## osaurus-ai/vmlx-swift-lm`, `## scouzi1966/mlx-swift-lm`, and
  `## scouzi1966/maclocal-api` is in the notice file one time only, at its
  section heading. `Copyright (c) 2026 Osaurus contributors` is in the file one
  time only, in the new block. If a person removes the new section, both new
  tests fail.

### Open items

- [x] `THIRD-PARTY-NOTICES.md:55` — The sentence "The header of each source file
      in that project gives the name `Osaurus AI`" is not true. Only the
      DeepSeek-V4 files in `osaurus-ai/vmlx-swift-lm` have that header. I read
      three files from that repository with `gh api`:
      `Libraries/MLXLLM/Models/DeepseekV4.swift` starts with
      `// Copyright © 2026 Osaurus AI. All rights reserved.`, but
      `Libraries/MLXLLM/Models/DeepseekV3.swift` starts with
      `// Copyright © 2025 Apple Inc.` and
      `Libraries/MLXLLM/Models/Llama.swift` starts with
      `// Copyright © 2024 Apple Inc.`. A notice file is a legal record, thus
      each statement in it must be true. Write "The header of each DeepSeek-V4
      source file in that project gives the name `Osaurus AI`". The same file
      already uses this correct form at `THIRD-PARTY-NOTICES.md:96`: "Some
      DeepSeek-V4 files in that repository have this header". Make the two
      statements agree.