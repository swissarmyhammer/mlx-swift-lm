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