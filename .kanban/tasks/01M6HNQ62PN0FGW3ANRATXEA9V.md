---
assignees:
- claude-code
position_column: done
position_ordinal: '9680'
title: Split Tests/MLXLMTests/Qwen35MTPTests.swift so the review engine can read it whole
---
Found in the ^qrqcy2j review: `Tests/MLXLMTests/Qwen35MTPTests.swift` renders to 265265 bytes, over the review engine's 262144-byte per-file cap, so the duplication validator skips it (`review-engine/prompt-cap`).

- [x] Split the suite into topic files (for example drafter forward / chained drafts, prompt-cache and state export, loading and sanitize), moving shared helpers into one file-private support file or MLXLMTestSupport.
- [x] Keep every test and assertion; MLXLMTests test count unchanged.
- [x] Re-run `review file` on each new file and confirm none is over the cap.

#test #mtp

## Result 2026-11-03 (fc6d60c5, worktree d2785c63)
Split into Qwen35MTPLoadingTests.swift (30358 B), Qwen35MTPMetalTests.swift (24042), Qwen35MTPDrafterStateTests.swift (25120), Qwen35MTPGenerateLoopTests.swift (22556) and Qwen35MTPTestSupport.swift (1541; shared tolerances, seededTokens, targetHiddenStates). Pure move: test names, suite names, traits and assertions unchanged; two tolerances went from private to internal for sharing. MLXLMTests 2123 Swift Testing / 938 XCTest before and after; all five bundles green. review sha HEAD~1..HEAD on mtp: 0 findings, no prompt-cap on any new file.
