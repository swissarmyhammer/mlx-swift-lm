---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: Split Tests/MLXLMTests/Qwen35MTPTests.swift so the review engine can read it whole
---
Found in the ^qrqcy2j review: `Tests/MLXLMTests/Qwen35MTPTests.swift` renders to 265265 bytes, over the review engine's 262144-byte per-file cap, so the duplication validator skips it (`review-engine/prompt-cap`).

- [ ] Split the suite into topic files (for example drafter forward / chained drafts, prompt-cache and state export, loading and sanitize), moving shared helpers into one file-private support file or MLXLMTestSupport.
- [ ] Keep every test and assertion; MLXLMTests test count unchanged.
- [ ] Re-run `review file` on each new file and confirm none is over the cap.

#test #mtp