---
assignees:
- claude-code
position_column: todo
position_ordinal: 8b80
title: 'scripts/verify-docs.sh fails: broken DocC symbol links and a picojson.h toolchain error'
---
## What

On 2026-10-13, during the test step of ^rnnxh8d, `scripts/verify-docs.sh` exited 1. All the errors are in code that this change did not touch:
- Broken DocC symbol links in doc comments, for example in `DeepSeekV4.swift` (from commit fdfdecf), `DeepSeekV4Cache.swift`, `DeepSeekV4Tokenization.swift`, `SwitchLayers.swift`, `Evaluate.swift` and `KVCache.swift`.
- A toolchain error in the ExtractAPI step for MLXCXGrammar: `picojson.h: 'algorithm' file not found`.

The log is at `/private/tmp/claude-501/scratch/docs.log` (a scratch file; it can be gone).

## Acceptance Criteria

- [ ] `scripts/verify-docs.sh` exits 0, or the MLXCXGrammar toolchain error is recorded as a known exclusion with the reason.
- [ ] Every broken symbol link is corrected or changed to plain code text.
