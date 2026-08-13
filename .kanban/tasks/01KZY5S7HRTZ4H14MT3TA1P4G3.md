---
assignees:
- claude-code
position_column: todo
position_ordinal: 9b80
title: Port onTokenCommitted onto the upstream GuidedGenerationLoop
---
`swift build --build-tests` fails with exactly one error:

```
Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift:186:31:
error: extra argument 'onTokenCommitted' in call
```

`swift build` and `xcodebuild build-for-testing -scheme IntegrationTesting` both pass. This one test target is the only thing left.

## Cause

The upstream catch-up merge (`-X theirs`) replaced `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`. Both sides created that file after the merge base (ours 1235 lines, upstream 598), so the merge took the upstream file whole.

Our `onTokenCommitted: ((Int) -> Void)?` parameter went with it. The upstream `GuidedGenerationLoop.run` has no such parameter and returns `Int`, while the test expects a value with a `tokenCount` member.

## Why this matters

The test is a regression guard for a real bug. Its own comment states it:

> without the fix, `committedTokenIDs` is only [66, 67, 68] ("B", "C", "D") -- the sampled token 'A' (65) that triggered the FF batch is silently never fed through the model

So a sampled token that starts a fast-forward batch never reaches the model. Check whether the upstream loop has this same bug. If it does, the port is a bug fix and not only an API addition.

## Decide first

Either port `onTokenCommitted` through the upstream loop and its helpers, or drop the test if upstream's structure makes the bug impossible. Do not delete the test without proving the bug cannot happen.

Recover the old code with:
`git show pre-upstream-merge-2026-08-13:Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift` #upstream-catch-up-guided-generation