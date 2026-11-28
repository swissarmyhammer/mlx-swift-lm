---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g9fjpp7pthev0270daqjzj
  text: |-
    ### Preference (foundationmodelsacpagent-a1, 2026-11-27)
    If a remote trim to the divergence point isn't cheap, implement `copyPromptCache(fromSession:toSession:keepingTokens: n)` so the Router can make an instructions-only template.
  timestamp: 2026-11-28T05:19:08.502370+00:00
position_column: todo
position_ordinal: '8180'
title: 'Prompt cache: a session seeded by copying a remote template gets no partial hit when its prompt diverges after the shared prefix'
---
## What
Bench cluster6i (Router 7bd4949f, mlx 35285ea9, Inkling on 2 Macs). The Router seeded a standard-slot session with copyPromptCache from a template, and the copy returned `.copied`. The template held a full first-session ledger: instructions, prompt and answer. The seeded session's first call had 4924 tokens in and tokens.cached=0. Its prompt shares only the instructions with the template, so the checkout should reuse the instructions and drop the rest. It went cold instead. Reported by foundationmodelsrouter-22.
^jxyd0ja's tests only extended the template, so divergence was never tested.

## Investigate first
Find the actual cause before changing code. Possible causes:
- **Can't trim:** a RemoteKVCache checkout can't trim back to the divergence point, because the Inkling cache's rewindLimit / maxTrimCount on the worker doesn't allow it (sliding window or recurrent layers).
- **Checkpoint dropped:** the plan's prompt-cache checkpoint at the instructions boundary, which a local Inkling cache would use, isn't carried by the copy, or doesn't work for remote checkpoints.
- **Different key:** a cache key or configuration mismatch between the template and the seeded session.
Compare with the same flow on a local Inkling model.

## Fix
Make the seeded session get a partial hit on the shared prefix, by trimming, by a checkpoint, or by both. If trimming is impossible for this cache, add `copyPromptCache(fromSession:toSession:keepingTokens:)`. It copies only the first n ledger tokens, or the nearest checkpoint at or below n, so the Router can make an instructions-only template. Record on the card which approach you took and why.

## Tests
- A session seeded from a remote template, whose prompt diverges after the shared instructions prefix, reports cached tokens > 0 (about the prefix length). Its output equals a cold run.
- The same test locally.
- If `keepingTokens:` is added: it copies exactly n tokens, or the nearest checkpoint at or below n, and refuses n larger than the ledger.
#cluster