---
assignees:
- claude-code
depends_on:
- 01M7J4BPYK5W5F22NDZFGKTQBT
position_column: todo
position_ordinal: '8280'
title: 'Real-workload A/B: DFlash2 4-bit vs MTP on ACP SWE-bench by context band'
---
Do this only if the router-path card lands.

## What
- Push the branch.
- Send the SHA to Router and ACP.
- Ask ACP to run a SWE-bench A/B: DFlash2 4-bit vs MTP blockSize 4, on the hint24 tasks or a subset, using the same telemetry split as hint24.

## Acceptance
- [ ] tok/s and tokens per verify, by band (0–8k, 8–16k, 16–32k, 32–64k, 64k+), for both.
- [ ] Resolved count is unchanged.
- [ ] A recommendation on which speculator the router should default to, recorded on this card. #dflash2-4bit