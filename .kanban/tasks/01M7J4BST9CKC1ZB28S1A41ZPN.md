---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jb6ses06y3ry3yvaj6n2mz
  text: '### skipped — depends on ^fgktqbt, which is skipped because ^sbwbavm is NO-GO'
  timestamp: 2026-11-16T14:12:04.697359+00:00
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