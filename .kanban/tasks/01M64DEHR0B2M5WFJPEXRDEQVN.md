---
assignees:
- claude-code
position_column: todo
position_ordinal: '8380'
title: 'Benchmarks: MTPChatTokenMatch must accept an exact top-2 tie (gap 0) whichever token argmax picked'
---
MTPNearTieDivergenceTests (^j2tnzpn phase 2) failed on the router path: first difference at generated token 74, plain 58614 vs MTP 16570, top-2 gap 0.0 on both the plain row and the verify row — an exact tie. The replay ranked 16570 first while plain decode wrote 58614 ("replay reproduces plain false"), so the check failed on tie ordering and printed the misleading "Bug, no near tie". By the near-tie rule this is a tie, not an MTP bug.

- [ ] MTPChatTokenMatch: treat the divergence as a near tie when the two tokens are both within `nearTieLogitTolerance` of the row's max (not "plain's token is top-1"), so argmax tie-breaking order does not matter.
- [ ] Unit test with an exact tie in both orders; keep a real non-tie divergence failing.
- [ ] Re-run MTPNearTieDivergenceTests on both paths.

#mtp #benchmark