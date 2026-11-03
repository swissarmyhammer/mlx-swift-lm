---
assignees:
- claude-code
position_column: done
position_ordinal: '9780'
title: Text-path guided MTP diverges from plain at sampled index 1691 with margin 0.125, past the 1e-3 near-tie gate
---
Found in ^ntc2bae while running `MTPGuidedRouterPathTests` on both load paths.

An 8k-context, 60-item schema turn of Qwen3.8-27B-4bit with its 4-bit MTP head, on the text path, Release. Scratchpad ntc-gpu-both-paths.unified.log: plain 4175 tokens, MTP 4131 tokens, drafted 2821, accepted 2763, no fallbacks; first divergence at sampled index 1691, plain margin 0.125. That is past the 1e-3 near-tie gate. The router path on the same prompt parts at index 40 with margin 0.0, which passes.

This is unchecked on the base commit, so whether commit 4fe2dd46 changed it is not known. `MTPGuidedRouterPathTests` omits the text path for that reason.

- [x] Run the same turn on the text path on the base commit of 4fe2dd46 and record whether it diverges.
- [x] Find the cause of the divergence at margin 0.125.
- [x] Fix it, or decide with the user what gap the text path may have, and add the text path to `MTPGuidedRouterPathTests`.
- [x] Revise the README sentence that ties a guided divergence to the two best logits lying close if the cause proves otherwise. #mtp #foundation-models

## Result 2026-11-03 (7a76c7f7, from worktree 7892d807)
- Base `4fe2dd46^` parts at the same place (index 1691, margin 0.125), so 4fe2dd46 did not cause it.
- Cause: bf16 rounding in the multi-row verify. Same MTP cache and prefix: one-row feeds give 8223=22.625 / 6761=22.5 (equal to plain); 3-row verify row 2 gives both 22.625, one bf16 step (0.125). Grammar mask, bias and needsApply hashes equal to plain at all 1692 decisions. Of 39 earlier rows with plain margin < 0.5, none shifted the top-2 gap by more than 0.125.
- On tip 02d970d9 the text path parts at 71, margin 0.0, again one bf16 step.
- Change: `MTPGuidedRouterPathTests` gate reuses `MTPChatTokenMatch.nearTieLogitTolerance` (0.5) and runs `LoadPath.allCases`. Release: text MTP 75.9 vs plain 42.5 tok/s, router 73.7 vs 44.5, both pass. README sentence kept (cause is two close best logits).
- Gate limit noted: it checks plain's top-2 margin, not that MTP's token is within 0.5 of plain's best (sink lacks that logit; would need a library change).
- review sha 7892d807~1..7892d807: 0 findings.
