---
assignees:
- claude-code
position_column: done
position_ordinal: 8f80
title: 'MTP: reduce per-round buffer churn (reuse GDN snapshot and attention temporaries between rounds)'
---
Follow-up to the buffer-pool discussion (^1abqd92). A round frees about 0.6–2 GiB (GDN snapshot slots plus attention temporaries that grow with context). The buffer pool only hides that churn. Root-cause fix: reuse those buffers between rounds so little is freed and the pool size stops mattering.

- [x] Measure bytes allocated and freed per round by source at 2k / 32k / 100k.
- [x] Reuse the dominant allocations with output unchanged (greedy up to the near-tie rule, sampled distribution unchanged, cache and drafter alignment intact).
- [x] Measure tok/s and resident memory before and after, on both load paths.

#mtp #perf

## Result 2026-11-03 (8c7bbab0, 2575a9a3, 0d8003b9, c93bef15; worktree aeb72420, 38c85305, 23c53219)
- Churn per round, text path (MTPRoundBufferChurnTests, median of 32 rounds, transient / freed MiB): 2k 45.5/139.4 -> 30.6/89.3; 32k 187.6/282.1 -> 21.0/198.4; 100k 443.1/737.7 -> 23.6/364.9.
- Sources: (1) the target's shared K/V for the drafter are views of the last full-attention layer's K/V buffer; kept alive through the verify, they forced MLX to copy the whole buffer each round (422 MiB at 100k). The Qwen MTP drafter never reads them, so the iterator drops them before the verify and the K/V write runs in place. (2) GDN snapshot slots: 146 MiB per verified token; the slot after the last verify token is never selected by a rollback, so the kernel and op path now keep T-1 slots (4-token verify 583 -> 437 MiB). (3) Unchanged: K/V growth every 256 tokens copies all 16 attention layers (~6.7 GiB at 100k).
- ABBA 3 runs per arm, MTP mean tok/s: router 100k 55.85 -> 56.60 (+1.3%, plain floor about ±0.5%); text 100k 60.1 -> 62.7 (plain varied up to 8 tok/s, noisy); 2k and 32k within noise on both paths. MTP peak memory down on every run: text -158/-311/-567 MiB, router -301/-455/-711 MiB at 2k/32k/100k.
- Greedy tokens identical between base and head (same first divergence from plain and same acceptance in every run).
- Tests: MTPRoundBufferReuseTests (fails on base, passes on head), aVerifyKeepsNoSlotForItsLastToken; existing slot/rollback/rewind/restore tests updated to T-1. Release guards qwen38FourBitMTPGeneration and qwen38FourBitMTPSystemPromptCache pass (edit turn 123 tokens, equal to plain).
- review sha (merged 3 commits): 4 findings (magic numbers :77/:104, print :108 in MTPRoundBufferChurnTests, duplicate bufferAddress helper) fixed in c93bef15; re-review 0 findings. On mtp: five bundles green on the merge (MLXLMTests 938 + 2123, 1 known issue), MLXLMTests re-run after the fix green, IntegrationTesting build-for-testing succeeded.
