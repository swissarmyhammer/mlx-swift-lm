---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
title: 'MTP: reduce per-round buffer churn (reuse GDN snapshot and attention temporaries between rounds)'
---
Follow-up to the buffer-pool discussion (^1abqd92). A round frees about 0.6–2 GiB (GDN snapshot slots plus attention temporaries that grow with context). The buffer pool only hides that churn. Root-cause fix: reuse those buffers between rounds so little is freed and the pool size stops mattering.

- [ ] Measure bytes allocated and freed per round by source at 2k / 32k / 100k.
- [ ] Reuse the dominant allocations with output unchanged (greedy up to the near-tie rule, sampled distribution unchanged, cache and drafter alignment intact).
- [ ] Measure tok/s and resident memory before and after, on both load paths.

#mtp #perf