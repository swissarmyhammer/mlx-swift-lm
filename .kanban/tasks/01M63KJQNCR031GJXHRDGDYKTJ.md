---
assignees:
- claude-code
position_column: todo
position_ordinal: a480
title: 'Guided MTP: verify grammar-forced tokens inside the draft round instead of a separate forward'
---
After ^ashrw1b (feed the decided token ahead of the forced tokens), guided JSON MTP dropped from 74.71 to 67.59 tok/s (plain 54.05 → 52.27). Part is the MTP-only cost: a decision that forces tokens now costs a 2-token forward plus a target decision before the next draft round. The implementer checked starting the round after the last forced token (fewer tokens per forward) but not treating forced tokens as zero-cost drafts in the verify batch.

User guidance: small perf gains add up; keep plain and MTP guided output identical.

- [ ] Append grammar-forced tokens to the draft chain as certain drafts (no drafter step), so one verify forward covers decision + forced + further drafts
- [ ] Keep matcher rollback counts and drafter hidden-row pairing correct on rejection after forced tokens
- [ ] Test: plain vs MTP guided output identical over every fixture schema; fed tokens equal emitted tokens
- [ ] Release `qwen38GuidedJSONWithMTP`: report tok/s before/after (baseline plain 52.27, MTP 67.59)

#mtp #guided-generation #perf