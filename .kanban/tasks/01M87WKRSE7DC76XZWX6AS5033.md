---
assignees:
- claude-code
depends_on:
- 01M87WKCQDVDJKNNFN5VVGHKM1
- 01M87WJZQS8TVVF0QD9RPAFC4Q
- 01M87WK2PZEB15W2RH7S1WZX5S
- 01M87WK7TF0XNSNQKAYFFN1STG
position_column: todo
position_ordinal: '8680'
title: Merge cluster into stable
---
Land the cluster work on `stable`, the mainline every FoundationModels package depends on.

- [ ] Rebase `cluster` onto the latest origin/stable. Resolve conflicts, rerun all 5 test bundles plus shard-check for Qwen3.8 and Inkling.
- [ ] Note in the PR that `stable` will then depend on the swissarmyhammer/mlx-swift fork (jaccl-no-ipv4). Every downstream package picks that up.
- [ ] Draft the PR description for the user to read and open (CLAUDE.md: don't open it yourself).

Depends on: the review+docs task, the ClusterLanguageModel seam, the cache-key discriminator, and chunked eval at load. #cluster