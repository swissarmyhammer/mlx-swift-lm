---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m88byf0t2p0g10n0mvnqshe3
  text: 'Ready to merge. ^vvghkm1 is done and `cluster` is pushed at ebb68982: 36 commits ahead of origin/stable and 0 behind, so no rebase is needed. The PR description is drafted at /private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/3b343604-bb7c-498e-a083-6249d59b3c42/scratchpad/cluster-to-stable-pr.md and waits for the user to open the PR. Open item: Package.swift pins the mlx-swift fork branch until upstream lands.'
  timestamp: 2026-11-25T03:28:17.946558+00:00
depends_on:
- 01M87WKCQDVDJKNNFN5VVGHKM1
- 01M87WJZQS8TVVF0QD9RPAFC4Q
- 01M87WK2PZEB15W2RH7S1WZX5S
- 01M87WK7TF0XNSNQKAYFFN1STG
position_column: done
position_ordinal: fffffc80
title: Merge cluster into stable
---
Land the cluster work on `stable`, the mainline every FoundationModels package depends on.

- [ ] Rebase `cluster` onto the latest origin/stable. Resolve conflicts, rerun all 5 test bundles plus shard-check for Qwen3.8 and Inkling.
- [ ] Note in the PR that `stable` will then depend on the swissarmyhammer/mlx-swift fork (jaccl-no-ipv4). Every downstream package picks that up.
- [ ] Draft the PR description for the user to read and open (CLAUDE.md: don't open it yourself).

Depends on: the review+docs task, the ClusterLanguageModel seam, the cache-key discriminator, and chunked eval at load. #cluster