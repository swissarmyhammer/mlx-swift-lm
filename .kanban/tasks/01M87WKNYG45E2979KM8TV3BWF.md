---
assignees:
- claude-code
position_column: todo
position_ordinal: '8580'
title: Draft upstream PRs for the JACCL LID fallback and FAST_SYNCH fence fix
---
We depend on forks: swissarmyhammer/mlx `jaccl-no-ipv4` (2907588 JACCL LID fallback; 59f0118 fence kernels from embedded source, Metal 3.2) and swissarmyhammer/mlx-swift `jaccl-no-ipv4` (6eb4ccc). Get both changes upstream so we can go back to ml-explore.

- [ ] Draft the ml-explore/mlx PR description for the JACCL change: `Connection::info()` leaves the GID zero when there's no IPv4-mapped GID, and RTR uses the GRH only when both ends published one, otherwise LID (is_global=0). Evidence: the m5u2↔m5u1 tests (queue-pair ping-pong 12 µs, 33 Gb/s; 2-rank allSum correct; Qwen3.8-27B 33/33 tokens). Reference mlx#3467 and mlx#4191.
- [ ] Draft the PR description for the fence fix: FAST_SYNCH crashed with "Unable to load kernel input_coherent" because fence.metal needs -std=metal3.2 and mlx-swift's SwiftPM build omits it. Decide whether it belongs in mlx or mlx-swift.
- [ ] Per CLAUDE.md, do NOT open the PRs. Give the drafts to the user to read and submit.
- [ ] Once merged and released, switch Package.swift back to ml-explore/mlx-swift and drop the fork. #cluster