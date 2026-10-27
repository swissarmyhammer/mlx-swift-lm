---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5ynqkn95sjwyz8b924j5bt3
  text: |-
    Upstream MTP PR check, 2026-10-27:

    | PR | State | Upstream head | Head on `mtp` | Action |
    |---|---|---|---|---|
    | 581 Resumable Qwen MTP in ChatSession | open, last push 2026-10-15 | f1c9c940 | f1c9c940 (843854ec) | none: head unchanged |
    | 459 Distribution-preserving MTP sampling | open, last push 2026-09-14 | 457fc96b | 457fc96b (82f273fd) | none: head unchanged |
    | 724 Batch greedy MTP verification into one GPU readback (aleroot) | open, 2026-10-24 | 5fccc0bc | 5fccc0bc | **ported** |
    | 711 Skip the checkpoint components a model does not have | open | — | — | out of scope: general loader change (also skips the Qwen MTP head in the target); comes with a stable sync |
    | 712 Nemotron 3.5 | open | — | — | out of scope: only skips `mtp.*` weights at load |
    | 607 DFlash2, 426 n-gram speculative, 689 Gemma4 | open | — | — | out of scope for this branch |

    Port of 724: the conflict was in the verify step, which PR 459 had already split into `selectDistributionPreservingTokens` / `selectGreedyTokens`. The batched path now runs ahead of both as a fast path, used only when the distribution path is not in use, the sampler is `ArgMaxSampler`, and the processor is nil or `SuppressTokensProcessor`. Everything else keeps the PR 459 walk.

    Tests: `swift build --build-tests` passes; xctest gives 895 XCTest and 1815 Swift Testing passing, 0 failures.

    Release benchmark `qwen38FourBitMTPGeneration`, M5 Ultra, output equal to plain decode in all categories:

    | | before 724 | after 724 | |
    |---|---|---|---|
    | math | 80.1 tok/s (1.35x) | 82.5 tok/s (1.39x) | +3.0% |
    | code | 71.2 (1.20x) | 74.4 (1.26x) | +4.5% |
    | prose | 70.2 (1.18x) | 71.6 (1.21x) | +2.0% |

    Plain decode is 59.2–59.3 tok/s. Acceptance (93.9 / 73.6 / 69.0%) and tok/verify (1.96 / 1.75 / 1.70) are the same as before the port. The values recorded in the acceptance criteria (51.5 / 44.0 / 29.3) were stale: the branch already ran at 80 / 71 / 70 before the port. The new baseline is 82.5 / 74.4 / 71.6.
  timestamp: 2026-10-27T12:35:31.113255+00:00
position_column: done
position_ordinal: ffe380
title: 'MTP: check the upstream MTP PRs and merge them into the mtp branch'
---
## What
The `mtp` branch holds ports of three open upstream PRs from ml-explore/mlx-swift-lm:
- PR 581 "Resumable Qwen MTP speculative decoding for ChatSession" (ported at head f1c9c940 in commit 843854ec).
- PR 459 "MTP speculative decoding: preserve the target distribution when sampling" (ported at head 457fc96b in commit 82f273fd).
- PR 724 "Batch greedy MTP verification into one GPU readback per round" (ported at head 5fccc0bc, 2026-10-27).

Our `stable` branch takes upstream as squashed merges, so a plain `git merge` of a PR branch conflicts in almost every file. Apply each PR as one diff with `git apply --3way` (diff from the parent of the first PR commit to the PR head), and resolve the conflicts by hand. The port notes are in the bodies of the port commits.

Before any other MTP task starts, and again before a release of the branch:
- [x] Check the state of PR 581 and PR 459 with `gh pr view <n> -R ml-explore/mlx-swift-lm --json state,mergedAt,headRefOid,commits`. Search for other open upstream PRs that touch MTP (`gh pr list -R ml-explore/mlx-swift-lm --search "MTP in:title,body" --state all`), and record each one with a short note on whether it is in scope.
- [x] If a PR head moved since the port, apply only the new commits (diff from the ported head to the new head) and resolve conflicts.
- [ ] If a PR merged upstream, check that the next `merge(stable)` sync from upstream main takes the merged version without a double change, and drop the local port where it becomes redundant.
- [x] Port any other in-scope MTP PR in the same way, as one commit for each PR, with the upstream author and head in the commit body.
- [x] Update the MTP tasks on the board when a port covers part of their work (comment or description).

## Acceptance Criteria
- [x] A comment on this task records, for each upstream MTP PR: its state, the head that the `mtp` branch holds, and the action taken.
- [x] After the merge, `swift build --build-tests` passes and `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes with no failures.
- [x] The Release benchmark `qwen38FourBitMTPGeneration` gives output equal to plain decode on all categories, and MTP tok/s within ±3% of the last recorded value (math 82.5, code 74.4, prose 71.6 tok/s on the M5 Ultra after the PR 724 port, 2026-10-27; plain decode 59.3).

## Tests
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.
- [x] `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES '-only-testing:IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPGeneration()'` passes.

## Workflow
- Use `/tdd` for any code change that a conflict resolution needs: write a failing test first. #mtp #qwen38 #upstream