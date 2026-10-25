---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: 'MTP: check the upstream MTP PRs and merge them into the mtp branch'
---
## What
The `mtp` branch holds ports of two open upstream PRs from ml-explore/mlx-swift-lm:
- PR 581 "Resumable Qwen MTP speculative decoding for ChatSession" (ported at head f1c9c940 in commit 843854ec).
- PR 459 "MTP speculative decoding: preserve the target distribution when sampling" (ported at head 457fc96b in commit 82f273fd).

Our `stable` branch takes upstream as squashed merges, so a plain `git merge` of a PR branch conflicts in almost every file. Apply each PR as one diff with `git apply --3way` (diff from the parent of the first PR commit to the PR head), and resolve the conflicts by hand. The port notes are in the bodies of commits 843854ec and 82f273fd.

Before any other MTP task starts, and again before a release of the branch:
- [ ] Check the state of PR 581 and PR 459 with `gh pr view <n> -R ml-explore/mlx-swift-lm --json state,mergedAt,headRefOid,commits`. Search for other open upstream PRs that touch MTP (`gh pr list -R ml-explore/mlx-swift-lm --search "MTP in:title,body" --state all`), and record each one with a short note on whether it is in scope.
- [ ] If a PR head moved since the port, apply only the new commits (diff from the ported head to the new head) and resolve conflicts.
- [ ] If a PR merged upstream, check that the next `merge(stable)` sync from upstream main takes the merged version without a double change, and drop the local port where it becomes redundant.
- [ ] Port any other in-scope MTP PR in the same way, as one commit for each PR, with the upstream author and head in the commit body.
- [ ] Update the MTP tasks on the board when a port covers part of their work (comment or description).

## Acceptance Criteria
- [ ] A comment on this task records, for each upstream MTP PR: its state, the head that the `mtp` branch holds, and the action taken.
- [ ] After the merge, `swift build --build-tests` passes and `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes with no failures.
- [ ] The Release benchmark `qwen38FourBitMTPGeneration` gives output equal to plain decode on all categories, and MTP tok/s within ±3% of the last recorded value (math 51.5, code 44.0, prose 29.3 tok/s after the PR 581/459 port).

## Tests
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.
- [ ] `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES '-only-testing:IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPGeneration()'` passes.

## Workflow
- Use `/tdd` for any code change that a conflict resolution needs: write a failing test first. #mtp #qwen38 #upstream