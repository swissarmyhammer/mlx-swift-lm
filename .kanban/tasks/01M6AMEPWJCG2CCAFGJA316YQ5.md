---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6c734h14f3qhqnqqfc4r8qd
  text: |-
    ### review — clean
    - evidence: `review sha 80b91675~1..80b91675`: 0 findings, 0 confirmed, 0 refuted, 7 attempted, 0 failed, 0 skipped; 3 files reviewed. No builds or tests run in this pass.
    - next: none on this card. CI follow-up is ^p4r82ye. Commit 80b91675 is on wt-a316yq5 only.
  timestamp: 2026-11-01T18:49:02.241024+00:00
position_column: done
position_ordinal: ffff9d80
title: MLXFoundationModelsTests fails under swiftpm-testing-helper --filter '.*' while xcrun xctest passes, and no CI job runs these tests on Xcode 27
---
Found in the ^84b33jx review (2026-10-31). Not fixed there. The cause is not established.

The whole MLXFoundationModelsTests bundle fails under `swiftpm-testing-helper --filter '.*'` and passes under `xcrun xctest`. It fails on clean HEAD too, so it predates the ^84b33jx change and depends on the runner.

| tree | runner | result | failing test |
|---|---|---|---|
| ^84b33jx working tree | `swiftpm-testing-helper --filter '.*'` | failed 4 of 4 | "A tool body may generate on the same model while its turn is in flight" |
| `git archive` of HEAD | `swiftpm-testing-helper --filter '.*'` | failed 4 of 4 | that test twice; "a turn whose stream body throws checks in nothing, and the next turn starts cold" twice |
| ^84b33jx working tree | `xcrun xctest` | passed 3 of 3 (460 tests) | none |
| `git archive` of HEAD | `xcrun xctest` | passed 1 of 1 | none |

Logs in the session scratchpad of ^84b33jx: 84-rev3-allrep1.log to 84-rev3-allrep3.log, 84-rev3-head1.log to 84-rev3-head4.log, 84-rev3-xct1.log to 84-rev3-xct3.log, 84-rev3-headxct.log.

No CI job runs these tests on Xcode 27:

- The PR job (`.github/workflows/pull_request.yml`) runs only `MLXLMTests.xctest`.
- The nightly (`integration_tests.yml`) pins Xcode 26, where `canImport(FoundationModels, _version: 2)` compiles these tests out.
- The Xcode 27 job (`integration_build_xcode27`) is `if: false`, and it is compile-only. It needs a macOS 27 host, which no hosted image offers yet (GitHub issue 499).

So the two failures above, and every test of this bundle, reach no CI.

- [x] Establish why the two named tests fail under the helper's whole-bundle run and pass under `xcrun xctest`.
- [x] Fix the cause, and show the bundle passes under both runners on repeated runs.
- [x] Decide with the user how these tests run in CI once a macOS 27 host exists (GitHub issue 499). Decided: see ^p4r82ye. No macOS 27 host exists yet, so nothing can change now.

## Result (commit 80b91675, branch wt-a316yq5)

Cause: a test isolation bug. `evictAll()` is process-wide, and 4 tests ran it in the test process, so a rebuilt scripted model replays its script.

Fix: the 4 tests run in child processes.

- After the fix: helper 5/5, xctest 3/3.
- Mutants A and B caught.
- Before the fix: helper failed 2/2.

#foundation-models #test