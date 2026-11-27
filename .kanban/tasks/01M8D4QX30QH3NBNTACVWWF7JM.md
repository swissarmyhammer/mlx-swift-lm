---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8dntb2r0b14f78s1gn50k2s
  text: 'Research: RemoteModelSession.send(_:) stores `clusterError(error)` in `failure` and throws it; every later op throws the stored failure, so a new `.memberLost` sticks with no extra code once clusterError maps it. recordFailure(_:) also goes through clusterError. Fake transport to reuse: InProcessRemoteModelWorker (Tests/MLXLMTestSupport) — `failOps(after:)` throws InProcessWorkerFault only; plan: add a `with:` error parameter (default InProcessWorkerFault) and a test-support member-loss fault conforming to the new protocol. Existing test `aFailedOpFailsEveryLaterOpOfTheModel` covers "other errors still map to transportFailed". No exhaustive switch over ClusterModelError outside RemoteModelTransport.swift.'
  timestamp: 2026-11-27T04:57:00.760862+00:00
- actor: claude-code
  id: 01m8dpcxch9m0p800tz16kh2w9
  text: |-
    Implementation landed (TDD: the two memberLost tests failed first with `.transportFailed("the cluster member mac-studio-2 is gone")`, then passed after the mapping).
    - MLXLMCommon: `public protocol ClusterMemberLoss: Error { var lostMember: String { get } }`; `ClusterModelError.memberLost(String)`, errorDescription "The cluster lost the Mac <member>. Load the model again to use it."; `RemoteModelSession.clusterError` maps ClusterModelError → itself, `any ClusterMemberLoss` → `.memberLost(lostMember)`, else `.transportFailed`. Stickiness comes from the existing stored `failure` (send/recordFailure), no new path.
    - Test support: `InProcessMemberLoss` (conforms ClusterMemberLoss); `InProcessRemoteModelWorker.failOps(after:with:)` (default fault InProcessWorkerFault, so old callers unchanged).
    - Tests (ClusterLanguageModelTests): op fails with memberLost; after a loss recorded by a non-throwing trim, the worker recovers (`failOps(after: .max)`) and the next op still throws memberLost (proves it sticks, does not re-send); description names the member. Existing "a failed op fails every later op of the model" covers other errors → transportFailed.
    - Docs: distributed-inference.md paragraph + topics entry for ClusterMemberLoss.

    Message to relay to the FoundationModelsCluster and Router sessions (not sent by this step — no cross-repo channel, and other repos are off limits):
    "mlx-swift-lm (branch cluster, uncommitted at time of writing) adds `public protocol ClusterMemberLoss: Error { var lostMember: String { get } }` and `ClusterModelError.memberLost(String)` in MLXLMCommon. Cluster: conform `ClusterRuntimeError.workerLost` to `ClusterMemberLoss` (lostMember = the Mac's name or rank). Router (e0mnse8): match `ClusterModelError.memberLost(let member)` to evict the clustered model and fall back to local; every later op of that model throws the same case."
  timestamp: 2026-11-27T05:07:09.329383+00:00
- actor: claude-code
  id: 01m8dpd107z9xr25yhja9p22aw
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift, Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift, Tests/MLXLMTests/ClusterLanguageModelTests.swift. swift build --build-tests OK (only the pre-existing "missing creator for mutated node … mlx-swift_Cmlx.bundle" build-system warning). xcrun xctest MLXLMTests: exit 0, 1261 XCTest + 2787 Swift Testing passed, 0 failures (3 known issues are existing withKnownIssue tests in AttentionKernelToleranceTests/ToolTests). MLXFoundationModelsTests: exit 0, 618 passed. swift-format applied to the 3 Swift files.
    - next: relay the prepared message to the Cluster and Router sessions (last acceptance box, open); /review.
  timestamp: 2026-11-27T05:07:13.031654+00:00
- actor: claude-code
  id: 01m8dphk4k36n93msjkz6b5tew
  text: 'Iteration 2 picked up. Design change from the Cluster session: `ClusterRuntimeError` is one enum, so the whole enum conforms and most of its cases are not member losses → requirement becomes `var lostMember: String? { get }`; `clusterError` maps to `.memberLost` only for a non-nil `lostMember`, else `.transportFailed(String(describing:))`. Coordinator added: `lostMember` / the `.memberLost` payload is descriptive text for logs and messages, not one host to parse (a hang names no single Mac, so FoundationModelsCluster lists every member''s .local name, comma-separated); wording becomes "The cluster lost <member>", not "the Mac <member>". Discovery: `xcrun xctest -XCTest <Suite>` runs only XCTest and skips every Swift Testing test (0 executed), so a Swift Testing RED/GREEN run must use the whole bundle.'
  timestamp: 2026-11-27T05:09:42.675543+00:00
- actor: claude-code
  id: 01m8dq7kgxsas874kxgkeydvdd
  text: |-
    Iteration 2 landed (TDD). RED: the new test "a member-loss error that names no member fails the op with transportFailed" failed with `.memberLost("")` against a stub mapping (`lostMember ?? ""`); GREEN after `if let member = (error as? any ClusterMemberLoss)?.lostMember { return .memberLost(member) }`.
    - Protocol: `ClusterMemberLoss: Error { var lostMember: String? { get } }`; doc says the whole error type conforms, nil = no member lost → transportFailed; the words are for logs/messages (one member or a comma-separated list of `.local` names, e.g. after a hang), not to be parsed for one host.
    - `memberLost` doc + errorDescription now "The cluster lost <member>. Load the model again to use it." (no "the Mac").
    - docc distributed-inference.md paragraph rewritten to match.
    - Test support: `InProcessMemberLoss.lostMember` is `String?`; description "the cluster lost <member>" or "the cluster failed and lost no member". Test constant now "mac-studio-1.local, mac-studio-2.local".
    - Existing memberLost tests unchanged and passing.
  timestamp: 2026-11-27T05:21:43.965586+00:00
- actor: claude-code
  id: 01m8dq7pkb8f6q23ph5rtn520v
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift, Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift, Tests/MLXLMTests/ClusterLanguageModelTests.swift. `swift build --build-tests` exit 0 (only the pre-existing "missing creator for mutated node … mlx-swift_Cmlx.bundle" build-system warning). `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` exit 0: 1261 XCTest, 0 failures; 2788 Swift Testing passed with 3 known issues (existing withKnownIssue tests). swift-format applied to the 3 Swift files.
    - next: /review (other bundles not run in this step).
  timestamp: 2026-11-27T05:21:47.115842+00:00
- actor: claude-code
  id: 01m8dqk68g57wr2r94cp35xp5a
  text: |-
    ### test — green
    - evidence: swift build --build-tests exit 0, zero warnings (only the known "missing creator" line). xcrun xctest on all five bundles, all exit 0: MLXLMTests XCTest 1261 tests 0 failures + Swift Testing 2788 tests/279 suites passed (3 known issues, expected withKnownIssue); MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 618 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests XCTest 11 tests 0 failures. No skips, no code changes.
    - next: review/commit
  timestamp: 2026-11-27T05:28:03.600135+00:00
- actor: claude-code
  id: 01m8dqnk1ntjhjcskbqwkfsafk
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (4fc41044): 0 findings (attempted 7, failed 0); not reviewed (no validator matched): .kanban/tasks/01M8D4QX30QH3NBNTACVWWF7JM.jsonl, .kanban/tasks/01M8D4QX30QH3NBNTACVWWF7JM.md, Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md
    - next: none; task moved to done
  timestamp: 2026-11-27T05:29:22.229480+00:00
- actor: claude-code
  id: 01m8dqnyp05gv93vbyyfqnb1rw
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 4 files. The design was reworked to `lostMember: String?` at the Cluster session's request.
    - test: green — all 5 bundles: MLXLMTests 1261 XCTest + 2788 Swift Testing, Guided 96, FM 618, CXGrammar 7, Macros 11
    - commit: 4fc41044
    - review: clean — 0 findings
  timestamp: 2026-11-27T05:29:34.144425+00:00
position_column: done
position_ordinal: ffffff8580
title: A lost cluster member surfaces as a typed ClusterModelError, not transportFailed(String)
---
## What
`RemoteModelTransport.clusterError(_:)` (Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift) maps every non-`ClusterModelError` to `.transportFailed(String(describing: error))`. FoundationModelsCluster throws `ClusterRuntimeError.workerLost` when a member Mac dies. That error becomes a string, so a host (the Router) can't tell a lost member apart from any other transport failure. It can't evict the dead clustered model and fall back to local.

mlx-swift-lm can't import FoundationModelsCluster. The transport needs a seam that lets the lower layer mark an error as a lost member:
- Add `case memberLost(String)` to `ClusterModelError`. The payload is descriptive text for logs and messages, not a single host to parse: a hang names no single Mac, so FoundationModelsCluster lists every member's `.local` name, comma-separated.
- Add a public protocol in MLXLMCommon: `ClusterMemberLoss: Error { var lostMember: String? { get } }`. `ClusterRuntimeError` is one enum, so the whole enum conforms, and most of its cases are not member losses: they return nil. `clusterError` maps to `.memberLost(member)` only when `(error as? ClusterMemberLoss)?.lostMember` is non-nil. A conforming error whose `lostMember` is nil still maps to `.transportFailed(String(describing:))`.
- Every later op after a member loss fails with the same `.memberLost`, the same way `transportFailed` sticks today.
- `errorDescription` for the new case gives plain words: "The cluster lost <member>" (not "the Mac <member>") and that the model must reload.
- Doc comments and the docc article (distributed-inference.md) say the above.

Requested by the Router session for its task e0mnse8, which covers Router eviction plus local fallback and hardware assertions for xsnhzft steps 3 and 5. FoundationModelsCluster needs a follow-up to conform `ClusterRuntimeError`.

## Acceptance
- [x] A transport that throws a `ClusterMemberLoss` error with a non-nil `lostMember` makes the op fail with `ClusterModelError.memberLost`, and every later op fails the same way
- [x] Other errors still map to `transportFailed`
- [x] A `ClusterMemberLoss` error whose `lostMember` is nil maps to `transportFailed(String(describing:))`
- [x] Doc comments, `errorDescription` and the docc article describe the optional `lostMember` and its descriptive (comma-separated, not parsed) words
- [x] Unit tests in MLXLMTests with a fake transport cover these cases
- [x] Cluster and Router sessions told the new case and protocol names (the orchestrator relayed them)
#cluster #cross-repo