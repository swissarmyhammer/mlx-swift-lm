---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: A lost cluster member surfaces as a typed ClusterModelError, not transportFailed(String)
---
## What
`RemoteModelTransport.clusterError(_:)` (Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift:252) maps every non-`ClusterModelError` to `.transportFailed(String(describing: error))`. FoundationModelsCluster throws `ClusterRuntimeError.workerLost` when a member Mac dies. That error becomes a string, so a host (the Router) can't tell a lost member apart from any other transport failure. It can't evict the dead clustered model and fall back to local.

mlx-swift-lm can't import FoundationModelsCluster. The transport needs a seam that lets the lower layer mark an error as a lost member:
- Add `case memberLost(String)` to `ClusterModelError`, with the member's name or rank for the message.
- Add a public marker protocol in MLXLMCommon, e.g. `ClusterMemberLoss: Error { var lostMember: String { get } }`. `clusterError` maps any error conforming to it to `.memberLost`. FoundationModelsCluster then conforms `ClusterRuntimeError.workerLost` to the protocol.
- Every later op after a member loss fails with the same `.memberLost`, the same way `transportFailed` sticks today (lines ~203/240).
- `errorDescription` for the new case gives plain words: which Mac was lost and that the model must reload.

Requested by the Router session for its task e0mnse8, which covers Router eviction plus local fallback and hardware assertions for xsnhzft steps 3 and 5. FoundationModelsCluster needs a follow-up to conform `workerLost`.

## Acceptance
- [ ] A transport that throws a `ClusterMemberLoss` error makes the op fail with `ClusterModelError.memberLost`, and every later op fails the same way
- [ ] Other errors still map to `transportFailed`
- [ ] Unit tests in MLXLMTests with a fake transport cover both cases
- [ ] Cluster and Router sessions told the new case and protocol names
#cluster #cross-repo