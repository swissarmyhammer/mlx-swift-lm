---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8h0vzhc5pxp2c6d0ff151hz
  text: 'Not necessary: the brief read bf85cc03. Commits that came after it fixed both problems. 6468c3f0 (^rrkhywq): `reportingClusterFailure` throws the failure from `ClusterLanguageModel.checkFailure()` after the generation, for respond and streamResponse (ClusterExecutorTests). 4fc41044 (^vwwf7jm): an error that conforms to `ClusterMemberLoss` gives `ClusterModelError.memberLost(member)`. Each other transport error is still `transportFailed(String)`. The original error value is not kept. Router ^xsnhzft is removed (Router f6d800a1). Closed with no code change.'
  timestamp: 2026-11-28T12:07:52.108793+00:00
position_column: done
position_ordinal: ffffff9780
title: A failed cluster forward must throw its error, not end the answer as normal (^xsnhzft)
---
Brief: /Users/wballard/github/swissarmyhammer/.briefs/mlx-swift-lm-cluster-failure-ends-normally-2026-11-26.md. FoundationModelsRouter card ^xsnhzft is blocked on it.

## Problem
`ClusterLanguageModel.forward` records a transport failure, cancels the generation task and gives zero logits. The generation loop ends with `.cancelled` and no error, and `MLXLanguageModel.respond` returns success. Also `RemoteModelSession.clusterError(_:)` wraps the original error in `ClusterModelError.transportFailed(String)`, thus the caller cannot match `ClusterRuntimeError.workerLost`.

## Plan
- Keep the first original error in `RemoteModelSession`; give it as `ClusterLanguageModel.failure: (any Error)?`.
- Add a public protocol `GenerationFailureReporting { var generationFailure: (any Error)? { get } }` in MLXLMCommon. `ClusterLanguageModel` conforms.
- In `MLXLanguageModel.respond(...)`: after `dispatchGeneration` and in the `CancellationError` branch (when the caller task is not cancelled), throw the failure of the model.

## Tests (first)
- Fake transport throws `E` after N forwards: `streamResponse` and `respond` throw `E` after some text.
- A cancel of the caller task still gives `CancellationError`.