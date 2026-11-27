---
position_column: todo
position_ordinal: '9080'
title: 'Clustered models: a generation whose forward fails returns its partial text instead of throwing the failure'
---
## What
FoundationModelsRouter's 2-Mac test lostMemberFallsBackToThisMac (2026-11-27, Qwen3.8-27B-mxfp4 split over m5u1+m5u2) kills the worker on the member mid-answer. The streamed answer ends 1.17 s after the kill with no error, and its text is "\n". The Router then can't tell the answer failed, so it never moves the session to a local copy (^hvygt9p). The next request of another session throws `ClusterModelError.memberLost` from `beginRequest`, and that session does fail over and answers on the local copy.

## Cause
`ClusterLanguageModel.forward` (Libraries/MLXLMCommon/Remote/ClusterLanguageModel.swift:204) catches the failure, records it (`session.recordFailure`), cancels the current task and returns zero logits. The decode loop then ends as if the generation had finished, and MLXLanguageModel returns the partial response. Nothing reads the recorded failure when the generation ends: `RemoteModelSession.hasFailed` is read only by `RemoteKVCache` at deinit, and the failure surfaces only at the next `beginRequest`.

## Fix
When a generation ends and its ClusterLanguageModel session has a recorded failure, the respond/stream of MLXLanguageModel throws that failure (`ClusterModelError.memberLost` for a lost member) and doesn't return the partial text. A stream throws after the snapshots it already gave. For example, expose the recorded failure on ClusterLanguageModel (a throwing `checkFailure()`), and call it in the executor after the generation loop ends and before the response completes.

Tests: a unit test with a stub transport that fails the Nth forward: `respond` throws `memberLost`, and `streamResponse` throws after its earlier snapshots. Then FoundationModelsRouter IntegrationTests lostMemberFallsBackToThisMac passes on the 2-Mac setup.

Router side: FoundationModelsRouter ^hvygt9p (ce284eab) already retries a submission that throws a ClusterModelError on a copy on this Mac.
Evidence: FoundationModelsRouter run 2026-11-27 11:35 on m5u1, output "E2E lost member: the answer ended 1.173709875 seconds after the kill" / "E2E lost member answer: \"\\n\"". #cluster