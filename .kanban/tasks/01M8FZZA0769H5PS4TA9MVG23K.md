---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g2p1cnedt4emk597er8ver
  text: |-
    Research done (no code changed). The coordinator said to keep this small, leave the plain cluster MTP path alone, add no round trips to plain decoding, and stop and report if the change gets large. It gets large, so I stopped.

    Why plain conformance is not enough. The local guided MTP path sends target hidden rows through the host at every step:
    - The prefill feeds the drafter through `MTPDrafterPrefillFeed.chunkHandler` with each chunk's `mtpLastHiddenStatesKey` rows.
    - `GuidedSpeculation.init` needs `prefill.segmentHidden`, which `primedState` uses.
    - `commitDrafter` needs `output.state?[mtpLastHiddenStatesKey]` from each verify, and stands down without it.
    - `lastKeptHidden` is passed into each `draftBlockWithLogits`.

    `ClusterLanguageModel` returns no hidden rows: rank 0 keeps them inside `RemoteDrafterTrack`. Two more gaps:
    - `GuidedTarget.verifiesInPlace` is false for a cluster target. `ClusterLanguageModel` is not a `SpeculativeCacheRewindModel`, and a `RemoteKVCache` has no non-attention leaf.
    - `keepVerified` calls `rewindSpeculativePromptCache`, which calls `RemoteKVCache.trim`. That costs one extra round trip on every rejected round.

    `draftBlockWithLogits` also gets no cache handle, so a remote drafter cannot tell which `RemoteKVCache` to draft on.

    What a one-round-trip-per-round design needs (estimated 800–1500 lines including tests):
    1. Wire, version 6:
       - Add a flag to `RemoteDraftRound`: return the draft logits. Rank 0 drafts unmasked with `draftBlockWithLogits`.
       - `.drafted` carries `draftLogits` for each draft step.
       - Add the flag word to the speculate announce.
       - `RemoteDrafting` must hold a `StatefulMTPDistributionDrafterModel`.
    2. `RemoteMTPDrafter` conforms to both protocols:
       - The state methods are no-ops that report success, because the rank-0 track follows the cache.
       - `draftBlockWithLogits` runs one speculate with draft logits. It puts each step through the host grammar processor and reports drafts only while the masked argmax equals the rank-0 draft. Every reported draft is then inside the verify rank 0 already ran.
       - It keeps the verify (tokens and logits) on the `RemoteKVCache`.
       - It needs a way to reach the cache, for example through `MTPDrafterState.cache` or the session.
    3. `RemoteKVCache` / `ClusterLanguageModel`:
       - Answer a verify whose tokens are a prefix of the kept verify from the kept logits, with no round trip. Otherwise trim the extra positions and forward the rest.
       - Defer rewinds into the next speculate's `rejectedCount`, and flush them as a trim before any other op.
       - Report placeholder hidden rows under the MTP emit flag, so the generic guided code does not stand down. The rows are never read on the host.
       - Make cluster `verifiesInPlace` / `mostVerifiedTokens` true.
    4. Guided code: thread the cluster cases through `GuidedTarget` (prefill segment hidden, verify, keepVerified) without changing the local path.
    5. Tests, and a FoundationModelsCluster card for the v6 wire change.

    This touches the guided core, the wire and announce encoding, the executor, `RemoteKVCache` and the drafter: effectively a new subsystem, which the scope message rules out.

    A smaller option (needs your approval, because it does not make `RemoteMTPDrafter` conform as the card orders): on a cluster target, the guided loop runs the existing `speculate` op unchanged, with no new wire fields and no version bump. It accepts on the host by walking the grammar along `[token] + drafts`: each forced token must equal the verified token, and each decided token must equal the constrained argmax of its row. It folds rejects into the next round's `rejectedCount`, as `ClusterMTPTokenIterator` does, and ends with a trim. That is one round trip per round, no plain-path change, and about 300–400 lines plus tests. The cost is that drafts are unmasked: a step the grammar masks ends acceptance at that row, and the target still decides that token, so the output stays exact.
  timestamp: 2026-11-28T03:20:20.117436+00:00
- actor: claude-code
  id: 01m8g2p47ygh38476t1tjvbsef
  text: |-
    ### implement — stuck
    - evidence: no files changed. The card's design (`RemoteMTPDrafter` conforms to `ResumableMTPDrafterModel` and `StatefulMTPDistributionDrafterModel`) needs host-side hidden rows, a cluster in-place verify, deferred rewinds, and wire v6. That is a large change, which the coordinator's scope rules out. The previous comment gives the full analysis and a smaller option.
    - next: a person decides between (a) the full conformance design (wire v6, about 800–1500 lines) and (b) a cluster guided round over the existing `speculate` op (no wire change, about 300–400 lines, drafts unmasked, output exact). Either one changes the card.
  timestamp: 2026-11-28T03:20:23.038514+00:00
- actor: claude-code
  id: 01m8g2pqmznty2wvdcqv3hxn9n
  text: |-
    ### finish iteration 1 — stuck
    - implement: stuck — no files changed. Making RemoteMTPDrafter conform, as the card says, needs host-side hidden rows, a cluster in-place verify, deferred rewinds and wire v6, about 800–1500 lines (Option A). That is against the scope set on 2026-11-27: keep it small and don't slow plain decoding. Option B uses the existing `speculate` op with no protocol change, about 300–400 lines. Rank 0 drafts unmasked, and the host accepts drafts by walking the grammar along the verified tokens. Output is exact; acceptance may be lower.
    - test/commit/review: not run
    - Blocker: A or B is a spec decision for the requester or the user. Asked foundationmodelsacpagent-a1 and foundationmodelsrouter-22; recommended B.
  timestamp: 2026-11-28T03:20:42.911252+00:00
- actor: claude-code
  id: 01m8g2qeap1g1y53es8ycka59r
  text: |-
    ### Decision: Option B (2026-11-27)
    foundationmodelsacpagent-a1 and foundationmodelsrouter-22 both chose B. This replaces the description's "RemoteMTPDrafter conforms to ResumableMTPDrafterModel and StatefulMTPDistributionDrafterModel".

    **Approach**
    - On a cluster target, the guided loop uses the existing `speculate` op unchanged. There is no protocol change (version stays 5), and nothing changes on the plain path or in FoundationModelsCluster.
    - Rank 0 drafts without the grammar mask. The host accepts drafts by walking the grammar along the verified tokens, and a rejected draft is folded into the next round.
    - Output must equal plain guided decoding exactly.

    **Rejected: the low-acceptance guard.** foundationmodelsacpagent-a1 also asked for a runtime guard: switch to plain decoding when acceptance falls below break-even, with fallback `lowAcceptance`. The user rejected adaptive MTP guards on 2026-10-30 ("I don't want that guard at all"), so it is not built. If acceptance is low, the fix is a cheaper round.

    **Tests (TDD)**
    - A guided schema turn on a clustered model with a drafter speculates (draft count > 0), and its output equals plain guided decoding on the cluster and on one Mac.
    - Acceptance counts are reported, and `speculated=true` is recorded.
    - Every rank's cache offset equals the ledger.
    - A second turn reuses the cache.
    - kvBits 8 works.
    - A lost member throws `memberLost` in under 10 s.

    **Measure on m5u1+m5u2 with the bench selection shape**
    - Inkling oQ4e mtp target, a prompt of about 8k tokens, the Selection JSON schema.
    - Report acceptance and tok/s (decode and end-to-end) against plain guided decoding on the cluster.
  timestamp: 2026-11-28T03:21:06.134429+00:00
- actor: claude-code
  id: 01m8g307tpn0t58wtg93kreemn
  text: |-
    Picked up for Option B. Research notes:
    - The guided loop keeps no prompt cache (runSchemaGeneration calls `promptCache.carriesNoCache()`), so "cache offsets equal ledger" means the worker cache (every rank) ends at prompt + fed tokens, as plain guided leaves it; "second turn reuses the cache" is checked as a plain session after a speculative schema turn.
    - Walk design: the host keeps a queue of known-next tokens (the decision's token + its forced tokens). Each round pops the queue head as the `speculate` round token. Each draft is accepted while it equals the queue head; when the queue empties, the host decides the next token from the verify row (the same `decoder.advance()` + `decoder.decide` as plain decoding). So no grammar matcher rollback is needed: every decision is plain guided decoding's decision, read from a verify row. Rejected drafts fold into the next round's `rejectedCount`; the last round's rejects are trimmed at the end.
    - A forced suffix the drafts missed is fed as the round token of the next round(s), so a round always moves at least one token: worst case one round trip per token, the cost of plain decoding.
    - `InProcessRemoteModelWorker.failForwards` counts `.forward` ops only, so a loss during speculative rounds needs a round-counting fault in the test worker.
    - ClusterLanguageModel.speculate / remoteCache(of:) are internal to MLXLMCommon; MLXGuidedGeneration needs a `package` entry.
  timestamp: 2026-11-28T03:25:54.390016+00:00
- actor: claude-code
  id: 01m8g4babrmmenpk7fp201vatd
  text: |-
    Option B implemented (not committed). On a cluster target the guided loop runs the existing `speculate` op unchanged: protocol stays v5, and the worker, FoundationModelsCluster, the plain cluster MTP path and plain decoding are unchanged.

    Code:
    - New `Libraries/MLXGuidedGeneration/GuidedClusterSpeculation.swift`. Each round is one `speculate` from the head of the queue of known tokens. A draft is kept while it equals the token plain guided decoding feeds there. When the queue empties, the host decides from that draft's verify row with the same `decoder.advance()` + `decode(...)` calls as `decodePlainly`, so no matcher rollback is needed. Rejects go into the next round's `rejectedCount`, and the last round's rejects are trimmed at the end.
    - `GuidedSpeculation.drafter(of:for:)` now returns `GuidedDrafter` (`.local` or `.cluster`). A `RemoteMTPDrafter` on its own `ClusterLanguageModel` selects `.cluster`. A local drafter that is not guided still reports `.unsupportedGuidance`.
    - `GuidedTarget.speculate(on:...)` commits the ledger the way `ClusterMTPTokenIterator` does, then runs `plan.apply` after each round (^2ry6yz7 quantize-after-round). `GuidedTarget.trimSpeculated` trims and then runs `checkFailure`.
    - `ClusterLanguageModel.speculate(in:...)` is a `package` throwing wrapper. A failed round keeps its failure, and the loop throws it in that same round (^axrb0cw, inside `withoutCancellingOnFailure`).
    - No runtime guard and no lowAcceptance fallback.

    Tests:
    - New `GuidedGenerationClusterMTPTests` (loop level, 1 follower rank). Every schema speculates, and its accepted count is > 0. Output, token count and decision records equal plain guided on the cluster (both drafter modes) and on one Mac. Every rank's offset equals what plain guided leaves (checked by mutation: without the final trim, both cases fail). kvBits 8 speculates and equals plain.
    - `ClusterExecutorTests`: the unsupportedGuidance and fallback-detail tests flip to a schema turn that speculates (speculativeDecoding=true), metadata drafted/accepted equal to the sink's counts, schema output equal to the local response, and a follow-up plain turn after a speculative schema turn reusing the prompt cache (cachedPromptTokenCount > 0). A loss in a speculative round (new `InProcessRemoteModelWorker.failRounds`) throws memberLost in under 10 s.
    - `InProcessRemoteModelWorker` gains `rankStatuses(of:)` and `failRounds(after:with:)`. `GuidedFixtureRun.generate` gains `kvCache`.

    Real 2-Mac measurement (m5u1+m5u2, Inkling-2-Mini-oQ4e-mtp, MTP configured, policy .always, worker built at FoundationModelsCluster da091cc, v5):
    - Harness: a scratchpad package `guided-bench` with the SelectionTier shape: 7,833-token instructions of `<candidate>` blocks plus a 794-token prompt (about 8.6k tokens), `@Generable Selection { ids: [String] }`, greedy. Calls ran through `LanguageModelSession.streamResponse`; 10 timed calls followed 1 warm-up.
    - Baseline: the same harness built against stable HEAD 448e8643 (guided falls back with `unsupportedGuidance` and decodes plainly), run before and after B.
    - Outputs: 16–41 tokens per call. ids and token counts are identical in all three runs (md5 of per-call ids and counts equal).

    | | e2e mean | decode tok/s | drafts accepted |
    |---|---|---|---|
    | plain guided (HEAD), run 1 | 3.451 s | 91.4 | none drafted |
    | plain guided (HEAD), run 2 | 3.440 s | 92.4 | none drafted |
    | guided with B | 3.292 s | 155.8 | 177 of 190 (93.2%) |

    - Decode is 1.70x faster with B. End to end is 4.6% faster, because the ~3.12 s prefill of the 8.6k prompt dominates these short outputs.
    - Per call, B decodes at 145–171 tok/s and plain at 84–112 tok/s.
  timestamp: 2026-11-28T03:49:26.008284+00:00
- actor: claude-code
  id: 01m8g4bky4yp42rfc076araky7
  text: |-
    ### implement — changed
    - evidence: new Libraries/MLXGuidedGeneration/GuidedClusterSpeculation.swift; changed GuidedSpeculation.swift, GuidedGenerationLoop.swift, GuidedTarget.swift, MLXLMCommon/Remote/ClusterLanguageModel.swift; tests: new Tests/MLXFoundationModelsTests/GuidedGenerationClusterMTPTests.swift, changed ClusterExecutorTests.swift, GuidedGenerationInklingMTPTests.swift, GuidedGenerationMTPTestSupport.swift, MLXLMTestSupport/InProcessRemoteModelWorker.swift. All five bundles ran in full with `xcrun xctest` and 0 failures: MLXLMTests 1261 XCTest + 2845 Swift Testing (3 known issues, pre-existing withKnownIssue), MLXFoundationModelsTests 672, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11. FoundationModels + GuidedGeneration were re-run after the final edit: green. swift-format run. Real 2-Mac: guided with B decodes at 155.8 tok/s vs plain guided at 91.4/92.4 (1.70x), 93.2% of drafts accepted (177/190), e2e 3.292 s vs 3.451/3.440 s (-4.6%), and the outputs are identical.
    - next: /review. Not committed.
  timestamp: 2026-11-28T03:49:35.812595+00:00
- actor: claude-code
  id: 01m8g4wmdfymcyh9hqg4h4bd6h
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 source warnings). xcrun xctest per bundle, all exit 0: MLXLMTests 1261 XCTest + 2845 Swift Testing, 0 failures; MLXGuidedGenerationTests 96 ST; MLXFoundationModelsTests 672 ST (GuidedGenerationClusterMTPTests passed); CXGrammarTests 7 ST; MLXHuggingFaceMacrosTests 11 XCTest. 0 skipped. swift-format lint on changed files clean.
    - note: MLXLMTests reports 3 known issues, all deliberate withKnownIssue in untouched files (ToolTests MiniCPM5 CDATA, AttentionKernelToleranceTests near-tie); pre-existing.
    - next: review/commit
    - task: ^9mvg23k
  timestamp: 2026-11-28T03:58:53.359828+00:00
- actor: claude-code
  id: 01m8g53xz77bzr224vke7r1rdn
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (eae83fbd): 2 findings (2 confirmed, 1 refuted) — Tests/MLXFoundationModelsTests/ClusterExecutorTests.swift:492, Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift:207
    - next: implement the two reuse/reuse findings in the 2026-11-27 21:59 Review Findings section, then re-review
  timestamp: 2026-11-28T04:02:52.519336+00:00
- actor: claude-code
  id: 01m8g54c1k48y2j4fmpce5f611
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — option B. GuidedClusterSpeculation.swift is new, plus 4 changed library files and 5 test files. 2-Mac: 155.8 vs 91.4/92.4 tok/s decode, 93.2% acceptance.
    - test: green — all 5 bundles (MLXLMTests 2845+1261, MLXFoundationModelsTests 672, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: eae83fbd
    - review: findings — ClusterExecutorTests.swift:492, InProcessRemoteModelWorker.swift:207
  timestamp: 2026-11-28T04:03:06.931188+00:00
- actor: claude-code
  id: 01m8g5ns13kfy08m0e1tqsq3d3
  text: |-
    Both 2026-11-27 21:59 review findings are fixed, and each one is checked on the card.
    - InProcessRemoteModelWorker: a new public `FailureKind` (`ops`, `forwards`, `rounds`) and `fail(_:after:with:)` set the matching budget and `fault` in one place, through a private keypath map `budget(of:)`. `failOps`, `failForwards` and `failRounds` are now one-line wrappers. failOps had the same copied shape, so I also routed it through the shared helper, to remove the cause from the whole file.
    - ClusterExecutorTests: `respondGuidedLosingAMember` now takes `losing kind: FailureKind, after count:` and an optional `speculativeDecoding`, which also sets `drafting`. `respondSpeculativelyLosingAMember` is now one call to it with `.rounds`, `roundsBeforeLoss` and the fixture's speculative decoding. Every assertion is unchanged.

    ### implement — changed
    - evidence: 2 files — Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift, Tests/MLXFoundationModelsTests/ClusterExecutorTests.swift. Both findings checked. swift-format was run, and `swift build --build-tests` is clean. `xcrun xctest` exits 0 for both bundles: MLXFoundationModelsTests ran 672 Swift Testing tests, all passed, including the 6 member-loss tests. MLXLMTests ran 1261 XCTest tests with 0 failures and 2845 Swift Testing tests, all passed. The 3 known issues are pre-existing `withKnownIssue` cases.
    - next: /review. Not committed.
  timestamp: 2026-11-28T04:12:37.283790+00:00
- actor: claude-code
  id: 01m8g66dxy7eq6sf5jkw99ertc
  text: |-
    ### test — green
    - evidence: swift build --build-tests complete, 0 code warnings (one SwiftPM note about the Cmlx bundle); swift-format lint clean on InProcessRemoteModelWorker.swift and ClusterExecutorTests.swift. xcrun xctest: MLXLMTests 1261 XCTest 0 failures + 2845 Swift Testing passed (3 known issues, in untouched AttentionKernelToleranceTests and ToolTests MiniCPM5, both withKnownIssue markers in committed code); MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 672 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest 0 failures. 0 skipped.
    - next: review/commit
  timestamp: 2026-11-28T04:21:42.974059+00:00
depends_on:
- 01M8FZZ0C8R7PYY6JH0ARXS2A5
position_column: doing
position_ordinal: '80'
title: 'Clustered models: guided speculation with the remote MTP drafter (RemoteMTPDrafter conforms to ResumableMTPDrafterModel and StatefulMTPDistributionDrafterModel)'
---
## What
A guided call (`respond(to:generating:)` with an @Generable type) on a clustered model never speculates. GuidedSpeculation.drafter(of:for:) (Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:95) needs `ResumableMTPDrafterModel & StatefulMTPDistributionDrafterModel`, and RemoteMTPDrafter (Libraries/MLXLMCommon/Remote/RemoteMTPDrafter.swift, from ^rgzk56j) conforms to neither. On bench cluster4i (Inkling MTP on 2 Macs), 66 of 452 calls were guided SelectionTier calls from FoundationModelsRanker, and all of them decoded plainly. A plain call on the same model speculates: 61% acceptance, 94.9 tok/s. Reported by foundationmodelsrouter-22.

## Fix
- Make RemoteMTPDrafter conform to ResumableMTPDrafterModel and StatefulMTPDistributionDrafterModel over the remote protocol, so the guided loop can resume the drafter and mask the draft distribution with the grammar, as it does locally.
- This likely needs new protocol ops, or new fields on `speculate`: resume the drafter state at a position, and return the draft distribution (or the masked draft) so the grammar on the host can mask each draft step. Keep each round to one round trip where possible. Rank 0 holds the drafter, but the grammar is on the host.
- Bump RemoteModelProtocol.version (now 5) if the wire format changes, and record what FoundationModelsCluster must adopt, as a card for foundationmodelscluster-4e like ^6pq8cee.
- Every rank's cache offset must equal the ledger after a guided speculative turn, so prompt cache reuse keeps working.
- No runtime guard.

## Tests
- A guided (schema) turn on a clustered model with a drafter speculates: draft count > 0, and the greedy output equals the local guided MTP output.
- Acceptance counts are reported.
- A second turn reuses the cache.
- A lost member on a guided speculative turn still throws `memberLost` promptly (^axrb0cw).
Depends on ^arxs2a5 (the fallback reason for the interim). #cluster

## Review Findings (2026-11-27 21:59)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8FZZA0769H5PS4TA9MVG23K.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8FZZA0769H5PS4TA9MVG23K.md` — no validator matches this file

- [x] `Tests/MLXFoundationModelsTests/ClusterExecutorTests.swift:492` `reuse/reuse` — `respondSpeculativelyLosingAMember(weights:)` duplicates the body of the existing `respondGuidedLosingAMember(to:capabilities:afterForwards:weights:)` (line 470). Both build a clustered model, arm a worker to fail, start a `ContinuousClock`, await `respondGuided`, and return a `GuidedLoss`. The new helper is a near-match that was not extended by parameterizing the difference, which is the speculative setup and the round-based failure. Extend `respondGuidedLosingAMember` with a parameter that carries the failure to arm (for example a closure `(LanguageModel worker) -> Void`, or an enum of forwards/rounds), and an optional `speculativeDecoding` argument passed to `makeClusteredLanguageModel`. Then `respondSpeculativelyLosingAMember` becomes a call with those arguments, or is removed.
- [x] `Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift:207` `reuse/reuse` — `failRounds(after:with:)` is a near-copy of the existing `failForwards(after:with:)` (line 194). Both set one `remaining*` counter and the shared `fault` under `lock.withLock`, with the same doc shape. The only difference is which counter is set. A second parallel copy means a future change to the fault-arming logic has to be made in two places and can diverge. Replace both with one helper, e.g. `private func fail(_ budget: WritableKeyPath<InProcessRemoteModelWorker, Int?>, after count: Int, with fault: any Error)`, or a single `failing(_ kind: FailureKind, after:with:)` that sets the matching counter. Keep `failForwards` and `failRounds` as one-line public wrappers, or have callers use the shared entry point.
