---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smznwhypm896qzvtvssbqy
  text: 'Change after the PR 581 port (commit 843854ec): the iterator `state` already returns the state without the MTP keys (`persistentTargetState`). What remains: the stop-token cache accounting, the passthrough order inside the iterator, `state:` on the public `generate` overloads, and log probabilities.'
  timestamp: 2026-10-25T13:46:14.801203+00:00
- actor: claude-code
  id: 01m5swy825zkfyaew76ffc1kj0
  text: 'Scope added from the removed task cbrddm5 (PR 581 did the rest of it): the public `generate` / `generateTokens(...mtpDrafter:)` overloads (Libraries/MLXLMCommon/Evaluate.swift ~2690-2790) take an optional `MTPDrafterContinuation` and give the final one back, so that callers other than ChatSession (the FoundationModels executor) can continue a drafter. Acceptance: two calls where the second continues the main cache and the continuation of the first give the same greedy tokens as one cold call and as plain decode; add tests for zero rounds and for cancellation in Tests/MLXLMTests/Qwen35MTPTests.swift.'
  timestamp: 2026-10-25T16:05:16.485953+00:00
- actor: claude-code
  id: 01m5zpdjsje0q1h7kjz96e7m37
  text: |-
    Research (picked up on mtp @ 2bfa5601):
    - Stop token: `generateLoopTask` sets `stopTokenFedToCache = token` for every suppressed stop token. True for `TokenIterator` (its `next()` steps the returned token before returning it). For `MTPSpeculativeTokenIterator` the returned token is in the main cache only when it is an accepted draft (`pendingIndex - 1 < committedPendingTokenCount`); the prepare bonus, the correction/bonus of a round, and every passthrough token are not fed. `SpeculativeTokenIterator` (draft model) has the same shape (`mainCommittedPendingTokenCount`) and the same wrong report; out of scope here, filed as a follow-up.
    - ChatSession does not read `stopTokenFedToCache`; it reconciles its ledger against `KVCacheStorage.processedTokenCount`, so the ^chgkh4c observation (MTP prefills the reply's final token next turn, plain already ran it) is a consistent ledger, not a count bug. Only the executor ledger (`MLXLanguageModel.generatedTokenCount(of:)`) reads the field, and it does not run MTP yet.
    - Passthrough order: `requiresGreedySampling` and the distribution-drafter check run after `prepare`, which by then ran the unchunked emit-flag prefill and `prepareDrafterState`.
    - `MTPDrafterContinuation` is `package`; the public overloads can only take it if it becomes public. `MLXArray` is not Sendable, so giving it back through a `Task` value needs the struct to be `Sendable`.
    - Log probabilities: only `TokenIterator` reports them; ChatSession MTP + logprobs currently runs MTP and reports none.
  timestamp: 2026-10-27T22:06:45.554907+00:00
- actor: claude-code
  id: 01m5zqvzvr585cv94d907hgf0n
  text: |-
    Implementation landed (uncommitted, on mtp):
    - Stop token: new public `TokenIteratorProtocol.hasCachedLastToken` (default `true`, so `TokenIterator` is unchanged). `MTPSpeculativeTokenIterator` sets it per returned token: `pendingIndex < committedPendingTokenCount` for pending tokens, `false` for plain-decode tokens. `generateLoopTask` sets `stopTokenFedToCache` only when it is `true`. Result: an MTP stop on a correction or bonus slot now reports `stopTokenFedToCache == nil`, a stop on an accepted draft reports the token, and prompt + generationTokenCount + fed equals the main cache count in all three (test `stopTokenReportsCacheState`). ChatSession is unchanged: it already reconciles against `processedTokenCount`, so its MTP turns still leave the reply's final token for the next turn's prefill (plain has run it); no contract requires that row to match plain, and closing it would cost one extra forward per turn — not done here.
    - Passthrough order: `standDownReason` (greedy-only drafter at temperature != 0, token-only drafter under a distribution sampler, logprobs requested) is decided in `init` before `prepare`. A stream that stands down gets no drafter state, no continuation, no cache staging probe, a normal chunked prefill without the emit flag, no priming forward, and no `prepareDrafterState`/cold start. First matching reason wins (the old code let the later switch overwrite `passthroughReason`).
    - Log probabilities: chose plain decode. `logProbabilitiesReason` ("logprobs requested; ..."); `lastLogProbabilities` is reported for every token through a shared `sampleToken(from:)` helper that also replaced four copies of the process/sample/didSample sequence. `LogProbabilityDistribution` widened from private to internal for reuse.
    - state + continuation: `MTPDrafterContinuation` is now `public` (members stay `package`, so it is opaque) and `@unchecked Sendable` with a swiftlint directive stating the invariant. The MTP public init takes `mainState:` and `mtpContinuation:`. `generate(...mtpDrafter:)` / `generateTokens(...mtpDrafter:)` take `state:` and `mtpContinuation:`; new `generateTask(...mtpDrafter:...)` / `generateTokensTask(...mtpDrafter:...)` return `Task<MTPDrafterContinuation?, Never>` which gives the final continuation back (the stream-only overloads cannot return it without a source-breaking return-type change). Tests: two calls == one cold call == plain for a complete first call, a zero-round first call, and a cancelled first call.
    - Follow-up filed: ^8y0atxm (`SpeculativeTokenIterator` has the same stop-token misreport).
    - Release benchmarks (M5 Ultra): qwen38FourBitMTPGeneration run 1 89.9/80.1/78.5, run 2 90.6/81.1/77.2 tok/s (baseline ~90.5/81.4/79.2; acceptance 93.9/73.6/69.0% unchanged, all match plain). qwen38FourBitMTPSystemPromptCache passes (2 known issues, the existing ^achvxy1/^ahnb0j9 edit-turn gaps). Per-turn: follow-ups MTP t1 cached 2403/prefilled 38 (plain 2404/37), t2 2544/29 (plain 2544/29); thinking t0 0/2191, t1 2302/25 (plain 2303/24), t2 2423/25 (plain 2425/24); edit t0 0/2197, t1 2264/26 (plain 2265/25), t2 (edit) 0/2292 (plain 2169/123, known gap).
  timestamp: 2026-10-27T22:32:06.264357+00:00
- actor: claude-code
  id: 01m5zqw4cee5ny5vf2y0cekmt6
  text: |-
    ### implement — changed
    - evidence: 8 files — Libraries/MLXLMCommon/Evaluate.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/BenchmarkHelpers/MTPChatTokenMatch.swift, Tests/MLXLMTestSupport/Qwen35MTPFixture.swift, Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift. `swift build --build-tests` clean (no new warnings); MLXLMTests 895 XCTest + 1854 Swift Testing, 0 failures (1 known env-gated skip); MLXFoundationModelsTests 376 passed; Release qwen38FourBitMTPGeneration x2 and qwen38FourBitMTPSystemPromptCache pass.
    - next: /review
  timestamp: 2026-10-27T22:32:10.894186+00:00
- actor: claude-code
  id: 01m5zqwdb16ta8sg5ee0v223bn
  text: |-
    ### implement — changed
    - evidence: 7 files (correction of the count above; the ChatSessionMTPFakes edit was reverted, the sampled-stream test reads `probe.coldStartPositions` instead) — Libraries/MLXLMCommon/Evaluate.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXLMCommon/MTPDrafterModel.swift, Libraries/BenchmarkHelpers/MTPChatTokenMatch.swift, Tests/MLXLMTestSupport/Qwen35MTPFixture.swift, Tests/MLXLMTests/Qwen35MTPTests.swift, Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift. MLXLMTests 895 XCTest + 1854 Swift Testing 0 failures; MLXFoundationModelsTests 376 passed.
    - next: /review
  timestamp: 2026-10-27T22:32:20.065256+00:00
- actor: claude-code
  id: 01m5zr4tzv0fs7c5n02vad662n
  text: |-
    ### test — green
    - evidence: swift build --build-tests (changed files touched, MLXLMCommon/BenchmarkHelpers recompiled) — 0 new warnings, only the known "missing creator ... mlx-swift_Cmlx.bundle" message; xcrun xctest MLXLMTests — 895 XCTest, 0 failures, plus 1854 Swift Testing passed; MLXFoundationModelsTests — 376 passed, 0 failures; only skip is the env-gated benchmarkVerification()
    - next: review
  timestamp: 2026-10-27T22:36:56.187533+00:00
depends_on:
- 01M5SK8FDMF5XH4FDH12H0FPX1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '8180'
title: 'MTP: fix the MTP iterator contract with the shared generate loop (stop token, passthrough order, state, logprobs)'
---
## What
The FoundationModels executor and ChatSession use the shared loop `generateLoopTask` (Libraries/MLXLMCommon/Evaluate.swift ~2946). It accepts `any TokenIteratorProtocol`, so it can run `MTPSpeculativeTokenIterator` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift). Research found four defects that make MTP incorrect or wasteful in that loop:

1. **Stop-token cache accounting.** After a stop token, the loop sets `stopTokenFedToCache` (Evaluate.swift ~3033), and the executor ledger counts the stop token as cached (Libraries/MLXFoundationModels/MLXLanguageModel.swift ~607-617). With MTP, the last token of a round (the correction or bonus token) is not in the cache (`committedPendingTokenCount = accepted`, MTPSpeculativeTokenIterator.swift ~566). A stop on that slot is then reported as "fed" when it was not, and the cache ledger is wrong by one.
2. **Passthrough order.** The `requiresGreedySampling` check runs after `prepare` (MTPSpeculativeTokenIterator.swift ~213-218). Thus a sampled stream first runs the full unchunked drafter prompt prefill, and then changes to plain decode.
3. **No `state:` input.** The executor passes `plan?.state` and reads `preparedState(iterator.state)` (Evaluate.swift ~2511, ~2849). The MTP `generate` overloads (~2690-2790) take no `state:` and expose no prepared state.
4. **Log probabilities.** Only `TokenIterator` reports them (Evaluate.swift ~214).

- [x] Add a property to `TokenIteratorProtocol` (or to the MTP iterator, read through a protocol cast in `generateLoopTask`) that tells if the last returned token is in the main cache. Use it to set `stopTokenFedToCache`, in place of the fixed assumption. `TokenIterator` keeps its current behaviour.
- [x] Move all passthrough checks that do not need the prompt (sampler, drafter compatibility, cache kinds) before `prepare`, so that a stream that cannot use MTP does no drafter work.
- [x] Add `state: LMOutput.State?` to the MTP `generate` / `generateTokens` overloads and expose the prepared state, as `TokenIterator` does. The MTP keys (`mtpEmitFlagKey`, `mtpSharedKVStatesKey`, `mtpLastHiddenStatesKey`, `mtpCacheCheckpointIndexKey`) must not stay in the exported state, because a non-MLXArray value makes the prompt-cache spill throw (Libraries/MLXLMCommon/LanguageModel.swift ~275-285).
- [x] When the caller asks for log probabilities, use plain decode with `passthroughReason` "logprobs", or report them from the verify logits of the accepted tokens. Choose one and document it.

## Acceptance Criteria
- [x] A stream that stops on a correction token and a stream that stops on a bonus token both report the true cache state, and the main cache offset after `finalizeGeneration` equals the ledger count that the loop reports.
- [x] A sampled stream with a drafter that requires greedy sampling does not call `prepareDrafterState`.
- [x] The exported state of an MTP stream contains only MLXArray values and no MTP keys.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPTests.swift: stop on the correction slot and on the bonus slot (scripted stop token), with a check of the cache offset against the reported ledger.
- [x] Tests/MLXLMTests: a test with a counting drafter that asserts no drafter prefill on a sampled stream.
- [x] Tests/MLXLMTests: a test that `savePromptCache` accepts the exported MTP state.
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #foundation-models