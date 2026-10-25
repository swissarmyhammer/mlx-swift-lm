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
depends_on:
- 01M5SK8FDMF5XH4FDH12H0FPX1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: 8d80
title: 'MTP: fix the MTP iterator contract with the shared generate loop (stop token, passthrough order, state, logprobs)'
---
## What
The FoundationModels executor and ChatSession use the shared loop `generateLoopTask` (Libraries/MLXLMCommon/Evaluate.swift ~2946). It accepts `any TokenIteratorProtocol`, so it can run `MTPSpeculativeTokenIterator` (Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift). Research found four defects that make MTP incorrect or wasteful in that loop:

1. **Stop-token cache accounting.** After a stop token, the loop sets `stopTokenFedToCache` (Evaluate.swift ~3033), and the executor ledger counts the stop token as cached (Libraries/MLXFoundationModels/MLXLanguageModel.swift ~607-617). With MTP, the last token of a round (the correction or bonus token) is not in the cache (`committedPendingTokenCount = accepted`, MTPSpeculativeTokenIterator.swift ~566). A stop on that slot is then reported as "fed" when it was not, and the cache ledger is wrong by one.
2. **Passthrough order.** The `requiresGreedySampling` check runs after `prepare` (MTPSpeculativeTokenIterator.swift ~213-218). Thus a sampled stream first runs the full unchunked drafter prompt prefill, and then changes to plain decode.
3. **No `state:` input.** The executor passes `plan?.state` and reads `preparedState(iterator.state)` (Evaluate.swift ~2511, ~2849). The MTP `generate` overloads (~2690-2790) take no `state:` and expose no prepared state.
4. **Log probabilities.** Only `TokenIterator` reports them (Evaluate.swift ~214).

- [ ] Add a property to `TokenIteratorProtocol` (or to the MTP iterator, read through a protocol cast in `generateLoopTask`) that tells if the last returned token is in the main cache. Use it to set `stopTokenFedToCache`, in place of the fixed assumption. `TokenIterator` keeps its current behaviour.
- [ ] Move all passthrough checks that do not need the prompt (sampler, drafter compatibility, cache kinds) before `prepare`, so that a stream that cannot use MTP does no drafter work.
- [ ] Add `state: LMOutput.State?` to the MTP `generate` / `generateTokens` overloads and expose the prepared state, as `TokenIterator` does. The MTP keys (`mtpEmitFlagKey`, `mtpSharedKVStatesKey`, `mtpLastHiddenStatesKey`, `mtpCacheCheckpointIndexKey`) must not stay in the exported state, because a non-MLXArray value makes the prompt-cache spill throw (Libraries/MLXLMCommon/LanguageModel.swift ~275-285).
- [ ] When the caller asks for log probabilities, use plain decode with `passthroughReason` "logprobs", or report them from the verify logits of the accepted tokens. Choose one and document it.

## Acceptance Criteria
- [ ] A stream that stops on a correction token and a stream that stops on a bonus token both report the true cache state, and the main cache offset after `finalizeGeneration` equals the ledger count that the loop reports.
- [ ] A sampled stream with a drafter that requires greedy sampling does not call `prepareDrafterState`.
- [ ] The exported state of an MTP stream contains only MLXArray values and no MTP keys.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: stop on the correction slot and on the bonus slot (scripted stop token), with a check of the cache offset against the reported ledger.
- [ ] Tests/MLXLMTests: a test with a counting drafter that asserts no drafter prefill on a sampled stream.
- [ ] Tests/MLXLMTests: a test that `savePromptCache` accepts the exported MTP state.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #foundation-models