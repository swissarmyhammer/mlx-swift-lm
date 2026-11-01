---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6cbmvt9fznxyx9vzz7gmnbp
  text: |-
    ### review — findings
    - evidence: `review sha 4fe2dd46~1..4fe2dd46`: 1 finding, 1 confirmed, 8 refuted; Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:145. README.md matched no validator. The Swift hygiene rules could not read the two new test files, because they are not in the main checkout.
    - hand check: finalSegment(of:) is the iterator's former expression unchanged. Every non-speculating path reports: incompatibleDrafter (incompatible, non-guided, or shared-KV drafter), unsupportedCache (no in-place verify at start), unavailableContinuation (no feed, no segment or hidden, block size, primedState nil, verify lost mid-generation, missing drafter state). Metadata keys unchanged: SpeculativeDecodingOutcome.swift and SpeculativeDecoding.swift are not in the commit. README claims match both logs: drafted > 0 and no fallbacks on both load paths.
    - next: resolve the finding, then re-review.
  timestamp: 2026-11-01T20:08:37.449608+00:00
- actor: claude-code
  id: 01m6cc4qe081haxf3t3s1sjx5y
  text: |-
    ### review — clean
    - evidence: `review sha 3cd05485~1..3cd05485`: 0 findings, 0 confirmed, 1 refuted, 7 attempted, 0 failed, 2 files reviewed, 0 not reviewed. Prior item Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:145 ticked: 3cd05485 routes all five reason builders through fallbackReason. The near-tie divergence is decided and accepted.
    - next: 3cd05485 is on branch wt-ntc2bae only, not on mtp; merge it.
  timestamp: 2026-11-01T20:17:17.248220+00:00
position_column: done
position_ordinal: ffffa080
title: 'Guided MTP does not speculate on the router path: the VLM prefill gives no prompt segment, so the generation decodes plainly and reports no fallback'
---
Found in ^84b33jx while measuring guided decode. Not fixed there.

A schema turn on `Qwen38FourBitMTPModel` with the MTP head, Release, 8192 tokens of context, 80-item schema. Instrumented run, scratchpad 84g-guided-cal2.log:

| load path | model | drafter | prefill segment | hidden | tokens | tok/s | drafted | accepted | fallback reason |
|---|---|---|---|---|---|---|---|---|---|
| router | `Qwen35` (MLXVLM) | yes | nil | yes | 7831 | 50.58 | 0 | 0 | none |
| text | `Qwen35Model` (MLXLLM) | yes | yes | yes | 7592 | 92.46 | 5186 | 4889 | none |

The router path is the load path of the FoundationModels router. There `GuidedTarget.prefill` takes the `.logits` result of `prepare`, so `GuidedPrefill.segmentTokens` is nil. `GuidedSpeculation.init?` returns nil on that, and `GuidedGenerationLoop.run` falls to `decodePlainly`. Nothing reaches the diagnostic sink, so the response metadata reports no fallback and a host cannot tell the turn did not speculate. `MTPSpeculativeTokenIterator` handles the same `.logits` case with one more forward that re-primes the drafter (its `prepare`), and guided decode has no such step.

The figures in the MLXFoundationModels README and the `using` article ("guided JSON at 68.2 against 50.8") come from the text path.

- [x] Decide with the user whether guided decode should speculate on the router path, and how: re-prime as the iterator does, or give the guided prefill a segment.
- [x] At the least, record a fallback when a configured drafter does not start, so the response metadata tells the truth.
- [x] Show the guided output is the plain output on the router path, as ^84b33jx's fixtures show it on the text path.

#mtp #foundation-models

## Results (commit 4fe2dd46, branch wt-ntc2bae)

Router path, Release, 8k context, 60-item schema, indicative rates (ntc-gpu-both-paths.unified.log):

| run | tokens | tok/s | drafted | accepted | fallbacks |
|---|---|---:|---:|---:|---|
| plain | 4912 | 44.2 | | | |
| MTP | 4304 | 74.2 | 2951 | 2889 | none |

Decided: real-model guided output matching plain up to a near tie is accepted, the same standard as the existing benchmark gate, so router divergence at sampled index 40 with margin 0.0 passes.

## Review Findings (2026-11-01 14:01)

> Scope: `review sha 4fe2dd46~1..4fe2dd46` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 1 not reviewed.

> 1 file(s) not reviewed — no validator matched:
> - `Libraries/MLXFoundationModels/README.md` — no validator matches this file

> ⚠️ tool rule 'code-hygiene/disallowed-constructs-swift' declined an item — it judged the rest of the code, and this it could not judge:
> disallowed-constructs-swift found no file at IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGuidedRouterPathTests.swift, so its constructs are unread

> ⚠️ tool rule 'code-hygiene/disallowed-constructs-swift' declined an item — it judged the rest of the code, and this it could not judge:
> disallowed-constructs-swift found no file at Tests/MLXFoundationModelsTests/GuidedGenerationMTPRouterPathTests.swift, so its constructs are unread

> ⚠️ tool rule 'code-hygiene/function-length-swift' declined an item — it judged the rest of the code, and this it could not judge:
> function-length-swift found no file at IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGuidedRouterPathTests.swift, so its bodies are unread

> ⚠️ tool rule 'code-hygiene/function-length-swift' declined an item — it judged the rest of the code, and this it could not judge:
> function-length-swift found no file at Tests/MLXFoundationModelsTests/GuidedGenerationMTPRouterPathTests.swift, so its bodies are unread

> ⚠️ tool rule 'code-hygiene/idioms-swift' declined an item — it judged the rest of the code, and this it could not judge:
> idioms-swift found no file at IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGuidedRouterPathTests.swift, so its declarations are unread

> ⚠️ tool rule 'code-hygiene/idioms-swift' declined an item — it judged the rest of the code, and this it could not judge:
> idioms-swift found no file at Tests/MLXFoundationModelsTests/GuidedGenerationMTPRouterPathTests.swift, so its declarations are unread

> ⚠️ tool rule 'code-hygiene/magic-numbers-swift' declined an item — it judged the rest of the code, and this it could not judge:
> magic-numbers-swift found no file at IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGuidedRouterPathTests.swift, so its literals are unread

> ⚠️ tool rule 'code-hygiene/magic-numbers-swift' declined an item — it judged the rest of the code, and this it could not judge:
> magic-numbers-swift found no file at Tests/MLXFoundationModelsTests/GuidedGenerationMTPRouterPathTests.swift, so its literals are unread

> ⚠️ tool rule 'code-hygiene/missing-docs-swift' declined an item — it judged the rest of the code, and this it could not judge:
> missing-docs-swift found no file at IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPGuidedRouterPathTests.swift, so its declarations are unread

> ⚠️ tool rule 'code-hygiene/missing-docs-swift' declined an item — it judged the rest of the code, and this it could not judge:
> missing-docs-swift found no file at Tests/MLXFoundationModelsTests/GuidedGenerationMTPRouterPathTests.swift, so its declarations are unread

- [x] `Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:145` `reuse/reuse` — unprimedDrafterReason repeats the template of unguidedDrafterReason: both build "the MTP drafter \(type(of: drafter)) ... so the generation decodes without speculation" from the drafter's type. Two near-identical builders for one message family will drift apart. Give the two a shared helper that takes the drafter and the distinguishing clause, e.g. `private static func drafterCannotSpeculateReason(of drafter: any MTPDrafterModel, _ clause: String) -> String`, and keep each call site as one line. This is the same family as the earlier lostInPlaceVerifyReason finding; a single shared template would cover both.