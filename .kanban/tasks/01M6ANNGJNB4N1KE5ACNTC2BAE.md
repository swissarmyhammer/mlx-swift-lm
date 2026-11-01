---
assignees:
- claude-code
position_column: todo
position_ordinal: '9380'
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

- [ ] Decide with the user whether guided decode should speculate on the router path, and how: re-prime as the iterator does, or give the guided prefill a segment.
- [ ] At the least, record a fallback when a configured drafter does not start, so the response metadata tells the truth.
- [ ] Show the guided output is the plain output on the router path, as ^84b33jx's fixtures show it on the text path.

#mtp #foundation-models