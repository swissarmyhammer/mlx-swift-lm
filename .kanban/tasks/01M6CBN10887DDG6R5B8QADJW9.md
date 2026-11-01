---
assignees:
- claude-code
position_column: todo
position_ordinal: '9780'
title: Text-path guided MTP diverges from plain at sampled index 1691 with margin 0.125, past the 1e-3 near-tie gate
---
Found in ^ntc2bae while running `MTPGuidedRouterPathTests` on both load paths.

An 8k-context, 60-item schema turn of Qwen3.8-27B-4bit with its 4-bit MTP head, on the text path, Release. Scratchpad ntc-gpu-both-paths.unified.log: plain 4175 tokens, MTP 4131 tokens, drafted 2821, accepted 2763, no fallbacks; first divergence at sampled index 1691, plain margin 0.125. That is past the 1e-3 near-tie gate. The router path on the same prompt parts at index 40 with margin 0.0, which passes.

This is unchecked on the base commit, so whether commit 4fe2dd46 changed it is not known. `MTPGuidedRouterPathTests` omits the text path for that reason.

- [ ] Run the same turn on the text path on the base commit of 4fe2dd46 and record whether it diverges.
- [ ] Find the cause of the divergence at margin 0.125.
- [ ] Fix it, or decide with the user what gap the text path may have, and add the text path to `MTPGuidedRouterPathTests`.
- [ ] Revise the README sentence that ties a guided divergence to the two best logits lying close if the cause proves otherwise. #mtp #foundation-models