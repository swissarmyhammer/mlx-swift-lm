---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7jb6rjde0m7yezbkq9r0gq3
  text: |-
    ### skipped — gated on GO, ^sbwbavm is NO-GO
    DFlash2 4-bit reached 0.88/0.89/0.93x of MTP bs4 on agent turns at 18-21k/28-29k/67-71k (round 1.85-2.5 plain steps vs MTP ~1.7). Not wired into the router. Re-measure DFlash2 b5 after the qwen-perf verify cards (^ets0cjp, ^117ymn8) land; the port is on branch dflash2-4bit (b4fd6c79).
  timestamp: 2026-11-16T14:12:03.789456+00:00
depends_on:
- 01M7J4BHB7KZS32WH5QSBWBAVM
position_column: todo
position_ordinal: '8180'
title: Wire the 4-bit DFlash2 drafter into the router path (MLXVLM Qwen35 + MLXLanguageModel) with prompt-cache reuse
---
Do this only if the go/no-go card records GO.

## Why
The router loads Qwen3.8 through `loadModelContainer` (the MLXVLM Qwen35 path) and generates through the FoundationModels `MLXLanguageModel` with MTP. The old dflash2 branch only touched the MLXLLM Qwen35 model and ChatSession, so the router could not use it.

## What
- Add the target-layer hidden-state taps to the MLXVLM Qwen35 model, sharing code with MLXLLM.
- Add drafter loading in the same style as the MTP drafter (deferred behind the memory policy). Choose MTP vs DFlash2 through generation parameters/metadata the same way MTP is chosen.
- In MLXLanguageModel and ExecutorPromptCache:
  - Cache the drafter state alongside the main KV.
  - Support text, reasoning, tools and guided generation (with grammar rollback).
  - Keep checkpoints for the hybrid model.
- Report fallbacks the way MTP does.

## Acceptance
- [ ] Follow-up turns prefill only the new message (cachedPromptTokenCount > 0, no full recompute), in both ChatSession and FoundationModels.
- [ ] Guided JSON is correct under DFlash2.
- [ ] Greedy output matches plain (near-tie tolerance as in the MTP tests).
- [ ] All five test bundles green; IntegrationTesting Qwen3.8 prompt-cache suites pass (run one invocation at a time). #dflash2-4bit