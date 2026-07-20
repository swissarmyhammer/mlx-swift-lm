---
assignees:
- claude-code
depends_on:
- 01KXY10EDJVJJRPZDPW8DBC476
- 01KXY126Z47BS8VE2DW9A2AW98
position_column: todo
position_ordinal: '8e80'
title: 'MiniMax-M3: MTP speculative decoding via the 7 MTP modules'
---
## What

Exploit M3's 7 multi-token-prediction (MTP) modules for speculative self-drafting — novel work; the mlx-vlm reference sanitizes these weights away. This repo already has the machinery: `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift` plus prior art in `Libraries/MLXLLM/Models/DeepseekV3.swift` (MTP drafter) and the Gemma4 drafter work (`Tests/MLXLMTests/MTPDrafterModelTests.swift`, `Gemma4EmitDrafterStateTests.swift`, `IntegrationTesting/.../MTPRung4TokenParityTests.swift`).

1. **Un-drop the MTP weights**: revise ^xgvth41's `sanitize` to keep the MTP-module weights (verify exact key prefixes against the checkpoint index — `model.mtp.*` or similar) instead of discarding them.
2. **MTP drafter module** in `Libraries/MLXVLM/Models/MiniMaxM3.swift`: port the MTP head structure from the MiniMaxAI reference implementation (transformers custom code in `MiniMaxAI/MiniMax-M3` — there is NO mlx reference for this; read the HF repo's modeling code for the MTP block: per-module norm/projection/decoder-layer + shared lm_head, `next_n_predict_layers: 1`).
3. **Wire into `MTPSpeculativeTokenIterator`** the same way DeepseekV3/Gemma4 expose their drafters (conform to whatever drafter protocol `MTPDrafterModelFactory` expects — see `IntegrationTesting/.../MTPDrafterModelFactoryIntegrationTests.swift`).

If, on reading the real checkpoint/modeling code, the MTP heads turn out structurally incompatible with `MTPSpeculativeTokenIterator`'s contract, STOP and report the mismatch with specifics — do not force it or silently descope (user requires sign-off on scope changes).

## Acceptance Criteria

- [ ] `sanitize` keeps MTP weights; full-weights load still reports zero unconsumed keys
- [ ] Tiny-config drafter emits k draft tokens per step and the verifier accepts/rejects correctly (token parity: speculative output == non-speculative output token-for-token, mirroring `MTPRung4TokenParityTests`)
- [ ] Real-weights gated test: speculative decoding on MiniMax-M3-4bit produces token-identical output to non-speculative for a fixed prompt/seed, with measured acceptance rate logged
- [ ] Non-speculative M3 generation and all other MTP-using models (DeepseekV3, Gemma4) unaffected — existing MTP suites green

## Tests

- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift` + `Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift`: tiny-config drafter/parity tests
- [ ] Gated integration parity test (pattern: `MTPRung4TokenParityTests.swift`)
- [ ] Run: `swift test --filter MLXLMTests` → green; integration parity case passes

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax