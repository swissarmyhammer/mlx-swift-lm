---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m88bwk6tp5rqbx50zjazmvh8
  text: |-
    ### implement — changed
    - evidence: 97d2b2f7. verify-docs.sh failed in 4 targets (MLXLMCommon, MLXLLM, MLXVLM, MLXFoundationModels), 30 links in all: the ones listed here plus Flint, LFM2DSpark, Qwen35 (LLM and VLM), Inkling2/Inkling2Draft, MLXLanguageModel and MLXSpeculativeDecoding links to internal, package, private or other-module symbols. Each is now code voice. `scripts/verify-docs.sh`: "All documentation builds passed."
    - next: none.
  timestamp: 2026-11-25T03:27:16.698574+00:00
position_column: done
position_ordinal: fffffb80
title: MLXLMCommon DocC build fails on pre-existing broken symbol links
---
Found while working ^ffn1stg. `MLX_SWIFT_BUILD_DOC=1 swift package --build-system native generate-documentation --target MLXLMCommon --warnings-as-errors` (scripts/verify-docs.sh) fails on links that predate it: `_load(configuration:tokenizerLoader:)` / `_wrap(_:)` at MTPDrafterModelFactory ContextType/ContainerType; `drafterCreator(_:create:)` in ModelTypeRegistry.registerDrafter (both overloads); `MatvecLayout` in MultiRowQuantizedMatvec.canRun; `offset` / `startPosition` in KVCacheSimple/ChunkedKVCache.currentKeyValues(). Fix each link so the docs build passes. #cluster