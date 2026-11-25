---
assignees:
- claude-code
position_column: todo
position_ordinal: 8c80
title: MLXLMCommon DocC build fails on pre-existing broken symbol links
---
Found while working ^ffn1stg. `MLX_SWIFT_BUILD_DOC=1 swift package --build-system native generate-documentation --target MLXLMCommon --warnings-as-errors` (scripts/verify-docs.sh) fails on links that predate it: `_load(configuration:tokenizerLoader:)` / `_wrap(_:)` at MTPDrafterModelFactory ContextType/ContainerType; `drafterCreator(_:create:)` in ModelTypeRegistry.registerDrafter (both overloads); `MatvecLayout` in MultiRowQuantizedMatvec.canRun; `offset` / `startPosition` in KVCacheSimple/ChunkedKVCache.currentKeyValues(). Fix each link so the docs build passes. #cluster