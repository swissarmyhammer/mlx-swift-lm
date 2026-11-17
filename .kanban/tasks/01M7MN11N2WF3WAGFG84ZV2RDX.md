---
assignees:
- claude-code
position_column: todo
position_ordinal: '9480'
title: 'MLXLanguageModel: KV-cache configuration option (kvCache:) applied to generation, warm-up and the prompt-cache key'
---
## Why
FoundationModelsRouter (Router card ^2fet0gs) asked for KV-cache quantization to be settable per model from the ACPAgent config (`profile.kvBits`). The chain is MLXLanguageModel (here) → Extras MLXModelLoader → Router LiveModelLoader → ACPAgent. Router maps `.bits8` / `.bits4` to `KVCacheConfiguration(strategy: .affine(try AffineKVCacheConfiguration(bits: N, groupSize: 64)))` (Libraries/MLXLMCommon/KVCacheConfiguration.swift). Write it TDD: failing tests first.

## What
- `MLXLanguageModel.init(..., kvCache: KVCacheConfiguration? = nil)` on the progress initializer (Libraries/MLXFoundationModels/MLXLanguageModel.swift ~513). The deprecated initializer (~485) may be removed; no backward compatibility is needed for it.
- Store it, and apply it as `parameters.kvCache` in `Executor.makeParameters` (~1033) and in the warm-up `GenerateParameters` (~620, including the speculative warm-up), so warm-up runs the quantized path.
- The executor prompt cache must never restore a full-precision cache under an 8-bit configuration, or the reverse. Put the configuration (or its resolved strategy identity) in the cache key / compatibility check.
- Real-model check: on Qwen3.8-27B (hybrid, 16 full-attention layers; mxfp4 if present locally, else the 4-bit build) affine 8-bit works with PromptCacheCheckpoint and multi-turn prefix reuse; also Flint oQ4e. Real-model runs take the scratchpad bench.lock.

## Acceptance
- [ ] Unit tests (written first, failing first): the option reaches makeParameters and warm-up; a prompt cache built under one KV configuration is not reused under another; a same-configuration second turn reuses the prefix.
- [ ] Real-model KV8 multi-turn reuse verified on Qwen3.8 and Flint oQ4e (prefix hit reported, coherent output).
- [ ] All 5 test bundles green via xcrun xctest; zero warnings; swift-format run.
- [ ] Reply to FoundationModelsRouter with the stable SHA once pushed. #kv-config