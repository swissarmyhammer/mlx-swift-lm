// Copyright © 2026 Apple Inc.
//
// Coverage for kanban `r9rf5g7`: hybrid Mamba/attention architectures
// (Qwen3.6/Qwen3-Next family -- `Qwen35Model`/`Qwen35TextModel` in
// `Libraries/MLXLLM/Models/Qwen35.swift`, `Qwen3NextModel` in
// `Libraries/MLXLLM/Models/Qwen3Next.swift`) build a `[KVCache]` mixing
// `KVCacheSimple` (full-attention layers) with `MambaCache` (Gated-DeltaNet
// linear-attention layers), so `PromptCache`'s chunk store -- which requires
// every layer to be a verified `KVCacheSimple` (see
// `PromptCacheChunks.swift`'s `verifiedSimpleLayers`) -- can never engage for
// them. See `PromptCache.isChunkable`'s doc comment (`PromptCacheChunks.swift`)
// for the full decision record on why that's a permanent architectural fact
// (not a temporary gap) and why partial/checkpointed reuse were both
// considered and rejected.
//
// This file proves:
//   1. `PromptCache.isChunkable`/`MLXLanguageModel.supportsPromptCacheReuse`
//      correctly signal `false` for REAL hybrid models (`Qwen35TextModel`,
//      `Qwen3NextModel`), and `true` for a real pure-attention model --
//      regression guard against (1) ever flipping accidentally and (2)
//      hybrid detection being a coincidence of some other check.
//   2. `store()` still safely no-ops (chunk count stays zero, no crash) when
//      handed a genuinely hybrid `[KVCacheSimple, MambaCache]` stack -- no
//      regression to the existing pure-attention chunk-cache behavior.
//
// NOTE on the task's referenced integration test: the task description names
// `Tests/FoundationModelsRouterIntegrationTests/LanguageModelSessionBackendTests.swift`'s
// `secondTurnReusesFirstTurnsKVCache` as the discovery vehicle. That path does
// not exist anywhere in this repository's git history (confirmed via `git log
// --all -- '**/FoundationModelsRouterIntegrationTests/**'`, zero hits) --
// there is no router-side integration test to update with a skip/xfail here.
// Running the real `mlx-community/Qwen3.6-27B-mxfp4` model this sandbox has
// no weights for and no network access to download was not attempted; the
// unit-level coverage below exercises the same fact (hybrid cache stacks
// never engage the chunk store) without needing real weights.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache hybrid Mamba/attention architecture handling")
struct PromptCacheHybridArchitectureTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixtures

    /// A tiny, real `Qwen35TextModel` (Qwen3.6 family) whose module init
    /// stays cheap -- small dims, no real weight download. `fullAttentionInterval:
    /// 2` over 4 layers yields the hybrid shape under test:
    /// `[Mamba, Attention, Mamba, Attention]` (`layerIdx + 1) % 2 != 0`
    /// selects Mamba for layers 0 and 2).
    private func makeHybridQwen35Model() throws -> Qwen35TextModel {
        let json = """
            {
                "hidden_size": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 0,
                "num_experts_per_tok": 0
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        return Qwen35TextModel(config)
    }

    /// A tiny, real `Qwen3NextModel` (Qwen3-Next family) with the same
    /// hybrid shape as `makeHybridQwen35Model`, over the sibling
    /// architecture the task also names.
    private func makeHybridQwen3NextModel() throws -> Qwen3NextModel {
        let json = """
            {
                "hidden_size": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 0,
                "moe_intermediate_size": 0,
                "rms_norm_eps": 0.000001,
                "vocab_size": 32,
                "num_key_value_heads": 1,
                "full_attention_interval": 2
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: Data(json.utf8))
        return Qwen3NextModel(config)
    }

    // MARK: - Capability signal: real hybrid models report `false`

    @Test("a real Qwen35TextModel's cache is not chunkable")
    func qwen35HybridModelIsNotChunkable() throws {
        let model = try makeHybridQwen35Model()
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4, "sanity: 4 layers, matching num_hidden_layers")
        #expect(
            cache.contains { type(of: $0) == MambaCache.self },
            "sanity: this shape must actually be hybrid, or the test proves nothing")
        #expect(PromptCache.isChunkable(cache) == false)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.supportsPromptCacheReuse(model: model) == false)
    }

    @Test("a real Qwen3NextModel's cache is not chunkable")
    func qwen3NextHybridModelIsNotChunkable() throws {
        let model = try makeHybridQwen3NextModel()
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4, "sanity: 4 layers, matching num_hidden_layers")
        #expect(
            cache.contains { type(of: $0) == MambaCache.self },
            "sanity: this shape must actually be hybrid, or the test proves nothing")
        #expect(PromptCache.isChunkable(cache) == false)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.supportsPromptCacheReuse(model: model) == false)
    }

    // MARK: - Capability signal: pure-attention models still report `true`

    @Test("a pure-attention model's cache is still chunkable (regression guard)")
    func pureAttentionModelIsChunkable() {
        let model = PromptCacheProbeModel()

        #expect(PromptCache.isChunkable(model.newCache(parameters: nil)) == true)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.supportsPromptCacheReuse(model: model) == true)
    }

    @Test("a maxKVSize-driven RotatingKVCache stack is not chunkable")
    func rotatingCacheModelIsNotChunkable() {
        let model = PromptCacheProbeModel()
        let parameters = GenerateParameters(maxKVSize: 16)

        #expect(PromptCache.isChunkable(model.newCache(parameters: parameters)) == false)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.supportsPromptCacheReuse(model: model, parameters: parameters)
                == false)
    }

    // MARK: - Direct isChunkable checks

    @Test("isChunkable is false for an empty cache stack")
    func isChunkableFalseForEmptyCache() {
        #expect(PromptCache.isChunkable([]) == false)
    }

    @Test("isChunkable is true for a uniform KVCacheSimple stack")
    func isChunkableTrueForUniformSimpleStack() {
        #expect(PromptCache.isChunkable([KVCacheSimple(), KVCacheSimple()]) == true)
    }

    @Test("isChunkable is false for a stack mixing KVCacheSimple and MambaCache")
    func isChunkableFalseForMixedStack() {
        #expect(PromptCache.isChunkable([KVCacheSimple(), MambaCache()]) == false)
        #expect(PromptCache.isChunkable([MambaCache(), KVCacheSimple()]) == false)
    }

    @Test("isChunkable is false for an all-MambaCache stack")
    func isChunkableFalseForAllMambaStack() {
        #expect(PromptCache.isChunkable([MambaCache(), MambaCache()]) == false)
    }

    // MARK: - store() safely no-ops on a genuinely hybrid stack (no regression)

    @Test("store() drops a round backed by a hybrid [KVCacheSimple, MambaCache] stack")
    func storeDropsRoundBackedByHybridStack() async {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let simple = makeChunkableCache(tokenCount: tokenCount)
        let mamba = MambaCache()
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [simple, mamba], chunkSize: chunkSize)
        #expect(chunks == nil, "sliceChunks must reject the hybrid stack, matching store()'s own check")

        let cache = PromptCache()
        let modelID = "hybrid-store-no-op-\(UUID().uuidString)"
        await cache.store(
            modelID: modelID, tokens: tokens, cache: SendableBox([simple, mamba]))

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "a hybrid stack must never populate the chunk store")
    }
}

#endif
