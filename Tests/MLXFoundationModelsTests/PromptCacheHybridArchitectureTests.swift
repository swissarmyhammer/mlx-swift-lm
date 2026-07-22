// Copyright © 2026 Apple Inc.
//
// Coverage for kanban `r9rf5g7`: hybrid Mamba/attention architectures
// (Qwen3.6/Qwen3-Next family -- `Qwen35Model`/`Qwen35TextModel` in
// `Libraries/MLXLLM/Models/Qwen35.swift`, `Qwen3NextModel` in
// `Libraries/MLXLLM/Models/Qwen3Next.swift`) build a `[KVCache]` mixing
// `KVCacheSimple` (full-attention layers) with `MambaCache` (Gated-DeltaNet
// linear-attention layers). The ordinary CHUNK store (`sliceChunks`/
// `sliceTailChunk`, which require every layer to be a verified
// `KVCacheSimple`) can never engage for them -- see `PromptCache.isChunkable`'s
// doc comment. But that does NOT mean hybrid architectures can never
// participate in prompt-cache reuse at all: `PromptCache.isHybridMambaAttention(_:)`/
// `snapshotHybridCheckpoint(tokens:cache:)`/`restoreHybridCheckpoint(_:)`
// implement a SEPARATE, parallel "hybrid checkpoint" mechanism -- capturing
// the ENTIRE, unsliced state of every layer at a round boundary and
// restoring it verbatim on a later round whose prompt starts with an exact
// match of the stored prefix. This works because `Qwen35GatedDeltaNet
// .callAsFunction`/`Qwen3NextGatedDeltaNet.callAsFunction` (the Mamba
// layer's forward pass) read only `cache?[0]`/`cache?[1]` as raw VALUES --
// never `cache.offset` or any other position signal -- so resuming from an
// externally-stored checkpoint is exactly as valid as resuming mid-forward-call
// (ordinary incremental generation already relies on this).
//
// This file proves:
//   1. `PromptCache.isChunkable` still correctly reports `false` for hybrid
//      stacks (chunk slicing genuinely cannot apply to Mamba's collapsed
//      state), while `MLXLanguageModel.supportsPromptCacheReuse` reports
//      `true` for them -- they now support reuse via the checkpoint
//      mechanism, not chunking.
//   2. `store()` populates the HYBRID checkpoint store (not the chunk
//      store) for a genuine hybrid stack, and never both.
//   3. `resolveHybridCheckpoint` picks the longest valid prefix match,
//      falls back to no-match on divergence, and never matches a
//      checkpoint covering the ENTIRE new prompt (must leave >=1 token to feed).
//   4. Byte-budget LRU eviction reclaims hybrid checkpoints, and can pick a
//      hybrid checkpoint over a newer chunk-store entry (global LRU spans
//      BOTH stores).
//   5. THE CORRECTNESS BAR: generating via resolve() → store() → resolve()
//      (restoring a checkpoint and feeding only the suffix) against a real
//      tiny `Qwen35TextModel`/`Qwen3NextModel` produces numerically
//      equivalent logits to one full, fresh forward pass over the whole
//      sequence -- proving the checkpoint scheme is actually sound, not
//      just plausible by construction.
//
// NOTE on the task's referenced integration test: the task description names
// `Tests/FoundationModelsRouterIntegrationTests/LanguageModelSessionBackendTests.swift`'s
// `secondTurnReusesFirstTurnsKVCache` as the discovery vehicle. That path does
// not exist anywhere in this repository's git history (confirmed via `git log
// --all -- '**/FoundationModelsRouterIntegrationTests/**'`, zero hits) --
// there is no router-side integration test to update with a skip/xfail here.
// Running the real `mlx-community/Qwen3.6-27B-mxfp4` model this sandbox has
// no weights for and no network access to download was not attempted; the
// coverage below exercises the same fact -- and the actual numerical
// correctness of hybrid cache reuse -- against real, tiny model instances
// instead.

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
    ///
    /// `linear_key_head_dim`/`linear_value_head_dim` must be `>= 32`: the
    /// custom Metal kernel backing `gatedDeltaUpdate` (`GatedDelta.swift`)
    /// computes `constexpr int n_per_t = Dk / 32` and declares
    /// `float state[n_per_t]` -- a smaller `Dk` (this task's prior
    /// capability-signaling-only tests used `8`, since they never actually
    /// ran a forward pass) makes `n_per_t == 0`, a zero-length C++ array the
    /// Metal shader compiler rejects outright ("zero-length arrays are not
    /// permitted"). `32` is the minimum that compiles -- mirrors
    /// `Qwen35ContinuationTests`' proven-working tiny-model dims.
    ///
    /// `head_dim: 32` is explicit so the full-attention layers' RoPE
    /// (`ropeDims = Int(headDim * partialRotaryFactor)`, `partialRotaryFactor`
    /// defaulting to `0.25`) lands on `8` -- even, satisfying MLX's RoPE
    /// kernel requirement (`dims must be even`).
    private func makeHybridQwen35Model() throws -> Qwen35TextModel {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 64,
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
    /// hybrid shape and the same `>= 32`-head-dim constraint as
    /// `makeHybridQwen35Model` (see that fixture's doc comment for why),
    /// over the sibling architecture the task also names.
    private func makeHybridQwen3NextModel() throws -> Qwen3NextModel {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "num_experts_per_tok": 0,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 0,
                "moe_intermediate_size": 0,
                "rms_norm_eps": 0.000001,
                "vocab_size": 64,
                "num_key_value_heads": 2,
                "full_attention_interval": 2
            }
            """
        let config = try JSONDecoder().decode(
            Qwen3NextConfiguration.self, from: Data(json.utf8))
        return Qwen3NextModel(config)
    }

    /// A `HybridCheckpoint` with placeholder (never content-inspected)
    /// tensors -- for tests exercising `resolveHybridCheckpoint`'s
    /// longest-prefix-match/eviction bookkeeping, which never reads
    /// `layers`' tensor content, only `tokens`/`byteSize`/`lastUsed`.
    private func makeHybridCheckpoint(
        tokens: [Int], byteSize: Int = 64
    ) -> PromptCache.HybridCheckpoint {
        let dummy = MLXArray([Int32(0)])
        return PromptCache.HybridCheckpoint(
            tokens: tokens,
            layers: [
                (kind: .mamba, state: [dummy, dummy]),
                (kind: .simple, state: [dummy, dummy]),
            ],
            byteSize: byteSize, lastUsed: 0)
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: - Capability signals

    /// Shared body for `qwen35HybridModelIsNotChunkableButSupportsReuse`/
    /// `qwen3NextHybridModelIsNotChunkableButSupportsReuse`: builds
    /// `modelFactory`'s model and asserts its cache is a genuine hybrid
    /// stack (4 layers, at least one `MambaCache`) that's not chunkable but
    /// IS a hybrid-checkpoint candidate reporting `supportsPromptCacheReuse
    /// == true`.
    ///
    /// - Parameter modelFactory: Builds the real, tiny hybrid model under test.
    private func assertHybridModelIsNotChunkableButSupportsReuse(
        modelFactory: () throws -> any MLXLMCommon.LanguageModel
    ) throws {
        let model = try modelFactory()
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4, "sanity: 4 layers, matching num_hidden_layers")
        #expect(
            cache.contains { type(of: $0) == MambaCache.self },
            "sanity: this shape must actually be hybrid, or the test proves nothing")
        #expect(PromptCache.isChunkable(cache) == false)
        #expect(PromptCache.isHybridMambaAttention(cache) == true)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.supportsPromptCacheReuse(model: model) == true)
    }

    @Test("a real Qwen35TextModel's cache is not chunkable, but IS a hybrid checkpoint candidate")
    func qwen35HybridModelIsNotChunkableButSupportsReuse() throws {
        try assertHybridModelIsNotChunkableButSupportsReuse(modelFactory: makeHybridQwen35Model)
    }

    @Test("a real Qwen3NextModel's cache is not chunkable, but IS a hybrid checkpoint candidate")
    func qwen3NextHybridModelIsNotChunkableButSupportsReuse() throws {
        try assertHybridModelIsNotChunkableButSupportsReuse(modelFactory: makeHybridQwen3NextModel)
    }

    @Test("a pure-attention model's cache is still chunkable (regression guard)")
    func pureAttentionModelIsChunkable() {
        let model = PromptCacheProbeModel()

        #expect(PromptCache.isChunkable(model.newCache(parameters: nil)) == true)
        #expect(PromptCache.isHybridMambaAttention(model.newCache(parameters: nil)) == false)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.supportsPromptCacheReuse(model: model) == true)
    }

    @Test("a maxKVSize-driven RotatingKVCache stack is neither chunkable nor hybrid")
    func rotatingCacheModelSupportsNeitherMechanism() {
        let model = PromptCacheProbeModel()
        let parameters = GenerateParameters(maxKVSize: 16)
        let cache = model.newCache(parameters: parameters)

        #expect(PromptCache.isChunkable(cache) == false)
        #expect(PromptCache.isHybridMambaAttention(cache) == false)

        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.supportsPromptCacheReuse(model: model, parameters: parameters)
                == false)
    }

    // MARK: - Direct isChunkable / isHybridMambaAttention checks

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

    @Test("isHybridMambaAttention requires at least one of EACH kind")
    func isHybridMambaAttentionRequiresBothKinds() {
        #expect(PromptCache.isHybridMambaAttention([KVCacheSimple(), MambaCache()]) == true)
        #expect(PromptCache.isHybridMambaAttention([MambaCache(), KVCacheSimple()]) == true)
        #expect(PromptCache.isHybridMambaAttention([KVCacheSimple(), KVCacheSimple()]) == false)
        #expect(PromptCache.isHybridMambaAttention([MambaCache(), MambaCache()]) == false)
        #expect(PromptCache.isHybridMambaAttention([]) == false)
    }

    // MARK: - store() routes a hybrid stack to the checkpoint store, not the chunk store

    @Test("store() checkpoints a round backed by a hybrid [KVCacheSimple, MambaCache] stack")
    func storeCheckpointsHybridStackInsteadOfChunking() async {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let simple = makeChunkableCache(tokenCount: tokenCount)
        let mamba = MambaCache()
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [simple, mamba], chunkSize: chunkSize)
        #expect(chunks == nil, "sliceChunks must reject the hybrid stack, matching store()'s own check")

        let cache = PromptCache()
        let modelID = "hybrid-store-checkpoints-\(UUID().uuidString)"
        await cache.store(
            modelID: modelID, tokens: tokens, cache: SendableBox([simple, mamba]))

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "a hybrid stack must never populate the ordinary chunk store")
        #expect(
            await cache.hybridCheckpointCount(modelID: modelID) == 1,
            "a hybrid stack's round must populate the hybrid checkpoint store instead")
    }

    // MARK: - resolveHybridCheckpoint matching semantics

    @Test("resolveHybridCheckpoint picks the LONGEST valid prefix match among multiple stored checkpoints")
    func resolveHybridCheckpointPicksLongestMatch() async {
        let cache = PromptCache()
        let modelID = "hybrid-longest-\(UUID().uuidString)"
        let tokens = Array(0 ..< 20)

        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: Array(tokens.prefix(5))))
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: Array(tokens.prefix(12))))
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: Array(tokens.prefix(3))))

        let resolved = await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: tokens)
            .consume()
        #expect(resolved?.tokens == Array(tokens.prefix(12)))
    }

    @Test("resolveHybridCheckpoint falls back to no match when the stored prefix diverges from newTokens")
    func resolveHybridCheckpointFallsBackOnMismatch() async {
        let cache = PromptCache()
        let modelID = "hybrid-mismatch-\(UUID().uuidString)"
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: [1, 2, 3, 4]))

        let divergent = [1, 2, 99, 4, 5, 6]
        let resolved = await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: divergent)
            .consume()
        #expect(resolved == nil)
    }

    @Test("resolveHybridCheckpoint never matches a checkpoint covering ALL of newTokens")
    func resolveHybridCheckpointRequiresAtLeastOneUnmatchedToken() async {
        let cache = PromptCache()
        let modelID = "hybrid-full-cover-\(UUID().uuidString)"
        let tokens = [1, 2, 3, 4]
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: tokens))

        let resolved = await cache.resolveHybridCheckpoint(modelID: modelID, newTokens: tokens)
            .consume()
        #expect(
            resolved == nil,
            "a checkpoint covering the entire prompt must never match -- generation needs >=1 fresh token")
    }

    @Test("resolveHybridCheckpoint returns nil for a model with nothing stored")
    func resolveHybridCheckpointNilWhenNothingStored() async {
        let cache = PromptCache()
        let resolved = await cache.resolveHybridCheckpoint(
            modelID: "hybrid-empty-\(UUID().uuidString)", newTokens: [1, 2, 3]
        ).consume()
        #expect(resolved == nil)
    }

    // MARK: - Byte-budget LRU eviction spans both stores

    @Test("byte-budget eviction reclaims hybrid checkpoints under pressure")
    func byteBudgetEvictionReclaimsHybridCheckpoints() async {
        let cache = PromptCache()
        let modelID = "hybrid-evict-\(UUID().uuidString)"
        await cache.setByteBudget(100)

        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: [1, 2, 3], byteSize: 80))
        #expect(await cache.hybridCheckpointCount(modelID: modelID) == 1)

        // A second checkpoint pushes the total over budget; the older
        // (lower-recency) entry must be evicted to reclaim its bytes.
        await cache.insertHybridCheckpoint(
            modelID: modelID,
            checkpoint: makeHybridCheckpoint(tokens: [1, 2, 3, 4, 5], byteSize: 80))

        #expect(await cache.hybridCheckpointCount(modelID: modelID) == 1)
        #expect(await cache.totalStoredByteCount() <= 100)
    }

    @Test("global LRU eviction can evict a hybrid checkpoint even when a chunk-store entry is newer")
    func globalLRUEvictionSpansBothStores() async {
        let cache = PromptCache()
        let modelID = "hybrid-mixed-evict-\(UUID().uuidString)"
        await cache.setByteBudget(150)

        // Older entry: a hybrid checkpoint.
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: [1, 2, 3], byteSize: 100))

        // Newer entry: an ordinary chunk, real content so its byteSize is genuine.
        let chunkTokenCount = 4
        let simple = makeChunkableCache(tokenCount: chunkTokenCount)
        let tokens = Array(0 ..< chunkTokenCount)
        if let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [simple], chunkSize: chunkTokenCount)
        {
            await cache.insert(modelID: modelID, chunks: chunks)
        }

        #expect(
            await cache.hybridCheckpointCount(modelID: modelID) == 0,
            "the OLDER hybrid checkpoint should be evicted before the NEWER chunk")
        #expect(await cache.chunkCount(modelID: modelID) == 1)
        #expect(await cache.totalStoredByteCount() <= 150)
    }

    // MARK: - setChunkSize preserves hybrid checkpoints (chunk-size-independent)

    @Test(
        "setChunkSize on a genuine change evicts the chunk store but preserves hybrid checkpoints"
    )
    func setChunkSizeEvictsChunksButPreservesHybridCheckpoints() async {
        let cache = PromptCache()
        let modelID = "hybrid-survives-chunk-resize-\(UUID().uuidString)"
        let chunkSize = 8

        // Seed the ordinary chunk store.
        let tokens = Array(0 ..< (chunkSize * 2))
        await cache.setChunkSize(chunkSize)
        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeChunkableCache(tokenCount: tokens.count)]))
        #expect(await cache.chunkCount(modelID: modelID) > 0)

        // Seed a hybrid checkpoint -- keyed by its full token prefix, not by
        // `chunkSize`-aligned windows.
        await cache.insertHybridCheckpoint(
            modelID: modelID, checkpoint: makeHybridCheckpoint(tokens: [1, 2, 3]))
        #expect(await cache.hybridCheckpointCount(modelID: modelID) == 1)

        // A genuine chunkSize change must evict the chunk store (existing
        // invariant, unchanged) but must NOT evict the hybrid checkpoint --
        // it isn't keyed by chunkSize at all.
        await cache.setChunkSize(chunkSize * 2)

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "a genuine chunkSize change must still evict the chunk-size-dependent chunk store")
        #expect(
            await cache.hybridCheckpointCount(modelID: modelID) == 1,
            "hybrid checkpoints are keyed by full token prefix, independent of chunkSize, so setChunkSize must not evict them"
        )
    }

    // MARK: - THE CORRECTNESS BAR: checkpoint-restored generation ≡ full forward pass

    /// Runs `tokens` through `model` fresh from an empty cache, in one shot,
    /// and returns the logits for the positions starting at `suffixStart`
    /// (the reference the checkpoint-restored path must match).
    private func referenceSuffixLogits(
        model: any MLXLMCommon.LanguageModel, tokens: [Int], suffixStart: Int
    ) -> MLXArray {
        let cache = model.newCache(parameters: nil)
        let input = MLXArray(tokens).expandedDimensions(axis: 0)
        let logits = model.callAsFunction(input, cache: cache)
        return logits[0..., suffixStart..., 0...]
    }

    /// Round 1: `resolve()` (a fresh cache, nothing stored yet) → forward
    /// pass over `prefixTokens` → `store()`. Round 2: `resolve()` again over
    /// the FULL token sequence -- this must restore the checkpoint stored by
    /// round 1 and hand back only the suffix to feed. Forward-passes that
    /// suffix through the restored cache and returns the resulting logits.
    private func checkpointRestoredSuffixLogits(
        promptCache: PromptCache, modelID: String,
        model: any MLXLMCommon.LanguageModel, allTokens: [Int], prefixLen: Int
    ) async -> MLXArray {
        let prefixTokens = Array(allTokens.prefix(prefixLen))

        let round1 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: prefixTokens, model: model)
        #expect(round1.tokensToFeed == prefixTokens, "first round: nothing stored yet, feed everything")
        let round1Input = MLXArray(round1.tokensToFeed).expandedDimensions(axis: 0)
        _ = model.callAsFunction(round1Input, cache: round1.cache)
        await promptCache.store(
            modelID: modelID, tokens: prefixTokens, cache: SendableBox(round1.cache))

        let round2 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: allTokens, model: model)
        let expectedSuffix = Array(allTokens.suffix(allTokens.count - prefixLen))
        #expect(
            round2.tokensToFeed == expectedSuffix,
            "second round: the stored checkpoint must be restored and only the suffix fed")

        let round2Input = MLXArray(round2.tokensToFeed).expandedDimensions(axis: 0)
        return model.callAsFunction(round2Input, cache: round2.cache)
    }

    /// Shared body for `qwen35CheckpointRestoreMatchesFullForwardPass`/
    /// `qwen3NextCheckpointRestoreMatchesFullForwardPass`: seeds `MLXRandom`,
    /// builds `modelFactory`'s model, derives a 24-token sequence from
    /// `tokenFormula`, and asserts that resolving/storing/restoring a
    /// checkpoint for the first `prefixLen` tokens then feeding just the
    /// suffix reproduces the SAME logits as one full, fresh forward pass.
    ///
    /// - Parameters:
    ///   - seed: The `MLXRandom` seed for this model's (otherwise
    ///     random-initialized) weights, so both the reference and
    ///     checkpoint-restored paths see identical weights.
    ///   - modelFactory: Builds the real, tiny hybrid model under test.
    ///   - tokenFormula: Maps a position `0..<24` to a token ID.
    ///   - prefixLen: How many leading tokens round 1 stores a checkpoint for.
    ///   - modelIDPrefix: A human-readable prefix for this run's unique model ID.
    private func assertCheckpointRestoreMatchesFullForwardPass(
        seed: UInt64, modelFactory: () throws -> any MLXLMCommon.LanguageModel,
        tokenFormula: (Int) -> Int, prefixLen: Int, modelIDPrefix: String
    ) async throws {
        MLXRandom.seed(seed)
        let model = try modelFactory()
        let allTokens = (0 ..< 24).map(tokenFormula)

        let reference = referenceSuffixLogits(
            model: model, tokens: allTokens, suffixStart: prefixLen)

        let promptCache = PromptCache()
        let modelID = "\(modelIDPrefix)-\(UUID().uuidString)"
        let restored = await checkpointRestoredSuffixLogits(
            promptCache: promptCache, modelID: modelID, model: model, allTokens: allTokens,
            prefixLen: prefixLen)

        let diff = maxAbsDiff(restored, reference)
        #expect(
            diff <= 1e-3,
            "checkpoint-restored suffix generation diverged from a full forward pass (diff \(diff))")
    }

    @Test("checkpoint-restored generation matches a full fresh forward pass (Qwen35TextModel)")
    func qwen35CheckpointRestoreMatchesFullForwardPass() async throws {
        try await assertCheckpointRestoreMatchesFullForwardPass(
            seed: 1001, modelFactory: { try self.makeHybridQwen35Model() },
            tokenFormula: { ($0 * 7 + 3) % 32 }, prefixLen: 9,
            modelIDPrefix: "qwen35-hybrid-correctness")
    }

    @Test("checkpoint-restored generation matches a full fresh forward pass (Qwen3NextModel)")
    func qwen3NextCheckpointRestoreMatchesFullForwardPass() async throws {
        try await assertCheckpointRestoreMatchesFullForwardPass(
            seed: 1002, modelFactory: { try self.makeHybridQwen3NextModel() },
            tokenFormula: { ($0 * 11 + 5) % 32 }, prefixLen: 10,
            modelIDPrefix: "qwen3next-hybrid-correctness")
    }

    /// Shared body for `qwen35SecondContinuationRoundAlsoMatches`/
    /// `qwen3NextSecondContinuationRoundAlsoMatches`: seeds `MLXRandom`,
    /// builds `modelFactory`'s model, derives a 30-token sequence from
    /// `tokenFormula`, and chains THREE rounds of resolve/generate/store --
    /// prefix-only, then extended to a longer prefix (restoring round 1's
    /// checkpoint), then the full sequence (restoring round 2's checkpoint)
    /// -- asserting each round's fed tokens are exactly the expected suffix
    /// and that the final round's logits match one full, fresh forward pass.
    ///
    /// - Parameters:
    ///   - seed: The `MLXRandom` seed for this model's weights.
    ///   - modelFactory: Builds the real, tiny hybrid model under test.
    ///   - tokenFormula: Maps a position `0..<30` to a token ID.
    ///   - modelIDPrefix: A human-readable prefix for this run's unique model ID.
    private func assertSecondContinuationRoundAlsoMatches(
        seed: UInt64, modelFactory: () throws -> any MLXLMCommon.LanguageModel,
        tokenFormula: (Int) -> Int, modelIDPrefix: String
    ) async throws {
        MLXRandom.seed(seed)
        let model = try modelFactory()
        let allTokens = (0 ..< 30).map(tokenFormula)
        let firstPrefixLen = 8
        let secondPrefixLen = 18

        let reference = referenceSuffixLogits(
            model: model, tokens: allTokens, suffixStart: secondPrefixLen)

        let promptCache = PromptCache()
        let modelID = "\(modelIDPrefix)-\(UUID().uuidString)"

        // Round 1: prefix only.
        let firstPrefix = Array(allTokens.prefix(firstPrefixLen))
        let round1 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: firstPrefix, model: model)
        _ = model.callAsFunction(
            MLXArray(round1.tokensToFeed).expandedDimensions(axis: 0), cache: round1.cache)
        await promptCache.store(
            modelID: modelID, tokens: firstPrefix, cache: SendableBox(round1.cache))

        // Round 2: extend to the second prefix, restoring round 1's checkpoint.
        let secondPrefix = Array(allTokens.prefix(secondPrefixLen))
        let round2 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: secondPrefix, model: model)
        #expect(round2.tokensToFeed == Array(secondPrefix.suffix(secondPrefixLen - firstPrefixLen)))
        _ = model.callAsFunction(
            MLXArray(round2.tokensToFeed).expandedDimensions(axis: 0), cache: round2.cache)
        await promptCache.store(
            modelID: modelID, tokens: secondPrefix, cache: SendableBox(round2.cache))

        // Round 3: the full sequence, restoring round 2's (longer) checkpoint.
        let round3 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: allTokens, model: model)
        #expect(round3.tokensToFeed == Array(allTokens.suffix(allTokens.count - secondPrefixLen)))
        let round3Logits = model.callAsFunction(
            MLXArray(round3.tokensToFeed).expandedDimensions(axis: 0), cache: round3.cache)

        let diff = maxAbsDiff(round3Logits, reference)
        #expect(
            diff <= 1e-3,
            "chained checkpoint reuse across three rounds diverged from a full forward pass (diff \(diff))")
    }

    @Test("a THIRD round extends reuse: resolving the full sequence again matches the same full forward pass")
    func qwen35SecondContinuationRoundAlsoMatches() async throws {
        try await assertSecondContinuationRoundAlsoMatches(
            seed: 1003, modelFactory: { try self.makeHybridQwen35Model() },
            tokenFormula: { ($0 * 5 + 1) % 32 }, modelIDPrefix: "qwen35-hybrid-chain")
    }

    @Test("a THIRD round extends reuse for Qwen3NextModel too: resolving the full sequence again matches the same full forward pass")
    func qwen3NextSecondContinuationRoundAlsoMatches() async throws {
        try await assertSecondContinuationRoundAlsoMatches(
            seed: 1004, modelFactory: { try self.makeHybridQwen3NextModel() },
            tokenFormula: { ($0 * 3 + 2) % 32 }, modelIDPrefix: "qwen3next-hybrid-chain")
    }

    // MARK: - Transcript-stable boundary (kanban er33v06)

    @Test("commonPrefixLength counts the shared leading run, and nothing else")
    func commonPrefixLengthBasics() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.commonPrefixLength([1, 2, 3], [1, 2, 3]) == 3)
        #expect(MLXLanguageModel.Executor.commonPrefixLength([1, 2, 3, 4], [1, 2, 9, 4]) == 2)
        #expect(MLXLanguageModel.Executor.commonPrefixLength([1, 2], [1, 2, 3, 4]) == 2)
        #expect(MLXLanguageModel.Executor.commonPrefixLength([1, 2, 3, 4], [1, 2]) == 2)
        #expect(MLXLanguageModel.Executor.commonPrefixLength([], [1, 2]) == 0)
        #expect(MLXLanguageModel.Executor.commonPrefixLength([9, 1], [1, 9]) == 0)
    }

    @Test("transcriptStableLength is the prompt's common prefix with the past-turns render")
    func transcriptStableLengthUsesPastTurnsRender() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tokenizer = StableBoundaryProbeTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "ab"]
        ]
        // The live prompt renders WITH the generation-priming region; the
        // base render (addGenerationPrompt: false) stops at the
        // transcript-stable boundary, so the stable length is exactly the
        // base render's full length.
        let promptTokens = try #require(
            try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: true))
        let baseTokens = try #require(
            try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false))
        #expect(promptTokens.count > baseTokens.count, "sanity: priming region must add tokens")

        let stableLength = MLXLanguageModel.Executor.transcriptStableLength(
            promptTokens: promptTokens, messages: messages, tools: nil,
            additionalContext: nil, tokenizer: tokenizer)
        #expect(stableLength == baseTokens.count)
    }

    @Test("transcriptStableLength renders the past-turns baseline with the history context")
    func transcriptStableLengthCarriesHistoryContext() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tokenizer = StableBoundaryProbeTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "ab"],
            ["role": "assistant", "content": "cd"],
            ["role": "user", "content": "ef"],
        ]
        let historyContext: [String: any Sendable] = ["preserve_thinking": true]
        // The live prompt renders WITH the history-preservation context (its
        // history region carries the preserved-thinking markers) plus the
        // generation-priming region.
        let promptTokens = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: historyContext,
                addGenerationPrompt: true))
        let baseWithContext = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: historyContext,
                addGenerationPrompt: false))
        let baseWithout = try #require(
            try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false))
        #expect(
            baseWithContext.count > baseWithout.count,
            "sanity: the history context must change the past-turns render")

        // The baseline must carry the SAME history-affecting context the
        // live render used — a context-free baseline diverges from the live
        // prompt at the first history marker, collapsing the stable prefix.
        let stableLength = MLXLanguageModel.Executor.transcriptStableLength(
            promptTokens: promptTokens, messages: messages, tools: nil,
            additionalContext: historyContext, tokenizer: tokenizer)
        #expect(stableLength == baseWithContext.count)
    }

    @Test("mergedAdditionalContext keeps nil renders nil and merges present fragments")
    func mergedAdditionalContextMergesFragments() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // nil + nil must stay nil: renders that never carried a context must
        // remain byte-identical to before (UserInput treats nil and [:]
        // differently only in intent, but nil is the historical shape).
        #expect(MLXLanguageModel.Executor.mergedAdditionalContext(nil, nil) == nil)

        let strategy: [String: any Sendable] = ["enable_thinking": false]
        let history: [String: any Sendable] = ["preserve_thinking": true]

        let strategyOnly = try #require(
            MLXLanguageModel.Executor.mergedAdditionalContext(strategy, nil))
        #expect(strategyOnly.count == 1)
        #expect(strategyOnly["enable_thinking"] as? Bool == false)

        let historyOnly = try #require(
            MLXLanguageModel.Executor.mergedAdditionalContext(nil, history))
        #expect(historyOnly.count == 1)
        #expect(historyOnly["preserve_thinking"] as? Bool == true)

        let merged = try #require(
            MLXLanguageModel.Executor.mergedAdditionalContext(strategy, history))
        #expect(merged.count == 2)
        #expect(merged["enable_thinking"] as? Bool == false)
        #expect(merged["preserve_thinking"] as? Bool == true)
    }

    @Test("transcriptStableLength is nil when the tokenizer opts out of generation-prompt control")
    func transcriptStableLengthNilForOptedOutTokenizer() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let stableLength = MLXLanguageModel.Executor.transcriptStableLength(
            promptTokens: [1, 2, 3],
            messages: [["role": "user", "content": "x"]],
            tools: nil,
            additionalContext: nil,
            tokenizer: OptedOutProbeTokenizer())
        #expect(stableLength == nil)
    }

    /// Shared body for `qwen35StableBoundaryCheckpointMatchesFullForwardPass`/
    /// `qwen3NextStableBoundaryCheckpointMatchesFullForwardPass`: proves the
    /// SPLIT-PREFILL flow `Executor.makePromptCacheSlot` runs for a hybrid
    /// stack is numerically sound. Round 1's prompt is a stable prefix plus a
    /// generation-priming suffix that later rounds re-render WITHOUT (the
    /// Qwen3.6 template shape from kanban er33v06). The round prefills up to
    /// the stable boundary via `prefillPromptCache`, snapshots a checkpoint
    /// THERE (pre-generation), then keeps feeding the same live cache through
    /// the priming region and stores the usual post-round checkpoint too.
    /// Round 2 diverges from round 1's fed sequence exactly at the boundary:
    /// the post-round checkpoint must miss, the stable-boundary checkpoint
    /// must win, and feeding only the continuation through the restored cache
    /// must reproduce a full fresh forward pass's logits within 1e-3.
    ///
    /// - Parameters:
    ///   - seed: The `MLXRandom` seed for this model's weights.
    ///   - modelFactory: Builds the real, tiny hybrid model under test.
    ///   - tokenFormula: Maps a position `0..<14` to a stable-prefix token ID.
    ///   - modelIDPrefix: A human-readable prefix for this run's unique model ID.
    private func assertStableBoundaryCheckpointMatchesFullForwardPass(
        seed: UInt64, modelFactory: () throws -> any MLXLMCommon.LanguageModel,
        tokenFormula: (Int) -> Int, modelIDPrefix: String
    ) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        MLXRandom.seed(seed)
        let model = try modelFactory()
        let stableLength = 14
        let stableTokens = (0 ..< stableLength).map(tokenFormula)
        // Round 1's fed prompt ends in a generation-priming region ([50, 51])
        // that round 2's re-render of these turns as history will NOT contain.
        let round1Prompt = stableTokens + [50, 51]
        // Round 2 extends the STABLE prefix with different tokens -- it
        // diverges from round 1's fed sequence exactly at the boundary.
        let continuation = [40, 41, 42, 43, 44, 45]
        let round2Tokens = stableTokens + continuation

        let reference = referenceSuffixLogits(
            model: model, tokens: round2Tokens, suffixStart: stableLength)

        let promptCache = PromptCache()
        let modelID = "\(modelIDPrefix)-\(UUID().uuidString)"

        // Round 1: nothing stored yet -- fresh cache, feed everything.
        let round1 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: round1Prompt, model: model)
        #expect(round1.tokensToFeed == round1Prompt, "first round: nothing stored yet")

        // Split prefill: advance the cache to the stable boundary and
        // snapshot a checkpoint THERE, before any priming/generation tokens.
        try MLXLanguageModel.Executor.prefillPromptCache(
            tokens: stableTokens, model: model, cache: round1.cache)
        await promptCache.store(
            modelID: modelID, tokens: stableTokens, cache: SendableBox(round1.cache))
        #expect(
            await promptCache.hybridCheckpointCount(modelID: modelID) == 1,
            "the pre-generation stable-boundary checkpoint must snapshot cleanly")

        // The SAME live cache then continues through the priming region
        // (generation's own prefill) -- the stored checkpoint must be
        // unaffected by this later mutation (owned-copy independence).
        _ = model.callAsFunction(
            MLXArray(Array(round1Prompt[stableLength...])).expandedDimensions(axis: 0),
            cache: round1.cache)
        // The existing post-round store is KEPT: it snapshots the full fed
        // sequence, which round 2's divergent re-render can never match.
        await promptCache.store(
            modelID: modelID, tokens: round1Prompt, cache: SendableBox(round1.cache))
        #expect(await promptCache.hybridCheckpointCount(modelID: modelID) == 2)

        // Round 2: the post-round checkpoint diverges at the boundary and
        // must MISS; the stable-boundary checkpoint must win the
        // longest-prefix scan, feeding only the continuation.
        let round2 = await resolveOnce(
            cache: promptCache, modelID: modelID, newTokens: round2Tokens, model: model)
        #expect(
            round2.tokensToFeed == continuation,
            "the stable-boundary checkpoint must match; the post-round checkpoint cannot")

        let round2Logits = model.callAsFunction(
            MLXArray(round2.tokensToFeed).expandedDimensions(axis: 0), cache: round2.cache)
        let diff = maxAbsDiff(round2Logits, reference)
        #expect(
            diff <= 1e-3,
            "stable-boundary checkpoint restore diverged from a full forward pass (diff \(diff))")
    }

    @Test("a checkpoint snapshotted at the stable boundary mid-round restores and matches (Qwen35TextModel)")
    func qwen35StableBoundaryCheckpointMatchesFullForwardPass() async throws {
        try await assertStableBoundaryCheckpointMatchesFullForwardPass(
            seed: 1005, modelFactory: { try self.makeHybridQwen35Model() },
            tokenFormula: { ($0 * 7 + 3) % 32 },
            modelIDPrefix: "qwen35-hybrid-stable-boundary")
    }

    @Test("a checkpoint snapshotted at the stable boundary mid-round restores and matches (Qwen3NextModel)")
    func qwen3NextStableBoundaryCheckpointMatchesFullForwardPass() async throws {
        try await assertStableBoundaryCheckpointMatchesFullForwardPass(
            seed: 1006, modelFactory: { try self.makeHybridQwen3NextModel() },
            tokenFormula: { ($0 * 11 + 5) % 32 },
            modelIDPrefix: "qwen3next-hybrid-stable-boundary")
    }
}

/// A minimal tokenizer implementing the OPTIONAL generation-prompt-controlled
/// render: each message renders as `[90, roleToken, contentBytes..., 91]`,
/// and `addGenerationPrompt: true` appends the priming region `[90, 7, 99]`
/// -- so the `addGenerationPrompt: false` render is a strict prefix of the
/// primed render, mirroring a real ChatML-style template.
private struct StableBoundaryProbeTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    private func render(
        messages: [[String: any Sendable]], addGenerationPrompt: Bool,
        additionalContext: [String: any Sendable]?
    ) -> [Int] {
        // Mirrors a history-preserving template (Qwen3.6's
        // `preserve_thinking`): the context flag changes how HISTORY turns
        // render (an extra marker token per message), so a stable-boundary
        // baseline computed without the live render's context diverges from
        // it at the first history turn.
        let preservesHistory = additionalContext?["preserve_thinking"] as? Bool == true
        var tokens: [Int] = []
        for message in messages {
            let role = message["role"] as? String ?? ""
            let content = message["content"] as? String ?? ""
            tokens += [90, role == "user" ? 1 : 2]
            if preservesHistory {
                tokens += [77]
            }
            tokens += encode(text: content, addSpecialTokens: false)
            tokens += [91]
        }
        if addGenerationPrompt {
            tokens += [90, 7, 99]
        }
        return tokens
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        render(messages: messages, addGenerationPrompt: true, additionalContext: additionalContext)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int]? {
        render(
            messages: messages, addGenerationPrompt: addGenerationPrompt,
            additionalContext: additionalContext)
    }
}

/// A tokenizer that keeps the protocol's DEFAULT `nil` implementation of the
/// generation-prompt-controlled render -- the opt-out shape callers must
/// treat as "no stable boundary computable".
private struct OptedOutProbeTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) }
    }
}

#endif
