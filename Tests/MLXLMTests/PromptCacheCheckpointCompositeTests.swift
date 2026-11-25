// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXLMCommon

/// A prompt checkpoint reads the caches inside a ``CacheList``, as Inkling 2 makes each layer
/// one: an attention cache, sliding or global, beside the states of its convolutions.
@Suite("A prompt checkpoint looks inside a composite cache", .serialized)
struct PromptCacheCheckpointCompositeTests {

    /// The window of a sliding layer of the composite caches.
    private static let window = 8

    /// The tile of a variance-normalized cache: a size the cache supports.
    private static let varianceTileSize = 32

    /// The convolution slots of each composite cache, as Inkling 2 has.
    private static let convolutionSlots = 4

    /// The verify positions of each convolution cache, as Inkling 2 sets.
    private static let verifyPositions = 16

    /// The vocabulary of the tiny Inkling 2 target.
    private static let vocabularySize = 128

    /// The tokens before the checkpoint.
    private static let checkpointTokenCount = 6

    /// The tokens after the checkpoint: more than the window, thus a sliding layer wraps.
    private static let laterTokenCount = 12

    /// A convolution cache as Inkling 2 makes one.
    private static func convolution() -> SpeculativeConvolutionCache {
        SpeculativeConvolutionCache(size: convolutionSlots, maximumVerifyPositions: verifyPositions)
    }

    /// The tiny Inkling 2 target, whose layers mix sliding and global attention.
    private static func inklingTarget() throws -> Inkling2 {
        try Inkling2TinyModels.target(
            json: Inkling2TinyModels.draftJSON(
                vocabularySize: vocabularySize, unpaddedVocabularySize: vocabularySize))
    }

    @Test("a list with a sliding layer can lose its rewind")
    func aListWithASlidingLayerCanLoseItsRewind() {
        let caches: [KVCache] = [
            CacheList(RotatingKVCache(maxSize: Self.window, keep: 0), Self.convolution())
        ]

        #expect(PromptCacheCheckpoint.applies(to: caches))
    }

    @Test("a list with a convolution can lose its rewind")
    func aListWithAConvolutionCanLoseItsRewind() {
        let caches: [KVCache] = [CacheList(KVCacheSimple(), Self.convolution())]

        #expect(PromptCacheCheckpoint.applies(to: caches))
    }

    @Test("a list of attention caches alone never loses its rewind")
    func aListOfAttentionCachesAloneNeverLosesItsRewind() {
        let caches: [KVCache] = [CacheList(KVCacheSimple(), KVCacheSimple())]

        #expect(!PromptCacheCheckpoint.applies(to: caches))
    }

    @Test("a list with a variance-normalized cache can lose its rewind")
    func aListWithAVarianceNormalizedCacheCanLoseItsRewind() {
        let caches: [KVCache] = [
            CacheList(VarianceNormalizedKVCache(tileSize: Self.varianceTileSize), KVCacheSimple())
        ]

        #expect(PromptCacheCheckpoint.applies(to: caches))
    }

    @Test("a checkpoint of a list keeps only the window of its sliding layer")
    func aCheckpointOfAListKeepsOnlyTheWindowOfItsSlidingLayer() throws {
        let tokenCount = Self.window + Self.laterTokenCount
        let listed = Self.slidingCache(fedWith: tokenCount)
        let bare = Self.slidingCache(fedWith: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let listCheckpoint = try #require(
            PromptCacheCheckpoint(
                caches: [CacheList(listed, Self.convolution())], tokens: tokens, state: nil))
        let bareCheckpoint = try #require(
            PromptCacheCheckpoint(caches: [bare], tokens: tokens, state: nil))

        #expect(listCheckpoint.byteCount == bareCheckpoint.byteCount)
    }

    /// A sliding cache fed `tokenCount` rows in one multi-token update, as a prefill feeds it.
    private static func slidingCache(fedWith tokenCount: Int) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: window, keep: 0)
        let rows = MLXArray.ones([1, 1, tokenCount, convolutionSlots])
        _ = cache.update(keys: rows, values: rows)
        return cache
    }

    @Test("the caches of an Inkling 2 model can lose their rewind")
    func theCachesOfAnInklingModelCanLoseTheirRewind() throws {
        let model = try Self.inklingTarget()

        #expect(PromptCacheCheckpoint.applies(to: try model.newCache(parameters: nil)))
    }

    /// The prompt of the Inkling 2 restore tests: the checkpoint tokens, then the later ones.
    private static let restoreTokens = (0 ..< checkpointTokenCount + laterTokenCount).map {
        $0 % vocabularySize
    }

    /// The largest logit difference a restored continuation may show beside a cold prefill of
    /// the same tokens, in float32.
    private static let restoreTolerance: Float = 1e-4

    /// The caches of `model` restored from a checkpoint of the first ``checkpointTokenCount``
    /// tokens of ``restoreTokens``, after the live caches took every later token, so the
    /// sliding window wrapped past the checkpoint.
    private static func restoredAfterWrap(_ model: Inkling2) throws -> [KVCache]? {
        let caches = try model.newCache(parameters: nil)
        let prefill = PrefillParameters(stepSize: checkpointTokenCount)
        let checkpointTokens = Array(restoreTokens.prefix(checkpointTokenCount))
        _ = try PromptCacheCheckpoint.prefill(
            checkpointTokens, into: caches, state: nil, model: model, prefill: prefill)
        let checkpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: checkpointTokens, state: nil))
        _ = try PromptCacheCheckpoint.prefill(
            Array(restoreTokens.dropFirst(checkpointTokenCount)), into: caches, state: nil,
            model: model, prefill: prefill)
        return checkpoint.restored(onto: caches)
    }

    /// The last-position logits of `tokens` run through `model` into `caches` in one forward.
    private static func lastLogits(
        of tokens: [Int], model: Inkling2, caches: [KVCache]
    ) -> MLXArray {
        let input = LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis])
        return model(input, cache: caches, state: nil).logits[0, -1].asType(.float32)
    }

    @Test("an Inkling 2 checkpoint restores after the sliding window wraps")
    func anInklingCheckpointRestoresAfterTheSlidingWindowWraps() throws {
        let model = try Self.inklingTarget()

        let restored = try Self.restoredAfterWrap(model)

        #expect(
            restored?.map(\.offset)
                == Array(repeating: Self.checkpointTokenCount, count: restored?.count ?? 0))
    }

    @Test("an Inkling 2 cache restored after the window wraps continues as a cold prefill does")
    func anInklingRestoreContinuesAsAColdPrefill() throws {
        let model = try Self.inklingTarget()
        let restored = try #require(try Self.restoredAfterWrap(model))
        let later = Array(Self.restoreTokens.dropFirst(Self.checkpointTokenCount))

        let warm = Self.lastLogits(of: later, model: model, caches: restored)
        let cold = Self.lastLogits(
            of: Self.restoreTokens, model: model, caches: try model.newCache(parameters: nil))

        #expect(allClose(warm, cold, atol: Double(Self.restoreTolerance)).item(Bool.self))
    }
}
