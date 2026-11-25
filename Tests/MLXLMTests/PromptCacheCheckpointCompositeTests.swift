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

    @Test("an Inkling 2 checkpoint restores after the sliding window wraps")
    func anInklingCheckpointRestoresAfterTheSlidingWindowWraps() throws {
        let model = try Self.inklingTarget()
        let caches = try model.newCache(parameters: nil)
        let tokens = (0 ..< Self.checkpointTokenCount + Self.laterTokenCount).map {
            $0 % Self.vocabularySize
        }
        let prefill = PrefillParameters(stepSize: Self.checkpointTokenCount)
        _ = try PromptCacheCheckpoint.prefill(
            Array(tokens.prefix(Self.checkpointTokenCount)), into: caches, state: nil,
            model: model, prefill: prefill)
        let checkpoint = try #require(
            PromptCacheCheckpoint(
                caches: caches, tokens: Array(tokens.prefix(Self.checkpointTokenCount)),
                state: nil))
        _ = try PromptCacheCheckpoint.prefill(
            Array(tokens.dropFirst(Self.checkpointTokenCount)), into: caches, state: nil,
            model: model, prefill: prefill)

        let restored = checkpoint.restored(onto: caches)

        #expect(
            restored?.map(\.offset)
                == Array(repeating: Self.checkpointTokenCount, count: caches.count))
    }
}
