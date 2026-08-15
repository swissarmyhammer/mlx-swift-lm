// Copyright © 2026 Apple Inc.
//
// How far back the DeepSeek-V4 prompt cache rewinds.
//
// `ChatSession` reuses a prompt cache with two rules. `ExtendCachedPrefixRule`
// feeds only the new tail, and it fires only when the new prompt starts with
// every token the cache holds. When one token breaks that prefix, the
// fall-back is `RewindToCommonPrefixRule`, which takes the cache back to the
// last token the two share. That second rule needs `canTrimPromptCache`, and
// a cache that answers false sends the turn to `.rebuild`, which feeds the
// whole prompt again.
//
// This file measures the second rule, and it needs NO weights. A cache
// reports whether it rewinds from its own position, thus a key of the right
// shape moves that position exactly as a key of a loaded model does.
//
// THE NUMBERS BELOW ARE A BASELINE, NOT A WISH.
//
// `DeepSeekV4Model.newCache` gives a `RotatingKVCache(maxSize: slidingWindow)`
// to a layer with no compressor and a `DeepSeekV4Cache` to the rest, and BOTH
// stop rewinding at the same position:
//
//   * `RotatingKVCache.isTrimmable(after:)` is
//     `offset + positions < maxCacheSize`. Past the window the ring has
//     wrapped, thus the keys a rewind would need are overwritten and gone.
//   * `DeepSeekV4Cache.isTrimmable(after:)` adds
//     `DeepSeekV4ChunkCache.holdsEveryRewindableRow`, and
//     `DeepSeekV4ChunkCache` drops its raw rows at the same window to keep its
//     memory bounded.
//
// Thus a DeepSeek-V4 conversation of more than `slidingWindow` tokens has NO
// rewind, and every turn whose prefix breaks pays the whole prefill again.
// Card ^mscrreq holds that measurement. When a correction gives this cache a
// rewind, the second test below fails on purpose: that is the signal to
// invert it and record the new behavior.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4PromptCacheRewindTests {

    /// The number of sequences the key/value fixture carries.
    private static let batchSize = 1

    /// The number of tokens that stands past the sliding window of the
    /// checkpoint. An agentic transcript reaches several thousand, thus any
    /// number past the window measures the same thing.
    private static let tokenCountPastTheWindow = 200

    /// Builds the empty prompt cache of the synthetic checkpoint.
    ///
    /// - Returns: the configuration, and one cache for each decoder layer.
    private static func newCache() throws -> (DeepSeekV4Configuration, [KVCache]) {
        let configuration = try DeepSeekV4SyntheticCheckpoint.configuration()
        let caches = try DeepSeekV4Model(configuration).newCache(parameters: nil)
        return (configuration, caches)
    }

    /// Feeds one block of keys and values to the cache of every layer.
    ///
    /// A rewind reads the POSITION of a cache and nothing else, thus a fixture
    /// of the right shape takes each cache to the position a prompt of the
    /// same length takes it to.
    ///
    /// - Parameters:
    ///   - caches: the cache of each decoder layer.
    ///   - configuration: the configuration of the checkpoint.
    ///   - tokenCount: the number of tokens to feed.
    private static func feed(
        _ caches: [KVCache], of configuration: DeepSeekV4Configuration, tokenCount: Int
    ) {
        let keyValues = MLXArray.zeros([
            batchSize, configuration.numKeyValueHeads, tokenCount, configuration.headDim,
        ])
        for cache in caches {
            _ = cache.update(keys: keyValues, values: keyValues)
        }
    }

    /// A cache that has not filled its window still holds every key, thus it
    /// rewinds. This is the premise of the second test: the rewind is not
    /// missing everywhere, it stops at one position.
    @Test func aPromptCacheInsideTheSlidingWindowRewinds() throws {
        let (configuration, caches) = try Self.newCache()
        #expect(canTrimPromptCache(caches), "the premise: an empty prompt cache rewinds")

        // `RotatingKVCache.isTrimmable` is `offset < maxCacheSize`, thus the
        // last position that still rewinds is one token inside the window.
        let tokenCount = configuration.slidingWindow - 1
        Self.feed(caches, of: configuration, tokenCount: tokenCount)

        #expect(caches.allSatisfy { $0.offset == tokenCount })
        #expect(
            canTrimPromptCache(caches),
            """
            a prompt cache of \(tokenCount) tokens stands inside the \
            \(configuration.slidingWindow)-token window, thus every key a rewind needs is \
            still here and RewindToCommonPrefixRule can rescue a broken prefix.
            """)
    }

    /// The measurement of card ^mscrreq: past the window the rewind is gone,
    /// thus `RewindToCommonPrefixRule` answers `.rebuild` and the turn feeds
    /// the whole prompt again.
    @Test func aPromptCachePastTheSlidingWindowNoLongerRewinds() throws {
        let (configuration, caches) = try Self.newCache()
        Self.feed(caches, of: configuration, tokenCount: Self.tokenCountPastTheWindow)

        #expect(caches.allSatisfy { $0.offset == Self.tokenCountPastTheWindow })
        #expect(
            !canTrimPromptCache(caches),
            """
            a prompt cache of \(Self.tokenCountPastTheWindow) tokens rewound, and it did not \
            rewind when card ^mscrreq measured it at the \(configuration.slidingWindow)-token \
            window. The fall-back of a broken prefix works again -- invert this expectation \
            and record the new behavior.
            """)
    }
}
