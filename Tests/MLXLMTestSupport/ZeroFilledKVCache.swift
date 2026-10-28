// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Attention caches that hold zero keys and values, for tests that need a cache at a
/// position but not the numbers it holds.
public enum ZeroFilledKVCache {

    /// An attention cache that a prefill or a verify of `tokens` tokens left behind: one
    /// sequence of zero keys and values, written with one `update(keys:values:)`.
    ///
    /// - Parameters:
    ///   - tokens: The positions the cache holds; its offset.
    ///   - heads: The key and value heads of each position.
    ///   - headDimension: The width of each head.
    /// - Returns: The cache.
    public static func make(tokens: Int, heads: Int = 1, headDimension: Int = 1)
        -> KVCacheSimple
    {
        let cache = KVCacheSimple()
        let entries = MLXArray.zeros([1, heads, tokens, headDimension])
        _ = cache.update(keys: entries, values: entries)
        return cache
    }
}
