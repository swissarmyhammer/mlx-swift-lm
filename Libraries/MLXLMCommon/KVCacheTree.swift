// Copyright © 2026 Apple Inc.

import MLX

/// A classified leaf in a potentially nested cache topology.
struct KVCacheLeaf {
    enum Kind {
        case recurrent
        case rotating(RotatingKVCache)
        case simple(KVCacheSimple)
        case affine(QuantizedKVCache)
        case turboQuant(TurboQuantKVCache)
        case varianceNormalized(VarianceNormalizedKVCache)
        case unsupported
    }

    let path: [Int]
    let cache: KVCache

    var kind: Kind {
        switch cache {
        case is MambaCache, is ArraysCache: .recurrent
        case let cache as RotatingKVCache: .rotating(cache)
        case let cache as TurboQuantKVCache: .turboQuant(cache)
        case let cache as VarianceNormalizedKVCache: .varianceNormalized(cache)
        case let cache as QuantizedKVCache: .affine(cache)
        case let cache as KVCacheSimple: .simple(cache)
        default: .unsupported
        }
    }

    var isAttentionCache: Bool {
        if case .recurrent = kind { return false }
        return true
    }

    var isCompressed: Bool {
        switch kind {
        case .affine, .turboQuant, .varianceNormalized: true
        default: false
        }
    }

    /// Whether this leaf participates in TurboQuant boundary protection.
    ///
    /// The ranking must remain stable as simple caches are replaced by their
    /// affine or TurboQuant representations. Rotating and unsupported caches
    /// are excluded because TurboQuant cannot rewrite them; including them
    /// would shift protection away from the first and last eligible layers in
    /// mixed cache topologies.
    var participatesInTurboQuantBoundaryProtection: Bool {
        switch kind {
        case .simple, .affine, .turboQuant: true
        case .recurrent, .rotating, .varianceNormalized, .unsupported: false
        }
    }

    /// Whether this leaf can satisfy a compression strategy. `nil` identifies
    /// state that is not an attention cache and does not participate.
    func supports(
        _ configuration: KVCacheConfiguration,
        protectedPaths: Set<[Int]>
    ) -> Bool? {
        let strategy = configuration.strategy.identifier
        switch kind {
        case .recurrent:
            return nil
        case .rotating, .unsupported:
            return strategy == .fullPrecision
        case .simple(let simple):
            let state = simple.innerState()
            switch configuration.strategy.storage {
            case .fullPrecision:
                return true
            case .affine(let affine):
                return Self.supportsGroupQuantization(state, requested: affine.groupSize)
            case .turboQuant(let turbo):
                guard protectedPaths.contains(path) || turbo.keyPrecision == .affineEightBit
                else { return true }
                return Self.supportsGroupQuantization(
                    state, requested: KVCacheTree.boundaryProtectionGroupSize)
            case .varianceNormalized(let varianceNormalized):
                guard let headDims = Self.headDims(of: state) else { return true }
                return supportsVarianceNormalizedKVCache(
                    keyHeadDim: headDims.key,
                    valueHeadDim: headDims.value,
                    tileSize: varianceNormalized.tileSize)
            }
        case .affine:
            return strategy == .affine
                || (strategy == .turboQuant && protectedPaths.contains(path))
        case .turboQuant:
            return strategy == .turboQuant
        case .varianceNormalized:
            return strategy == .varianceNormalized
        }
    }

    /// The position of the keys in the inner state of a simple cache.
    private static let keyStateIndex = 0

    /// The position of the values in the inner state of a simple cache.
    private static let valueStateIndex = 1

    /// The axis of the head dimension in a `[batch, heads, tokens, headDim]` state array.
    private static let headDimAxis = 3

    /// The key and value head dimensions of a simple cache's inner state, or `nil` while the
    /// cache has not seen a token yet.
    private static func headDims(of state: [MLXArray]) -> (key: Int, value: Int)? {
        guard state.indices.contains(keyStateIndex), state.indices.contains(valueStateIndex)
        else { return nil }
        return (state[keyStateIndex].dim(headDimAxis), state[valueStateIndex].dim(headDimAxis))
    }

    /// Whether group quantization with the `requested` group size fits the head dimensions of
    /// `state`. An empty state has no shape yet and is accepted.
    private static func supportsGroupQuantization(_ state: [MLXArray], requested: Int) -> Bool {
        guard let headDims = headDims(of: state) else { return true }
        return resolvedKVQuantizationGroupSize(
            requested: requested,
            keyHeadDim: headDims.key,
            valueHeadDim: headDims.value) != nil
    }
}

/// The sole traversal and mutation boundary for nested cache topologies.
enum KVCacheTree {
    /// The affine group size of the boundary layers that TurboQuant protects with 8-bit affine.
    static let boundaryProtectionGroupSize = 64

    /// The affine bit width of the boundary layers that TurboQuant protects.
    static let boundaryProtectionBits = 8

    /// Key bit widths below this make a TurboQuant scheme fragile at its boundary layers.
    private static let fragileKeyBitsBound = 8

    /// Value bit widths at or below this make a TurboQuant scheme fragile at its boundary layers.
    private static let fragileValueBitsLimit = 2

    /// The most layers TurboQuant protects at each end of the eligible layers.
    private static let maximumProtectedLayersPerEnd = 2

    /// The ends of the eligible layers that TurboQuant protects: the first and the last.
    private static let protectedEndCount = 2

    /// Whether `cache` is a leaf: neither a ``CacheList`` of further caches nor an
    /// ``AuxiliaryKVCache``, which appends and trims alongside the leaves but is not one.
    static func isLeaf(_ cache: KVCache) -> Bool {
        !(cache is CacheList) && !(cache is any AuxiliaryKVCache)
    }

    static func leaves(in cache: [KVCache]) -> [KVCacheLeaf] {
        var leaves: [KVCacheLeaf] = []
        for (index, entry) in cache.enumerated() {
            appendLeaves(from: entry, path: [index], to: &leaves)
        }
        return leaves
    }

    /// The most tokens any leaf of `cache` evicted.
    static func maxEvictedTokenCount(in cache: [KVCache]) -> Int {
        leaves(in: cache).map(\.cache.evictedTokenCount).max() ?? 0
    }

    static func rewrite(
        _ cache: inout [KVCache],
        using transform: (KVCacheLeaf) -> KVCache
    ) {
        for index in cache.indices {
            if let list = cache[index] as? CacheList {
                list.rewriteLeaves(path: [index], using: transform)
            } else if isLeaf(cache[index]) {
                let leaf = KVCacheLeaf(path: [index], cache: cache[index])
                cache[index] = transform(leaf)
            }
        }
    }

    static func turboQuantProtectedPaths(
        in leaves: [KVCacheLeaf],
        keyBits: Int,
        valueBits: Int
    ) -> Set<[Int]> {
        let fragile =
            (keyBits > 0 && keyBits < fragileKeyBitsBound) || valueBits <= fragileValueBitsLimit
        guard fragile else { return [] }

        let eligiblePaths =
            leaves
            .filter(\.participatesInTurboQuantBoundaryProtection)
            .map(\.path)
        let boundaryCount = min(
            maximumProtectedLayersPerEnd, eligiblePaths.count / protectedEndCount)
        guard boundaryCount > 0 else { return [] }
        return Set(eligiblePaths.prefix(boundaryCount))
            .union(eligiblePaths.suffix(boundaryCount))
    }

    private static func appendLeaves(
        from cache: KVCache,
        path: [Int],
        to leaves: inout [KVCacheLeaf]
    ) {
        if let list = cache as? CacheList {
            for (index, child) in list.children.enumerated() {
                appendLeaves(from: child, path: path + [index], to: &leaves)
            }
        } else if isLeaf(cache) {
            leaves.append(KVCacheLeaf(path: path, cache: cache))
        }
    }
}
