// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    /// The class name of the cache in a prompt cache file.
    public override class var serializedClassName: String? { "CacheList" }

    /// Rebuilds the list and its children from the values that a prompt cache file saved.
    public override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        return try CacheList.fromState(state: state, metaState: metaState)
    }

    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    /// Internal initializer for reconstruction from deserialized children.
    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    /// The number of tokens that the list holds: the largest offset of its children.
    ///
    /// Each child moves its own offset when the model feeds tokens through it. The list has no
    /// tokens of its own, thus it reads the offset from its children. The prompt cache compares
    /// the offset of each top-level cache with the length of its token ledger, thus a list whose
    /// offset stays at 0 can never serve a next prompt. An empty list holds no tokens and gives 0.
    ///
    /// The offset cannot be set, because it comes from the children. Set the offset of each child
    /// that does not keep its offset in its state.
    public override var offset: Int {
        get { caches.map(\.offset).max() ?? 0 }
        set {
            preconditionFailure(
                "The offset of a CacheList comes from its children. Set the offset of each child.")
        }
    }

    /// The sum of ``KVCache/residentByteCount`` over the children, so that each
    /// child counts with its own rule.
    public override var residentByteCount: Int {
        caches.reduce(0) { $0 + $1.residentByteCount }
    }

    public override var evictedTokenCount: Int {
        caches.map(\.evictedTokenCount).max() ?? 0
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    /// Recursively apply a transformation to every non-composite child cache.
    ///
    /// `CacheList` children are descended into; any other cache is passed to
    /// `transform` and replaced by the returned value. This is the primitive
    /// used by dynamic cache quantization and other cache-wide rewrites for
    /// models with hybrid attention/recurrent caches (e.g. Falcon-H1).
    public func mapChildren(_ transform: (KVCache) -> KVCache) {
        caches = caches.map { child in
            if let list = child as? CacheList {
                list.mapChildren(transform)
                return list
            }
            return transform(child)
        }
    }

    /// Rewrite every non-composite child while preserving its stable tree path.
    func rewriteLeaves(
        path: [Int],
        using transform: (KVCacheLeaf) -> KVCache
    ) {
        caches = caches.enumerated().map { index, child in
            let childPath = path + [index]
            if let list = child as? CacheList {
                list.rewriteLeaves(path: childPath, using: transform)
                return list
            }
            if child is any AuxiliaryKVCache {
                return child
            }
            let leaf = KVCacheLeaf(path: childPath, cache: child)
            return transform(leaf)
        }
    }

    public override func prepare(lengths: [Int]?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func prepare(lengths: MLXArray?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func finalize() {
        caches.forEach { $0.finalize() }
    }

    public override var isTrimmable: Bool {
        isTrimmable(after: 0)
    }

    public override func isTrimmable(after positions: Int) -> Bool {
        caches.allSatisfy { $0.isTrimmable(after: positions) }
    }

    public override var maxTrimCount: Int {
        caches.map(\.maxTrimCount).min() ?? 0
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    /// Internal accessor for child caches (used by serialization and policy reporting).
    internal var children: [KVCache] { caches }

    // MARK: - Serialization

    /// metaState format: [childCount, (className, stateCount, metaStateCount, ...metaState)*]
    ///
    /// Like Python's CacheList.meta_state which returns [child_class_names, child_meta_states],
    /// but flattened for Swift's [String] format.
    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className =
                    type(of: cache).serializedClassName ?? String(describing: type(of: cache))
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList.metaState should not be set directly. Use CacheList.fromState() instead")
        }
    }

    /// Reconstruct a CacheList from flattened state + metaState, like Python's from_state()
    internal static func fromState(state: [MLXArray], metaState: [String]) throws -> CacheList {
        let children = try savedChildren(state: state, metaState: metaState).map {
            try restoreCacheFromMetaState(
                className: $0.className, state: $0.state, metaState: $0.metaState)
        }
        return CacheList(caches: children)
    }

    /// Replaces the children after a prompt cache restore. A restore can give a child another
    /// cache class, for example a `QuantizedKVCache` in place of a `KVCacheSimple`.
    ///
    /// - Parameter children: The new children, one for each old child, in the same order.
    internal func replaceChildren(with children: [KVCache]) {
        precondition(
            children.count == caches.count,
            "a CacheList has \(caches.count) children and the restore gave \(children.count)")
        caches = children
    }
}

extension CacheList {
    /// The number of meta-state values in front of each child: its class name, its array
    /// count and its meta-state count.
    private static let childHeaderCount = 3

    /// The place of the array count in the header of a child.
    private static let childStateCountOffset = 1

    /// The place of the meta-state count in the header of a child.
    private static let childMetaStateCountOffset = 2

    /// Splits the flat `state` and `metaState` of a saved `CacheList` into its children.
    ///
    /// The format is `[childCount, (className, stateCount, metaStateCount, metaState...)*]`.
    /// Every count is checked against the values that are there, and every value must belong
    /// to a child, thus no slice can start past an end.
    ///
    /// - Parameters:
    ///   - state: The saved arrays of the whole list.
    ///   - metaState: The saved meta state of the whole list.
    /// - Returns: The children, in order.
    /// - Throws: ``KVCacheError`` when a count does not fit the saved values.
    static func savedChildren(
        state: [MLXArray], metaState: [String]
    ) throws -> [SavedPromptCacheLayer] {
        guard let childCount = metaState.first.flatMap({ Int($0) }), childCount >= 0 else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }
        var children: [SavedPromptCacheLayer] = []
        var metaIndex = 1
        var stateIndex = 0
        for _ in 0 ..< childCount {
            let child = try savedChild(
                state: state, metaState: metaState, metaIndex: metaIndex, stateIndex: stateIndex)
            children.append(child)
            metaIndex += childHeaderCount + child.metaState.count
            stateIndex += child.state.count
        }
        guard metaIndex == metaState.count, stateIndex == state.count else {
            throw KVCacheError(
                message: "Corrupt prompt cache: CacheList holds values that no child owns.")
        }
        return children
    }

    /// Reads one child of a saved `CacheList`.
    ///
    /// - Parameters:
    ///   - state: The saved arrays of the whole list.
    ///   - metaState: The saved meta state of the whole list.
    ///   - metaIndex: The place of the header of the child in `metaState`.
    ///   - stateIndex: The place of the first array of the child in `state`.
    /// - Returns: The child.
    /// - Throws: ``KVCacheError`` when the header is truncated or a count is out of range.
    private static func savedChild(
        state: [MLXArray], metaState: [String], metaIndex: Int, stateIndex: Int
    ) throws -> SavedPromptCacheLayer {
        let metaStart = metaIndex + childHeaderCount
        guard metaStart <= metaState.count else {
            throw KVCacheError(message: "CacheList metaState truncated")
        }
        guard let stateCount = Int(metaState[metaIndex + childStateCountOffset]),
            let metaCount = Int(metaState[metaIndex + childMetaStateCountOffset]),
            stateCount >= 0, metaCount >= 0,
            stateIndex + stateCount <= state.count, metaStart + metaCount <= metaState.count
        else {
            throw KVCacheError(message: "CacheList: invalid array or metaState count for child")
        }
        return SavedPromptCacheLayer(
            className: metaState[metaIndex],
            state: Array(state[stateIndex ..< stateIndex + stateCount]),
            metaState: Array(metaState[metaStart ..< metaStart + metaCount]))
    }
}
