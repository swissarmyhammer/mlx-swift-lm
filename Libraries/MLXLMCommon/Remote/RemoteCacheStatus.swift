// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// How many positions a cache can append and still rewind them, the answer of
/// ``KVCache/isTrimmable(after:)`` for every count at once.
public enum RemoteRewindLimit: Codable, Sendable, Hashable {

    /// The cache rewinds after any number of appended positions.
    case always

    /// The cache rewinds after at most this many appended positions.
    case within(Int)

    /// The cache cannot rewind, even before it appends.
    case never

    /// The largest count a probe asks about. A cache that still rewinds after this many
    /// positions rewinds after any count.
    static let probeLimit = Int(Int32.max)

    /// Each step of the binary search halves the counts still open.
    private static let bisection = 2

    /// The answer of ``KVCache/isTrimmable(after:)`` for `positions`.
    ///
    /// - Parameter positions: The positions the cache would append.
    /// - Returns: `true` when the cache could still rewind after them.
    public func allows(_ positions: Int) -> Bool {
        guard positions >= 0 else { return false }
        switch self {
        case .always: return true
        case .within(let limit): return positions <= limit
        case .never: return false
        }
    }

    /// The limit of `caches`: the largest count each of them accepts.
    ///
    /// ``KVCache/isTrimmable(after:)`` holds up to some count and fails past it, thus a binary
    /// search finds that count.
    ///
    /// - Parameter caches: The cache of each layer.
    init(of caches: [any KVCache]) {
        let accepts = { (positions: Int) in caches.allSatisfy { $0.isTrimmable(after: positions) } }
        guard accepts(0) else {
            self = .never
            return
        }
        guard !accepts(Self.probeLimit) else {
            self = .always
            return
        }
        var (accepted, refused) = (0, Self.probeLimit)
        while refused - accepted > 1 {
            let middle = accepted + (refused - accepted) / Self.bisection
            if accepts(middle) {
                accepted = middle
            } else {
                refused = middle
            }
        }
        self = .within(accepted)
    }
}

/// The state of the caches of one handle, which every op that changes them returns.
///
/// ``RemoteKVCache`` mirrors it, so each read of the cache answers without a round trip.
public struct RemoteCacheStatus: Codable, Sendable, Hashable {

    /// The positions the cache holds: ``KVCache/offset``.
    public let offset: Int

    /// The smallest window of the caches, or nil when none is bounded: ``KVCache/maxSize``.
    public let maxSize: Int?

    /// The most positions any layer evicted: ``KVCache/evictedTokenCount``.
    public let evictedTokenCount: Int

    /// Whether every layer can trim: ``KVCache/isTrimmable``.
    public let isTrimmable: Bool

    /// The most positions every layer can rewind exactly: ``KVCache/maxTrimCount``.
    public let maxTrimCount: Int

    /// The answer of ``KVCache/isTrimmable(after:)``.
    public let rewindLimit: RemoteRewindLimit

    /// Whether some layer cannot rewind or can stop being able to, as a recurrent or a
    /// sliding-window layer does. The prompt cache then copies the cache to keep a position,
    /// rather than trim back to it.
    public let canLoseRewind: Bool

    /// The bytes the caches hold on the worker: ``KVCache/residentByteCount``.
    public let residentByteCount: Int

    /// Makes a status.
    ///
    /// - Parameters:
    ///   - offset: The positions the cache holds.
    ///   - maxSize: The smallest window of the caches, or nil.
    ///   - evictedTokenCount: The most positions any layer evicted.
    ///   - isTrimmable: Whether every layer can trim.
    ///   - maxTrimCount: The most positions every layer can rewind exactly.
    ///   - rewindLimit: How many positions the cache can append and still rewind.
    ///   - canLoseRewind: Whether some layer cannot rewind or can stop being able to.
    ///   - residentByteCount: The bytes the caches hold on the worker.
    public init(
        offset: Int, maxSize: Int?, evictedTokenCount: Int, isTrimmable: Bool, maxTrimCount: Int,
        rewindLimit: RemoteRewindLimit, canLoseRewind: Bool, residentByteCount: Int
    ) {
        self.offset = offset
        self.maxSize = maxSize
        self.evictedTokenCount = evictedTokenCount
        self.isTrimmable = isTrimmable
        self.maxTrimCount = maxTrimCount
        self.rewindLimit = rewindLimit
        self.canLoseRewind = canLoseRewind
        self.residentByteCount = residentByteCount
    }

    /// The status of `caches`, the cache of each layer of one handle, as a worker reports it.
    ///
    /// - Parameter caches: The cache of each layer, as ``LanguageModel/newCache(parameters:)``
    ///   made them.
    public init(describing caches: [any KVCache]) {
        self.init(
            offset: caches.first?.offset ?? 0,
            maxSize: caches.compactMap(\.maxSize).min(),
            evictedTokenCount: KVCacheTree.maxEvictedTokenCount(in: caches),
            isTrimmable: canTrimPromptCache(caches),
            maxTrimCount: caches.map(\.maxTrimCount).min() ?? 0,
            rewindLimit: RemoteRewindLimit(of: caches),
            canLoseRewind: PromptCacheCheckpoint.applies(to: caches),
            residentByteCount: caches.reduce(0) { $0 + $1.residentByteCount })
    }

    /// This status with `bytes` more on the worker, which state beside the caches holds, such
    /// as the drafter state of rank 0.
    ///
    /// - Parameter bytes: The bytes to add.
    /// - Returns: The status.
    func addingResidentBytes(_ bytes: Int) -> RemoteCacheStatus {
        RemoteCacheStatus(
            offset: offset, maxSize: maxSize, evictedTokenCount: evictedTokenCount,
            isTrimmable: isTrimmable, maxTrimCount: maxTrimCount, rewindLimit: rewindLimit,
            canLoseRewind: canLoseRewind, residentByteCount: residentByteCount + bytes)
    }
}

/// The element type of ``RemoteLogits``.
public enum RemoteLogitsElementType: String, Codable, Sendable, CaseIterable {
    /// Four-byte IEEE floats.
    case float32

    /// Two-byte IEEE floats.
    case float16

    /// Two-byte brain floats.
    case bfloat16

    /// The MLX element type.
    var dtype: DType {
        switch self {
        case .float32: .float32
        case .float16: .float16
        case .bfloat16: .bfloat16
        }
    }

    /// The type of `dtype`, which logits of any other type convert to.
    init(_ dtype: DType) {
        switch dtype {
        case .float16: self = .float16
        case .bfloat16: self = .bfloat16
        default: self = .float32
        }
    }
}

/// The logits of a forward, as the bytes of a row-major array.
///
/// The bytes keep the element type the model computed, so the app samples from exactly the
/// values a local model would give it.
public struct RemoteLogits: Codable, Sendable, Hashable {

    /// The shape of the logits, `[1, positions, vocabulary]`.
    public let shape: [Int]

    /// The element type of ``bytes``.
    public let elementType: RemoteLogitsElementType

    /// The elements, row-major and contiguous.
    public let bytes: Data

    /// Makes logits of their bytes.
    ///
    /// - Parameters:
    ///   - shape: The shape of the logits.
    ///   - elementType: The element type of `bytes`.
    ///   - bytes: The elements, row-major and contiguous.
    public init(shape: [Int], elementType: RemoteLogitsElementType, bytes: Data) {
        self.shape = shape
        self.elementType = elementType
        self.bytes = bytes
    }

    /// The logits of `array`, as a worker sends them. It evaluates `array`.
    ///
    /// An element type other than float32, float16 or bfloat16 converts to float32.
    ///
    /// - Parameter array: The logits of a forward.
    public init(_ array: MLXArray) {
        let elementType = RemoteLogitsElementType(array.dtype)
        let converted = array.dtype == elementType.dtype ? array : array.asType(elementType.dtype)
        self.init(
            shape: converted.shape, elementType: elementType,
            bytes: converted.asData(access: .copy).data)
    }

    /// The logits as an array, as the app samples them.
    ///
    /// - Returns: The array.
    /// - Throws: ``RemoteModelProtocolError/logitsByteCount(expected:found:)`` when the bytes
    ///   do not fill the shape.
    public func array() throws -> MLXArray {
        let expected = shape.reduce(1, *) * elementType.dtype.size
        guard bytes.count == expected else {
            throw RemoteModelProtocolError.logitsByteCount(expected: expected, found: bytes.count)
        }
        return MLXArray(bytes, shape, dtype: elementType.dtype)
    }
}
