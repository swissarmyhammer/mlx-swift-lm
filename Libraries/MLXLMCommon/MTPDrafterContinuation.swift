// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The identity of an MTP drafter, which an ``MTPDrafterContinuation`` records so that only a
/// drafter of the same identity resumes it.
///
/// Two drafters of one identity build the same cache layers and read them the same way. The
/// identity does not cover the weights: it names the drafter type and its configuration.
public struct MTPDrafterID: Hashable, Sendable, CustomStringConvertible {

    /// The text of the identity, as a serialized continuation records it.
    public let rawValue: String

    /// Makes an identity from its text.
    ///
    /// - Parameter rawValue: The text of the identity.
    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    /// The identity of a drafter of `drafterType` built from `configuration`: the fully qualified
    /// type name and a hash of the JSON of `configuration` with sorted keys.
    ///
    /// The JSON encoder refuses a non-finite floating-point value. The hash then reads the
    /// description of `configuration`, which is the same in every process for the same values.
    ///
    /// - Parameters:
    ///   - drafterType: The type of the drafter.
    ///   - configuration: The configuration the drafter was built from.
    public init(drafterType: Any.Type, configuration: some Encodable) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes =
            (try? encoder.encode(configuration)) ?? Data(String(describing: configuration).utf8)
        self.init("\(String(reflecting: drafterType))#\(FNV1aHash.hexadecimalDigest(of: bytes))")
    }

    /// The text of the identity.
    public var description: String { rawValue }
}

/// The 64-bit FNV-1a hash, which gives the same digest on every platform and in every process.
private enum FNV1aHash {

    /// The offset basis of the 64-bit FNV-1a hash.
    static let offsetBasis: UInt64 = 0xcbf2_9ce4_8422_2325

    /// The prime of the 64-bit FNV-1a hash.
    static let prime: UInt64 = 0x0000_0100_0000_01b3

    /// The radix of the digest text.
    static let digestRadix = 16

    /// The hash of `bytes`, as hexadecimal text.
    ///
    /// - Parameter bytes: The bytes to hash.
    /// - Returns: The digest.
    static func hexadecimalDigest(of bytes: Data) -> String {
        let hash = bytes.reduce(offsetBasis) { ($0 ^ UInt64($1)) &* prime }
        return String(hash, radix: digestRadix)
    }
}

/// Iterator-owned MTP continuation carried atomically with a ChatSession's
/// target cache, target model state, and transcript.
///
/// A caller of ``generateTokensTask(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:components:wiredMemoryTicket:)``
/// receives one when the generation ends, and hands it to the next generation
/// that continues the same main cache. Its drafter state is opaque to callers outside
/// the package. A prompt cache keeps a continuation beside its main caches: ``copy()`` forks it,
/// ``byteCount`` measures it, and ``arrays()``, ``metadata()`` and
/// ``init(arrays:metadata:drafter:)`` serialize it.
///
/// The drafter state cannot be trimmed by itself: entry `j` needs the target hidden row of
/// position `j`, and no cache keeps old hidden rows. A continuation is thus aligned with a main
/// cache (see ``isAligned(withMainTokenCount:drafterID:)``), or a generation drops it and
/// cold-starts the drafter.
///
/// `targetProcessedTokenCount` is only an alignment witness. The authoritative
/// position remains `KVCacheStorage.processedTokenCount`.
// swiftlint:disable:next no_unchecked_sendable Every array is evaluated before a value is handed over (the generation evaluates the continuation it gives back, and copy() and the deserializing init evaluate theirs), and no other value holds its drafter caches; the value then passes to one generation at a time, which owns the drafter caches until it ends.
public struct MTPDrafterContinuation: @unchecked Sendable {
    package var state: MTPDrafterState
    package let targetBoundaryHidden: MLXArray
    package let targetProcessedTokenCount: Int

    /// The identity of the drafter whose state this is.
    package let drafterID: MTPDrafterID

    package init(
        state: consuming MTPDrafterState,
        targetBoundaryHidden: MLXArray,
        targetProcessedTokenCount: Int,
        drafterID: MTPDrafterID
    ) {
        self.state = consume state
        self.targetBoundaryHidden = targetBoundaryHidden
        self.targetProcessedTokenCount = targetProcessedTokenCount
        self.drafterID = drafterID
    }

    /// A copy that shares no drafter cache with this continuation, so a generation that
    /// continues one leaves the other as it is.
    ///
    /// The arrays themselves are shared: MLX does not write into an array that another array
    /// still holds.
    ///
    /// - Returns: The copy, with each array evaluated.
    public func copy() -> MTPDrafterContinuation {
        var copied = self
        copied.state.cache = state.cache.map { $0.copy() }
        eval(copied.state.cache.flatMap(\.state))
        return copied
    }

    /// A copy that holds each array in a buffer of its own, sized to the array, so that its
    /// ``byteCount`` is what it keeps alive. A prompt cache checkpoint stores this form.
    ///
    /// A ``copy()`` of a `KVCacheSimple` layer views the rows the live layer holds, so it keeps
    /// the whole live buffer (up to 255 rows more) alive once the live layer grows into a new
    /// buffer, and the boundary row of a prefill feed views the hidden rows of its whole chunk.
    /// Both undercount, and while the live continuation shares the arrays both count them. This
    /// copy copies each such view into a buffer of its own. MLX keeps a view shared when its
    /// buffer is at most 16 KiB larger than the view, thus the count is exact up to that slack
    /// for each array. A drafter cache layer of another type is copied as
    /// ``KVCache/copy()`` copies it.
    ///
    /// - Returns: The copy, with each array evaluated.
    package func compacted() -> MTPDrafterContinuation {
        var compactedState = state
        compactedState.cache = state.cache.map(Self.compactedCopy(of:))
        compactedState.seedToken = state.seedToken.map { contiguous($0) }
        compactedState.seedHidden = state.seedHidden.map { contiguous($0) }
        let compacted = MTPDrafterContinuation(
            state: compactedState, targetBoundaryHidden: contiguous(targetBoundaryHidden),
            targetProcessedTokenCount: targetProcessedTokenCount, drafterID: drafterID)
        eval(compacted.heldArrays)
        return compacted
    }

    /// A copy of a drafter cache layer whose arrays hold only the rows the layer holds.
    ///
    /// - Parameter cache: A drafter cache layer.
    /// - Returns: The compacted copy of a `KVCacheSimple`, or ``KVCache/copy()`` of any other
    ///   layer, whose state arrays need not name the rows the layer holds.
    private static func compactedCopy(of cache: KVCache) -> KVCache {
        let copied = cache.copy()
        guard let simple = copied as? KVCacheSimple, type(of: simple) == KVCacheSimple.self,
            !simple.state.isEmpty
        else { return copied }
        simple.state = simple.state.map { contiguous($0) }
        return simple
    }

    /// Every array this continuation holds: the drafter cache arrays, the boundary hidden row,
    /// and the seed proposal.
    private var heldArrays: [MLXArray] {
        state.cache.flatMap(\.state) + [targetBoundaryHidden]
            + [state.seedToken, state.seedHidden].compactMap { $0 }
    }

    /// The bytes this continuation holds in memory: the buffers of the drafter caches (see
    /// ``KVCache/residentByteCount``), the boundary hidden row, and the seed proposal.
    ///
    /// The count reads only shapes and element types, thus it evaluates nothing.
    public var byteCount: Int {
        state.cache.reduce(0) { $0 + $1.residentByteCount }
            + ([targetBoundaryHidden] + [state.seedToken, state.seedHidden].compactMap { $0 })
            .totalByteCount
    }

    /// Whether a generation that continues a main cache of `mainTokenCount` tokens with the
    /// drafter `drafterID` can resume this continuation.
    ///
    /// True only when the drafter ends one position before the main cache, the boundary hidden
    /// row of the main cache's last position is present, every drafter cache holds the entries
    /// the positions name, and the continuation was made by that drafter.
    ///
    /// - Parameters:
    ///   - mainTokenCount: The tokens the main cache holds.
    ///   - drafterID: The identity of the drafter of the generation.
    /// - Returns: `true` when the continuation is aligned.
    public func isAligned(withMainTokenCount mainTokenCount: Int, drafterID: MTPDrafterID) -> Bool {
        drafterID == self.drafterID && targetProcessedTokenCount == mainTokenCount
            && state.nextPosition == mainTokenCount - 1 && state.proposalAppended == 0
            && targetBoundaryHidden.dim(1) == 1 && state.isAligned
    }

    /// The boundary hidden row a generation that continues a main cache of `mainTokenCount`
    /// tokens with the drafter `drafterID` resumes from, or `nil` when the continuation is not
    /// aligned with that cache (see ``isAligned(withMainTokenCount:drafterID:)``) and the
    /// drafter must cold-start.
    ///
    /// - Parameters:
    ///   - mainTokenCount: The tokens the main cache holds.
    ///   - drafterID: The identity of the drafter of the generation.
    /// - Returns: The boundary hidden row, shape `[B, 1, hidden]`, of an aligned continuation.
    public func alignedBoundary(
        withMainTokenCount mainTokenCount: Int, drafterID: MTPDrafterID
    ) -> MLXArray? {
        isAligned(withMainTokenCount: mainTokenCount, drafterID: drafterID)
            ? targetBoundaryHidden : nil
    }
}
