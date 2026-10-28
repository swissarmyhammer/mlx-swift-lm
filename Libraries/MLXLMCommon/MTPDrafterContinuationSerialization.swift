// Copyright © 2026 Apple Inc.

import Foundation
import MLX

extension MTPDrafterContinuation {

    /// The prefix of every key of ``arrays()`` and ``metadata()``.
    ///
    /// A prompt cache file keeps the continuation beside the arrays and the metadata of its main
    /// caches. The keys of the main caches never start with this prefix, so a reader removes the
    /// keys that do before it restores the main caches, and hands them to
    /// ``init(arrays:metadata:drafter:)``.
    public static let reservedKeyPrefix = "__mlx_lm_mtp_drafter_"

    /// The version of the serialized form that ``metadata()`` records and
    /// ``init(arrays:metadata:drafter:)`` reads.
    static let serializationFormatVersion = "1"

    /// The rank of the boundary hidden row: `[B, 1, hidden]`.
    static let boundaryHiddenRank = 3

    /// The keys of the serialized form.
    enum SerializedKey {

        /// The metadata key of the format version.
        static let formatVersion = reservedKeyPrefix + "version"

        /// The metadata key of the drafter identity.
        static let drafterID = reservedKeyPrefix + "drafter_id"

        /// The metadata key of ``MTPDrafterState/cacheStartPosition``.
        static let cacheStartPosition = reservedKeyPrefix + "cache_start_position"

        /// The metadata key of ``MTPDrafterState/nextPosition``.
        static let nextPosition = reservedKeyPrefix + "next_position"

        /// The metadata key of ``MTPDrafterState/proposalAppended``.
        static let proposalAppended = reservedKeyPrefix + "proposal_appended"

        /// The metadata key of the main token count the continuation was made at.
        static let targetTokenCount = reservedKeyPrefix + "target_token_count"

        /// The array key of the boundary hidden row.
        static let boundaryHidden = reservedKeyPrefix + "boundary_hidden"

        /// The array key of ``MTPDrafterState/seedToken``.
        static let seedToken = reservedKeyPrefix + "seed_token"

        /// The array key of ``MTPDrafterState/seedHidden``.
        static let seedHidden = reservedKeyPrefix + "seed_hidden"

        /// The prefix of the prompt cache keys of the drafter caches. The prompt cache layout
        /// records the class of each cache (simple or rotating) and its configuration (the window
        /// of a rotating cache) beside its arrays.
        static let cachePrefix = reservedKeyPrefix + "cache."
    }

    /// The arrays of this continuation: the drafter caches in the prompt cache layout, the
    /// boundary hidden row, and the seed proposal when there is one. Each key starts with
    /// ``reservedKeyPrefix``.
    ///
    /// - Returns: The arrays, as new handles that no drafter cache writes.
    /// - Throws: ``KVCacheError`` when a drafter cache class cannot be saved to a prompt cache.
    public func arrays() throws -> [String: MLXArray] {
        var arrays = Self.prefixed(try preparePromptCacheSave(cache: state.cache).arrays)
        arrays[SerializedKey.boundaryHidden] = targetBoundaryHidden
        arrays[SerializedKey.seedToken] = state.seedToken
        arrays[SerializedKey.seedHidden] = state.seedHidden
        return arrays
    }

    /// The metadata of this continuation: the format version, the drafter identity, the
    /// positions, and the prompt cache metadata of the drafter caches. Each key starts with
    /// ``reservedKeyPrefix``.
    ///
    /// - Returns: The metadata.
    /// - Throws: ``KVCacheError`` when a drafter cache class cannot be saved to a prompt cache.
    public func metadata() throws -> [String: String] {
        Self.prefixed(try preparePromptCacheSave(cache: state.cache).metadata).merging([
            SerializedKey.formatVersion: Self.serializationFormatVersion,
            SerializedKey.drafterID: drafterID.rawValue,
            SerializedKey.cacheStartPosition: String(state.cacheStartPosition),
            SerializedKey.nextPosition: String(state.nextPosition),
            SerializedKey.proposalAppended: String(state.proposalAppended),
            SerializedKey.targetTokenCount: String(targetProcessedTokenCount),
        ]) { _, recorded in recorded }
    }

    /// Restores a continuation from its ``arrays()`` and ``metadata()``.
    ///
    /// The drafter caches restore into the fresh caches of `drafter`, which checks the number of
    /// layers, the class of each cache and its configuration. The restored continuation resumes
    /// a stream the way the original one does.
    ///
    /// - Parameters:
    ///   - arrays: The arrays of the continuation. Keys without ``reservedKeyPrefix`` are ignored.
    ///   - metadata: The metadata of the continuation. Keys without ``reservedKeyPrefix`` are
    ///     ignored.
    ///   - drafter: The drafter that resumes the continuation.
    /// - Throws: ``KVCacheError`` for an unknown format version, a continuation of another
    ///   drafter identity, drafter caches that do not fit the caches of `drafter`, positions that
    ///   do not match each other or the entries of the drafter caches, or a missing or malformed
    ///   boundary hidden row.
    public init(
        arrays: [String: MLXArray], metadata: [String: String],
        drafter: any ResumableMTPDrafterModel
    ) throws {
        guard metadata[SerializedKey.formatVersion] == Self.serializationFormatVersion else {
            throw KVCacheError(message: "Unsupported MTP drafter continuation format")
        }
        let drafterID = drafter.drafterID
        guard metadata[SerializedKey.drafterID] == drafterID.rawValue else {
            throw KVCacheError(
                message:
                    "The MTP drafter continuation belongs to \(metadata[SerializedKey.drafterID] ?? "no drafter"), not \(drafterID)"
            )
        }
        let targetBoundaryHidden = try Self.boundaryHidden(in: arrays)
        let snapshot = try restorePromptCacheSnapshot(
            arrays: Self.cacheEntries(of: arrays), metadata: Self.cacheEntries(of: metadata),
            into: drafter.makeState(parameters: nil).cache)
        let state = try Self.drafterState(cache: snapshot.cache, arrays: arrays, metadata: metadata)
        eval([targetBoundaryHidden] + [state.seedToken, state.seedHidden].compactMap { $0 })
        self.init(
            state: state, targetBoundaryHidden: targetBoundaryHidden,
            targetProcessedTokenCount: try Self.position(
                SerializedKey.targetTokenCount, in: metadata),
            drafterID: drafterID)
    }

    /// The serialized boundary hidden row.
    ///
    /// - Parameter arrays: The serialized arrays.
    /// - Returns: The row, shape `[B, 1, hidden]`.
    /// - Throws: ``KVCacheError`` when the row is missing or has another shape.
    private static func boundaryHidden(in arrays: [String: MLXArray]) throws -> MLXArray {
        guard let row = arrays[SerializedKey.boundaryHidden], row.ndim == boundaryHiddenRank,
            row.dim(1) == 1
        else {
            throw KVCacheError(message: "The MTP drafter continuation has no boundary hidden row")
        }
        return row
    }

    /// The drafter state the serialized positions and seed proposal describe over the restored
    /// drafter caches.
    ///
    /// - Parameters:
    ///   - cache: The restored drafter caches.
    ///   - arrays: The serialized arrays.
    ///   - metadata: The serialized metadata.
    /// - Returns: The state.
    /// - Throws: ``KVCacheError`` when the positions do not match each other or the entries of
    ///   the caches, or the seed proposal is only half present.
    private static func drafterState(
        cache: [KVCache], arrays: [String: MLXArray], metadata: [String: String]
    ) throws -> MTPDrafterState {
        let seedToken = arrays[SerializedKey.seedToken]
        let seedHidden = arrays[SerializedKey.seedHidden]
        guard (seedToken == nil) == (seedHidden == nil) else {
            throw KVCacheError(message: "The MTP drafter continuation has half a seed proposal")
        }
        let state = MTPDrafterState(
            cache: cache,
            cacheStartPosition: try position(SerializedKey.cacheStartPosition, in: metadata),
            nextPosition: try position(SerializedKey.nextPosition, in: metadata),
            seedToken: seedToken, seedHidden: seedHidden,
            proposalAppended: try position(SerializedKey.proposalAppended, in: metadata))
        guard state.entryCount >= 0, state.proposalAppended <= state.entryCount, state.isAligned
        else {
            throw KVCacheError(
                message:
                    "The MTP drafter continuation positions \(state.cacheStartPosition)..<\(state.nextPosition) do not match its drafter caches"
            )
        }
        return state
    }

    /// The non-negative integer the metadata records under `key`.
    ///
    /// - Parameters:
    ///   - key: The metadata key.
    ///   - metadata: The serialized metadata.
    /// - Returns: The integer.
    /// - Throws: ``KVCacheError`` when the value is missing, not an integer, or negative.
    private static func position(_ key: String, in metadata: [String: String]) throws -> Int {
        guard let value = metadata[key].flatMap({ Int($0) }), value >= 0 else {
            throw KVCacheError(message: "The MTP drafter continuation has no valid \(key)")
        }
        return value
    }

    /// The entries of `values` with ``SerializedKey/cachePrefix`` added to each key.
    private static func prefixed<Value>(_ values: [String: Value]) -> [String: Value] {
        Dictionary(
            uniqueKeysWithValues: values.map { (SerializedKey.cachePrefix + $0.key, $0.value) })
    }

    /// The entries of `values` whose key starts with ``SerializedKey/cachePrefix``, with that
    /// prefix removed: the prompt cache of the drafter caches.
    private static func cacheEntries<Value>(of values: [String: Value]) -> [String: Value] {
        Dictionary(
            uniqueKeysWithValues: values.compactMap { key, value in
                key.hasPrefix(SerializedKey.cachePrefix)
                    ? (String(key.dropFirst(SerializedKey.cachePrefix.count)), value) : nil
            })
    }
}
