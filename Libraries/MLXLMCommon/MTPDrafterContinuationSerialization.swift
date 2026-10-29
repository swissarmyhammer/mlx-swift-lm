// Copyright © 2026 Apple Inc.

import Foundation
import MLX

extension MTPDrafterContinuation {

    /// The prefix of every key of ``arrays()`` and ``metadata()``.
    ///
    /// A prompt cache file keeps the continuation beside the arrays and the metadata of its main
    /// caches. The keys of the main caches never start with this prefix, so a reader removes the
    /// keys that do before it restores the main caches, and hands them to
    /// ``init(arrays:metadata:drafter:parameters:)``.
    public static let reservedKeyPrefix = "__mlx_lm_mtp_drafter_"

    /// The version of the serialized form that ``metadata()`` records and
    /// ``init(arrays:metadata:drafter:parameters:)`` reads.
    static let serializationFormatVersion = "1"

    /// The axis of the batch rows of every serialized array: the boundary and the seed hidden
    /// rows, the seed token, and each array of the drafter caches.
    static let batchAxis = 0

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
    /// The drafter caches restore into the fresh caches `drafter` makes for `parameters`, which
    /// checks the number of layers, the class of each cache and its configuration: a ring
    /// restores only into a ring of its window, as the stream of `parameters` rotates its
    /// caches, and a cache of another class is refused. The boundary and the seed hidden
    /// rows must have the width of ``ResumableMTPDrafterModel/hiddenSize`` when the drafter
    /// states one. The seed hidden row, the seed token and every array of the drafter caches
    /// must have the batch of the boundary hidden row. The restored continuation resumes a
    /// stream the way the original one does.
    ///
    /// - Parameters:
    ///   - arrays: The arrays of the continuation. Keys without ``reservedKeyPrefix`` are ignored.
    ///   - metadata: The metadata of the continuation. Keys without ``reservedKeyPrefix`` are
    ///     ignored.
    ///   - drafter: The drafter that resumes the continuation.
    ///   - parameters: The generation parameters of the stream that resumes the continuation,
    ///     which decide the drafter caches (see
    ///     ``StatefulMTPDrafterModel/makeState(parameters:)``), or `nil` for the caches of a
    ///     stream without a cache configuration.
    /// - Throws: ``KVCacheError`` for an unknown format version, a continuation of another
    ///   drafter identity, drafter caches that do not fit the caches of `drafter`, positions that
    ///   do not match each other or the entries of the drafter caches, a missing boundary hidden
    ///   row, a boundary or seed hidden row of another shape or width, a seed token that is not
    ///   one token for each batch row, or a seed proposal or drafter cache of another batch.
    public init(
        arrays: [String: MLXArray], metadata: [String: String],
        drafter: any ResumableMTPDrafterModel, parameters: GenerateParameters? = nil
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
        let targetBoundaryHidden = try Self.boundaryHidden(
            in: arrays, hiddenSize: drafter.hiddenSize)
        let batchSize = targetBoundaryHidden.dim(Self.batchAxis)
        let seed = try Self.seedProposal(
            in: arrays, batchSize: batchSize, hiddenSize: drafter.hiddenSize)
        let snapshot = try restorePromptCacheSnapshot(
            arrays: Self.cacheEntries(of: arrays), metadata: Self.cacheEntries(of: metadata),
            into: drafter.makeState(parameters: parameters).cache)
        try Self.checkCacheBatch(snapshot.cache, batchSize: batchSize)
        let state = try Self.drafterState(cache: snapshot.cache, seed: seed, metadata: metadata)
        eval([targetBoundaryHidden] + [state.seedToken, state.seedHidden].compactMap { $0 })
        self.init(
            state: state, targetBoundaryHidden: targetBoundaryHidden,
            targetProcessedTokenCount: try Self.position(
                SerializedKey.targetTokenCount, in: metadata),
            drafterID: drafterID)
    }

    /// The serialized boundary hidden row.
    ///
    /// - Parameters:
    ///   - arrays: The serialized arrays.
    ///   - hiddenSize: The width the drafter reads, or `nil` when it does not state one.
    /// - Returns: The row, shape `[B, 1, hidden]`.
    /// - Throws: ``KVCacheError`` when the row is missing or has another shape or width.
    private static func boundaryHidden(
        in arrays: [String: MLXArray], hiddenSize: Int?
    ) throws -> MLXArray {
        guard let row = arrays[SerializedKey.boundaryHidden] else {
            throw KVCacheError(message: "The MTP drafter continuation has no boundary hidden row")
        }
        try checkShape(row, named: "boundary hidden row", expected: [nil, 1, hiddenSize])
        return row
    }

    /// The serialized seed proposal: one token and one hidden row for each batch row.
    ///
    /// - Parameters:
    ///   - arrays: The serialized arrays.
    ///   - batchSize: The batch of the boundary hidden row.
    ///   - hiddenSize: The width of the drafter's seed hidden row, or `nil` when the drafter does
    ///     not state one.
    /// - Returns: The seed token, shape `[B, 1]`, and the seed hidden row, shape `[B, 1, hidden]`,
    ///   or `nil` when the continuation holds no seed proposal.
    /// - Throws: ``KVCacheError`` when the seed proposal is only half present, or the token or
    ///   the hidden row has another shape, batch or width.
    private static func seedProposal(
        in arrays: [String: MLXArray], batchSize: Int, hiddenSize: Int?
    ) throws -> (token: MLXArray, hidden: MLXArray)? {
        let token = arrays[SerializedKey.seedToken]
        let hidden = arrays[SerializedKey.seedHidden]
        guard (token == nil) == (hidden == nil) else {
            throw KVCacheError(message: "The MTP drafter continuation has half a seed proposal")
        }
        guard let token, let hidden else { return nil }
        try checkShape(token, named: "seed token", expected: [batchSize, 1])
        try checkShape(hidden, named: "seed hidden row", expected: [batchSize, 1, hiddenSize])
        return (token, hidden)
    }

    /// Checks that a serialized array has the shape `expected`.
    ///
    /// - Parameters:
    ///   - array: The serialized array.
    ///   - name: The name of the array in the error message.
    ///   - expected: The size of each axis, or `nil` for an axis of any size.
    /// - Throws: ``KVCacheError`` when the array has another rank or another size on an axis.
    private static func checkShape(
        _ array: MLXArray, named name: String, expected: [Int?]
    ) throws {
        guard array.ndim == expected.count,
            zip(array.shape, expected).allSatisfy({ size, wanted in
                wanted.map { $0 == size } ?? true
            })
        else {
            let described = expected.map { $0.map(String.init) ?? "_" }.joined(separator: ", ")
            throw KVCacheError(
                message:
                    "The MTP drafter continuation's \(name) has shape \(array.shape), not [\(described)]"
            )
        }
    }

    /// Checks that every array of the restored drafter caches has `batchSize` batch rows.
    ///
    /// - Parameters:
    ///   - cache: The restored drafter caches.
    ///   - batchSize: The batch of the boundary hidden row.
    /// - Throws: ``KVCacheError`` when an array of a cache has another batch.
    private static func checkCacheBatch(_ cache: [KVCache], batchSize: Int) throws {
        for array in cache.flatMap(\.state)
        where array.ndim <= batchAxis || array.dim(batchAxis) != batchSize {
            throw KVCacheError(
                message:
                    "The MTP drafter continuation's drafter cache holds an array of shape \(array.shape), not of batch \(batchSize)"
            )
        }
    }

    /// The drafter state the serialized positions and seed proposal describe over the restored
    /// drafter caches.
    ///
    /// - Parameters:
    ///   - cache: The restored drafter caches.
    ///   - seed: The checked seed proposal, or `nil` when there is none.
    ///   - metadata: The serialized metadata.
    /// - Returns: The state.
    /// - Throws: ``KVCacheError`` when the positions do not match each other or the entries of
    ///   the caches.
    private static func drafterState(
        cache: [KVCache], seed: (token: MLXArray, hidden: MLXArray)?, metadata: [String: String]
    ) throws -> MTPDrafterState {
        let state = MTPDrafterState(
            cache: cache,
            cacheStartPosition: try position(SerializedKey.cacheStartPosition, in: metadata),
            nextPosition: try position(SerializedKey.nextPosition, in: metadata),
            seedToken: seed?.token, seedHidden: seed?.hidden,
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

/// An ``MTPDrafterContinuation`` in its serialized form: its ``MTPDrafterContinuation/arrays()``
/// and ``MTPDrafterContinuation/metadata()``, as a prompt cache file keeps them beside its main
/// caches.
///
/// Only a drafter can check and restore a continuation, and a prompt cache load has none. A
/// ``PromptCacheSnapshot`` thus carries the serialized form, and the generation that has the
/// drafter restores it with ``restored(by:parameters:)``. A restore that throws (a continuation
/// of another drafter, drafter caches of another class or window, or damaged values) leaves the
/// drafter to cold-start at the cached position.
// swiftlint:disable:next no_unchecked_sendable The value is immutable, and each MLXArray in it is a handle that no cache writes: the arrays of a live continuation come from MTPDrafterContinuation.arrays(), which gives new handles, and a prompt cache load evaluates the arrays it reads before it returns.
public struct SerializedMTPDrafterContinuation: @unchecked Sendable {

    /// The arrays of the continuation. Each key starts with
    /// ``MTPDrafterContinuation/reservedKeyPrefix``.
    public let arrays: [String: MLXArray]

    /// The metadata of the continuation. Each key starts with
    /// ``MTPDrafterContinuation/reservedKeyPrefix``.
    public let metadata: [String: String]

    /// Serializes a continuation.
    ///
    /// - Parameter continuation: The continuation.
    /// - Throws: ``KVCacheError`` when a drafter cache class cannot be saved to a prompt cache.
    public init(_ continuation: MTPDrafterContinuation) throws {
        self.init(arrays: try continuation.arrays(), metadata: try continuation.metadata())
    }

    /// Takes the serialized form of a continuation from arrays and metadata that can hold other
    /// entries too.
    ///
    /// - Parameters:
    ///   - arrays: The arrays. Only the entries whose key starts with
    ///     ``MTPDrafterContinuation/reservedKeyPrefix`` are kept.
    ///   - metadata: The metadata. Only the entries whose key starts with
    ///     ``MTPDrafterContinuation/reservedKeyPrefix`` are kept.
    public init(arrays: [String: MLXArray], metadata: [String: String]) {
        self.arrays = Self.reservedEntries(of: arrays)
        self.metadata = Self.reservedEntries(of: metadata)
    }

    /// Restores the continuation for `drafter` in the stream of `parameters`, with every check of
    /// ``MTPDrafterContinuation/init(arrays:metadata:drafter:parameters:)``.
    ///
    /// - Parameters:
    ///   - drafter: The drafter that resumes the continuation.
    ///   - parameters: The generation parameters of the stream that resumes the continuation, or
    ///     `nil` for a stream without a cache configuration.
    /// - Returns: The continuation, with each array evaluated.
    /// - Throws: ``KVCacheError`` for each error of
    ///   ``MTPDrafterContinuation/init(arrays:metadata:drafter:parameters:)``.
    public func restored(
        by drafter: any ResumableMTPDrafterModel, parameters: GenerateParameters? = nil
    ) throws -> MTPDrafterContinuation {
        try MTPDrafterContinuation(
            arrays: arrays, metadata: metadata, drafter: drafter, parameters: parameters)
    }

    /// Removes the serialized continuation from the flat arrays of a prompt cache file and reads
    /// it with the flat metadata of that file.
    ///
    /// - Parameters:
    ///   - fileArrays: The flat arrays of the file. The entries of the continuation are removed,
    ///     so the rest is the arrays of the main caches and the model state.
    ///   - fileMetadata: The flat metadata of the file.
    /// - Returns: The continuation, or `nil` when the file holds no entry of one.
    static func removed(
        from fileArrays: inout [String: MLXArray], metadata fileMetadata: [String: String]
    ) -> SerializedMTPDrafterContinuation? {
        let serialized = SerializedMTPDrafterContinuation(
            arrays: fileArrays, metadata: fileMetadata)
        guard !serialized.arrays.isEmpty || !serialized.metadata.isEmpty else { return nil }
        fileArrays = fileArrays.filter { !isReserved($0.key) }
        return serialized
    }

    /// Whether `key` names an entry of a serialized continuation.
    private static func isReserved(_ key: String) -> Bool {
        key.hasPrefix(MTPDrafterContinuation.reservedKeyPrefix)
    }

    /// The entries of `values` whose key names an entry of a serialized continuation.
    private static func reservedEntries<Value>(of values: [String: Value]) -> [String: Value] {
        values.filter { isReserved($0.key) }
    }
}
