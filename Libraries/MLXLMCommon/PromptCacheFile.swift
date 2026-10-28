// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// A cache type outside `MLXLMCommon` that ``loadPromptCacheSnapshot(url:into:)`` restores in
/// place.
///
/// Such a type often needs model configuration to build, thus its
/// ``KVCache/restore(state:metaState:)`` cannot make it. The template restore writes the saved
/// values into an instance that the model made. The type names itself in a prompt cache file
/// with ``KVCache/serializedClassName``. The setters of a cache usually stop the process on bad
/// input, thus the restore calls ``validatePromptCacheRestore(state:metaState:)`` for every
/// layer before it calls ``restorePromptCache(state:metaState:)`` for any layer.
public protocol PromptCacheRestorable: KVCache {
    /// Checks saved values before any setter reads them.
    ///
    /// - Parameters:
    ///   - state: The saved arrays.
    ///   - metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when ``restorePromptCache(state:metaState:)`` cannot accept
    ///   the values.
    func validatePromptCacheRestore(state: [MLXArray], metaState: [String]) throws

    /// Writes values into this cache. The values passed the check of
    /// ``validatePromptCacheRestore(state:metaState:)``.
    ///
    /// - Parameters:
    ///   - state: The saved arrays.
    ///   - metaState: The saved meta state.
    func restorePromptCache(state: [MLXArray], metaState: [String])
}

/// A prompt cache and the model state that belongs with it.
///
/// Keeping the two together is the point: a KV cache restored without its model state positions
/// later tokens as if the cached prefix contained no images, which changes the output silently.
/// Pass a snapshot to a `ChatSession` initializer that accepts a `promptCache` rather than
/// unpacking it into the `cache:` initializer.
///
/// A snapshot does not carry the chat transcript. A `ChatSession` restored from one appends each
/// new message rather than re-rendering the conversation, so a later image-bearing turn builds a
/// different prompt than it would in a session that still holds its history. Positions stay
/// correct either way. On vision encoders that attend across image boundaries (Qwen2-VL today)
/// the new image's features differ too, since it is encoded alone rather than beside the cached
/// one; Qwen2.5-VL, Qwen3-VL, and GLM-OCR isolate each image and are unaffected.
///
/// The cache instances are mutable reference types. Transfer a snapshot to one session or copy the
/// caches before constructing multiple sessions from it.
///
/// A snapshot of an MTP session also carries the drafter continuation that stands where the main
/// cache ends, so a session restored from it resumes the drafter rather than cold-starting it.
public struct PromptCacheSnapshot {
    public let cache: [KVCache]
    public let metadata: [String: String]
    public let state: LMOutput.State?

    /// The MTP drafter continuation saved with the cache, or `nil` when the save had none. A file
    /// written before drafter continuations were saved loads with `nil`.
    public let drafterContinuation: SerializedMTPDrafterContinuation?

    /// Pair a cache with the model state that belongs with it.
    ///
    /// Use this when the cache was built in process — ``loadPromptCacheSnapshot(url:)`` returns a
    /// snapshot for caches read from disk.
    ///
    /// - Parameters:
    ///   - cache: the KV cache
    ///   - metadata: optional caller metadata to carry alongside it
    ///   - state: the model state captured with the cache, if the model produced any
    ///   - drafterContinuation: the MTP drafter continuation that stands where `cache` ends, if
    ///     an MTP generation left one
    public init(
        cache: [KVCache], metadata: [String: String] = [:], state: LMOutput.State? = nil,
        drafterContinuation: SerializedMTPDrafterContinuation? = nil
    ) {
        self.cache = cache
        self.metadata = metadata
        self.state = state
        self.drafterContinuation = drafterContinuation
    }
}

/// Save a pre-computed prompt cache to a file.
///
/// The file also records the `offset` of each top-level cache, because the `metaState` of some
/// classes (for example `MambaCache`) does not hold it. The load functions apply the record and
/// do not give it back in the user metadata.
///
/// This is ``preparePromptCacheSave(cache:metadata:state:drafterContinuation:)`` and then
/// ``writePromptCache(_:url:)``. A caller that must give the caches to another task before the
/// file write ends calls the two steps itself.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
///   - state: Optional model state associated with the cache
///   - drafterContinuation: Optional MTP drafter continuation that stands where `cache` ends
/// - Throws: ``KVCacheError`` when a key of `metadata` uses a reserved prefix
///   (`__mlx_lm_state_` or `__mlx_lm_offset_`), or when `state` has values or
///   `drafterContinuation` is given and `cache` is empty.
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:],
    state: LMOutput.State? = nil,
    drafterContinuation: SerializedMTPDrafterContinuation? = nil
) throws {
    try writePromptCache(
        preparePromptCacheSave(
            cache: cache, metadata: metadata, state: state,
            drafterContinuation: drafterContinuation),
        url: url)
}

/// The arrays and the metadata of one prompt cache file, read from the caches and ready to write.
///
/// ``preparePromptCacheSave(cache:metadata:state:drafterContinuation:)`` makes it, and
/// ``writePromptCache(_:url:)`` writes it. Each array is a new `MLXArray` handle, and the input
/// holds no reference to a cache. Thus a cache can take more tokens after the prepare step, and
/// the file still holds the values that the cache held at that step.
// swiftlint:disable:next no_unchecked_sendable The input is immutable, and each MLXArray in it is a handle that only this input holds: no cache writes it, and the write only reads it.
public struct PromptCacheSaveInput: @unchecked Sendable {
    /// The flat arrays of the file: `"i.j"` for array `j` of cache `i`, and the model-state
    /// tensors under their reserved prefix.
    public let arrays: [String: MLXArray]

    /// The flat metadata of the file: the meta state of each cache, the user metadata with the
    /// offset record and the model-state keys, and the class name of each cache.
    public let metadata: [String: String]

    /// The class name of each top-level cache, in cache order, as the file writes it.
    public let classNames: [String]
}

/// Reads the caches and the model state that a prompt cache file saves, and makes the input of
/// ``writePromptCache(_:url:)``.
///
/// The function reads `state` and `metaState` of each cache at once, thus the caller runs it
/// while it owns the caches. It does not evaluate the arrays and it does not write a file.
///
/// The drafter continuation goes in the top level of the arrays and the metadata, under
/// ``MTPDrafterContinuation/reservedKeyPrefix``, which no key of the main caches starts with.
///
/// - Parameters:
///   - cache: The model cache state.
///   - metadata: Optional metadata to save along with cache state.
///   - state: Optional model state associated with the cache.
///   - drafterContinuation: Optional MTP drafter continuation that stands where `cache` ends.
/// - Returns: The input, which holds new array handles and no reference to a cache.
/// - Throws: ``KVCacheError`` when a key of `metadata` uses a reserved prefix
///   (`__mlx_lm_state_` or `__mlx_lm_offset_`), or when `state` has values or
///   `drafterContinuation` is given and `cache` is empty.
public func preparePromptCacheSave(
    cache: [KVCache],
    metadata: [String: String] = [:],
    state: LMOutput.State? = nil,
    drafterContinuation: SerializedMTPDrafterContinuation? = nil
) throws -> PromptCacheSaveInput {
    try PromptCacheOffsetRecord.validateUserMetadata(metadata)
    let stateArrays = try promptCacheStateArrays(state, userMetadata: metadata)
    guard stateArrays.isEmpty || !cache.isEmpty else {
        throw KVCacheError(message: "Model state requires at least one prompt cache")
    }
    guard drafterContinuation == nil || !cache.isEmpty else {
        throw KVCacheError(
            message: "An MTP drafter continuation requires at least one prompt cache")
    }

    let cacheClasses = try cache.map {
        promptCacheClassName(
            try KVCacheTypeRegistry.shared.serializedClassName(for: $0),
            hasState: !stateArrays.isEmpty)
    }
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }

    // Flatten cache data using tree_flatten compatible structure: "i.j" format
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // Create cache_metadata structure compatible with Python: [cache_info, metadata, cache_classes]
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata). The offset record
    // goes in the same part, under its reserved prefix, which no user key uses.
    let recordedMetadata = metadata.merging(PromptCacheOffsetRecord.entries(for: cache)) {
        user, _ in user
    }
    for (key, value) in recordedMetadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    addPromptCacheState(
        stateArrays, flattenedData: &flattenedData, flattenedMetadata: &flattenedMetadata)
    if let drafterContinuation {
        flattenedData.merge(drafterContinuation.arrays) { main, _ in main }
        flattenedMetadata.merge(drafterContinuation.metadata) { main, _ in main }
    }

    // A cache can write into the array object that its `state` gave back (a full
    // `RotatingKVCache` does), thus the input takes a new handle of each array.
    return PromptCacheSaveInput(
        arrays: flattenedData.mapValues { $0[.ellipsis] }, metadata: flattenedMetadata,
        classNames: cacheClasses)
}

/// Writes a prompt cache file that
/// ``preparePromptCacheSave(cache:metadata:state:drafterContinuation:)`` prepared.
///
/// The function reads no cache, thus it can run on another task after the caches have
/// taken more tokens.
///
/// - Parameters:
///   - input: The prepared arrays and metadata.
///   - url: The URL of the `.safetensors` file.
/// - Throws: The error of the safetensors writer, for example when `url` does not end in
///   `.safetensors` or the file cannot be written.
public func writePromptCache(_ input: PromptCacheSaveInput, url: URL) throws {
    try save(arrays: input.arrays, metadata: input.metadata, url: url)
}

/// Load a prompt cache from a file, without model state.
///
/// Prefer ``loadPromptCacheSnapshot(url:)``, which also restores the model state a cache needs to
/// be continued correctly. This tuple form has no slot for that state, so it rejects files that
/// carry any rather than dropping it. Models that position from a carried anchor refuse a warm
/// cache that arrives without one, so a cache saved through this API cannot warm them.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
/// - Throws: If the file contains model state that this tuple return value cannot represent, or
///   for each error of ``loadPromptCacheSnapshot(url:)``
public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let promptCache = try loadPromptCacheSnapshot(url: url)
    guard promptCache.state == nil else {
        throw KVCacheError(
            message:
                "Prompt cache contains model state; use loadPromptCacheSnapshot(url:) to restore it"
        )
    }
    return (promptCache.cache, promptCache.metadata)
}

/// Load a prompt cache and its associated model state from a file.
///
/// The arrays are read before this returns, so the file can be overwritten or removed afterwards.
///
/// - Parameter url: The URL of the `.safetensors` file.
/// - Returns: The snapshot. Each cache has the offset that the file records, when the file has
///   an offset record.
/// - Throws: ``KVCacheError`` when the file is not a prompt cache that this reader can restore,
///   or when a restored offset is not the offset that the file records.
public func loadPromptCacheSnapshot(url: URL) throws -> PromptCacheSnapshot {
    let contents = try PromptCacheFileContents(url: url)
    // A loaded array reads the file only when it is evaluated.
    eval(contents.fileArrays)
    let caches = try contents.layers.map {
        try restoreCacheFromMetaState(
            className: $0.className, state: $0.state, metaState: $0.metaState)
    }
    try PromptCacheOffsetRecord.apply(contents.offsets, to: caches)
    return contents.snapshot(of: caches)
}

/// Load a prompt cache and its model state from a file into the fresh caches that the model
/// made, for example with `model.newCache(parameters:)`.
///
/// A cache type that needs model configuration to build (for example `DeepSeekV4Cache`) has no
/// `(state, metaState)` factory, thus ``loadPromptCacheSnapshot(url:)`` cannot build it. This
/// function writes the saved values into the caches in `templates` instead:
///
/// - The saved class of each layer must be the class of its template. One conversion is
///   accepted: generation can change a `KVCacheSimple` layer into a `QuantizedKVCache` or a
///   `TurboQuantKVCache`, and a model never makes those classes. For a `KVCacheSimple`
///   template, a saved layer of one of these two classes is built new from the file, and the
///   snapshot holds that new cache in place of the template. The same rule applies to each
///   child of a `CacheList`.
/// - The saved configuration of a layer must be the configuration of its template. For
///   example, a `RotatingKVCache` layer must have the window and the kept prefix of its
///   template, thus a ring never restores into a model that asks for another window. A
///   `QuantizedKVCache` layer must have the group size and the bit width of its template.
/// - Every layer is checked before any setter runs, because the cache setters stop the process
///   on bad input.
/// - The function evaluates every array that it read from the file before it returns. The
///   safetensors load is lazy, thus without this step the file must stay until the first use.
///   After the return the caller can delete the file.
///
/// - Parameters:
///   - url: The URL of the `.safetensors` file.
///   - templates: The fresh caches of the model, one for each saved layer.
/// - Returns: The snapshot. Its caches are the templates, or the new converted caches.
/// - Throws: ``KVCacheError`` on a layer-count mismatch, a class mismatch, saved values that
///   do not fit a class, or a restored offset that is not the offset that the file records.
///   After a throw the state of the templates is not defined. Discard them.
public func loadPromptCacheSnapshot(
    url: URL, into templates: [KVCache]
) throws -> PromptCacheSnapshot {
    try restorePromptCacheSnapshot(PromptCacheFileContents(url: url), into: templates)
}

/// Restores the arrays and the metadata of a prompt cache, as
/// ``preparePromptCacheSave(cache:metadata:state:drafterContinuation:)`` gives them, into the
/// fresh caches that the
/// model made.
///
/// This is ``loadPromptCacheSnapshot(url:into:)`` for values held in memory, with the same rules
/// and the same checks.
///
/// - Parameters:
///   - arrays: The flat arrays of the prompt cache.
///   - metadata: The flat metadata of the prompt cache.
///   - templates: The fresh caches of the model, one for each saved layer.
/// - Returns: The snapshot. Its caches are the templates, or the new converted caches.
/// - Throws: ``KVCacheError`` for each error of ``loadPromptCacheSnapshot(url:into:)``. After a
///   throw the state of the templates is not defined. Discard them.
package func restorePromptCacheSnapshot(
    arrays: [String: MLXArray], metadata: [String: String], into templates: [KVCache]
) throws -> PromptCacheSnapshot {
    try restorePromptCacheSnapshot(
        PromptCacheFileContents(arrays: arrays, metadata: metadata), into: templates)
}

/// Restores the parsed contents of a prompt cache into the fresh caches that the model made.
///
/// - Parameters:
///   - contents: The parsed arrays and metadata of the prompt cache.
///   - templates: The fresh caches of the model, one for each saved layer.
/// - Returns: The snapshot. Its caches are the templates, or the new converted caches.
/// - Throws: ``KVCacheError`` for each error of ``loadPromptCacheSnapshot(url:into:)``.
private func restorePromptCacheSnapshot(
    _ contents: PromptCacheFileContents, into templates: [KVCache]
) throws -> PromptCacheSnapshot {
    guard contents.layers.count == templates.count else {
        throw KVCacheError(
            message:
                "The prompt cache holds \(contents.layers.count) layers and the model gave \(templates.count) caches."
        )
    }
    let restoreSteps = try zip(contents.layers, templates).map { layer, template in
        try PromptCacheTemplateRestore.prepare(layer, into: template)
    }
    let caches = restoreSteps.map { $0() }
    try PromptCacheOffsetRecord.apply(contents.offsets, to: caches)
    eval(contents.fileArrays)
    return contents.snapshot(of: caches)
}

/// The contents of a prompt cache file, before any cache is built.
private struct PromptCacheFileContents {
    /// The place of the cache information (the meta state of each layer) in the metadata.
    private static let cacheInfoPart = 0

    /// The place of the user metadata in the metadata.
    private static let userMetadataPart = 1

    /// The place of the class names in the metadata.
    private static let classNamesPart = 2

    /// The saved layers, in order.
    let layers: [SavedPromptCacheLayer]

    /// The caller metadata.
    let userMetadata: [String: String]

    /// The recorded offset of each layer, or `nil` when the file has no offset record.
    let offsets: [Int]?

    /// The model state, when the file holds one.
    let state: LMOutput.State?

    /// The MTP drafter continuation, when the file holds one.
    let drafterContinuation: SerializedMTPDrafterContinuation?

    /// Every array that the file holds: the cache arrays, the model-state arrays and the arrays
    /// of the drafter continuation.
    let fileArrays: [MLXArray]

    /// Reads a prompt cache file.
    ///
    /// The arrays are not evaluated: the safetensors load is lazy.
    ///
    /// - Parameter url: The URL of the `.safetensors` file.
    /// - Throws: ``KVCacheError`` when the metadata is not a prompt cache layout.
    init(url: URL) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        try self.init(arrays: arrays, metadata: metadata)
    }

    /// Parses the flat arrays and metadata of a prompt cache.
    ///
    /// - Parameters:
    ///   - arrays: The flat arrays, as a prompt cache file holds them.
    ///   - metadata: The flat metadata, as a prompt cache file holds it.
    /// - Throws: ``KVCacheError`` when the metadata is not a prompt cache layout.
    init(arrays: [String: MLXArray], metadata: [String: String]) throws {
        var arrays = arrays
        fileArrays = Array(arrays.values)
        // The drafter continuation stands beside the three metadata parts, which ignore its
        // keys, and beside the cache arrays, which must not count it as a layer.
        drafterContinuation = SerializedMTPDrafterContinuation.removed(
            from: &arrays, metadata: metadata)

        // Unflatten metadata using tree_unflatten compatible logic.
        // Structure: [cache_info, user_metadata, cache_classes]. `unflattenMetadata` always
        // gives back the three parts, thus no count check is necessary.
        let unflattenedMetadata = unflattenMetadata(metadata)
        let cacheInfo = unflattenedMetadata[Self.cacheInfoPart] as? [[String]] ?? []
        let storedUserMetadata =
            unflattenedMetadata[Self.userMetadataPart] as? [String: String] ?? [:]
        let (loadedState, loadedUserMetadata) = try loadPromptCacheState(
            arrays: &arrays, metadata: storedUserMetadata)
        state = loadedState
        let storedCacheClasses = unflattenedMetadata[Self.classNamesPart] as? [String] ?? []
        let cacheClasses = try loadPromptCacheClasses(
            storedCacheClasses, hasState: loadedState != nil)
        (offsets, userMetadata) = try PromptCacheOffsetRecord.read(
            from: loadedUserMetadata, layerCount: cacheClasses.count)

        guard cacheInfo.count == cacheClasses.count else {
            throw KVCacheError(message: "Mismatch in cache counts")
        }

        // Metadata carries the cache count even when one or more valid caches have no arrays.
        // State tensors were removed from `arrays` above, so only cache arrays remain here.
        let cacheData = try unflattenArrays(arrays, cacheCount: cacheClasses.count)
        layers = cacheData.indices.map {
            SavedPromptCacheLayer(
                className: cacheClasses[$0], state: cacheData[$0], metaState: cacheInfo[$0])
        }
    }

    /// The snapshot of the restored caches, with the user metadata, the model state and the
    /// drafter continuation of the file.
    ///
    /// - Parameter caches: The caches restored from ``layers``.
    /// - Returns: The snapshot.
    func snapshot(of caches: [KVCache]) -> PromptCacheSnapshot {
        PromptCacheSnapshot(
            cache: caches, metadata: userMetadata, state: state,
            drafterContinuation: drafterContinuation)
    }
}

/// The record of the offset of each top-level cache in a prompt cache file.
///
/// The `metaState` of some classes does not hold the offset: an `ArraysCache` or a
/// `MambaCache` comes back at offset 0, and a `ChunkedKVCache` after a front trim comes back at
/// the number of rows that it keeps. The save thus writes the offset of each top-level cache in
/// the user-metadata part, under a reserved key prefix. The load removes the record from the
/// user metadata that it gives back, and applies it after the restore. A file without the record
/// loads as before.
private enum PromptCacheOffsetRecord {
    /// The reserved prefix of the metadata keys of the record. User metadata cannot use it.
    static let metadataPrefix = "__mlx_lm_offset_"

    /// The metadata key of the record of one layer.
    ///
    /// - Parameter layer: The index of the top-level cache.
    /// - Returns: The key, without the prefix of the user-metadata part.
    static func key(layer: Int) -> String {
        "\(metadataPrefix)\(layer)"
    }

    /// Refuses user metadata that uses the reserved prefix.
    ///
    /// - Parameter metadata: The caller metadata of a save.
    /// - Throws: ``KVCacheError`` when a key starts with ``metadataPrefix``.
    static func validateUserMetadata(_ metadata: [String: String]) throws {
        guard !metadata.keys.contains(where: { $0.hasPrefix(metadataPrefix) }) else {
            throw KVCacheError(
                message: "User metadata uses the reserved prompt cache offset namespace")
        }
    }

    /// Makes the record of a list of caches.
    ///
    /// - Parameter caches: The top-level caches of a save.
    /// - Returns: One metadata entry for each cache.
    static func entries(for caches: [KVCache]) -> [String: String] {
        Dictionary(
            uniqueKeysWithValues: caches.enumerated().map { layer, cache in
                (key(layer: layer), String(cache.offset))
            })
    }

    /// Reads the record from the user metadata of a file, and removes it.
    ///
    /// - Parameters:
    ///   - metadata: The user metadata of the file, which can hold the record.
    ///   - layerCount: The number of top-level caches in the file.
    /// - Returns: The offset of each layer, or `nil` when the file has no record, and the user
    ///   metadata without the record.
    /// - Throws: ``KVCacheError`` when the record does not have exactly one non-negative integer
    ///   for each layer.
    static func read(
        from metadata: [String: String], layerCount: Int
    ) throws -> (offsets: [Int]?, userMetadata: [String: String]) {
        let record = metadata.filter { $0.key.hasPrefix(metadataPrefix) }
        let userMetadata = metadata.filter { !$0.key.hasPrefix(metadataPrefix) }
        guard !record.isEmpty else { return (nil, userMetadata) }
        guard record.count == layerCount else {
            throw KVCacheError(
                message:
                    "The prompt cache offset record has \(record.count) entries for \(layerCount) caches."
            )
        }
        let offsets = try (0 ..< layerCount).map { layer in
            guard let value = record[key(layer: layer)], let offset = Int(value), offset >= 0
            else {
                throw KVCacheError(
                    message: "The prompt cache offset record of cache \(layer) is not valid.")
            }
            return offset
        }
        return (offsets, userMetadata)
    }

    /// Applies the record to the restored caches.
    ///
    /// `KVCache.offset` has no setter, and some classes compute it from inner caches. Thus the
    /// function sets the offset only of a class whose saved values do not hold it. For each
    /// other class it checks that the restored offset equals the recorded offset.
    ///
    /// - Parameters:
    ///   - offsets: The recorded offsets, or `nil` when the file has no record.
    ///   - caches: The restored top-level caches, in the order of the record.
    /// - Throws: ``KVCacheError`` when a restored offset is not the recorded offset.
    static func apply(_ offsets: [Int]?, to caches: [KVCache]) throws {
        guard let offsets else { return }
        for (layer, (cache, offset)) in zip(caches, offsets).enumerated() {
            try apply(offset, to: cache, layer: layer)
        }
    }

    /// Applies the recorded offset of one layer to its restored cache.
    ///
    /// A `CacheList` reads its offset from its children: the largest offset of a child. Thus the
    /// function sets the recorded offset on each child whose saved values do not hold it. Then it
    /// checks the largest offset of the children whose saved values hold it. A child that holds
    /// fewer tokens than the list, such as a child that the model never fed, does not make the
    /// restore fail.
    ///
    /// - Parameters:
    ///   - offset: The recorded offset of the layer.
    ///   - cache: The restored cache of the layer.
    ///   - layer: The index of the layer, for the error message.
    /// - Throws: ``KVCacheError`` when the restored offset of the layer is not the recorded
    ///   offset.
    private static func apply(_ offset: Int, to cache: KVCache, layer: Int) throws {
        setOffset(offset, ofSettableCachesIn: cache)
        if let restored = savedOffset(of: cache), restored != offset {
            throw KVCacheError(
                message:
                    "The restored cache \(layer) has offset \(restored) and the prompt cache records \(offset)."
            )
        }
    }

    /// The offset that the saved values of `cache` hold: the offset of a cache whose saved
    /// values hold it, or the largest such offset of the children of a `CacheList`.
    ///
    /// - Parameter cache: The restored cache of the layer, or a child of it.
    /// - Returns: The offset, or `nil` when no saved value of the cache holds its offset.
    private static func savedOffset(of cache: KVCache) -> Int? {
        if let list = cache as? CacheList {
            return list.children.compactMap(savedOffset(of:)).max()
        }
        return offsetSettableCache(cache) == nil ? cache.offset : nil
    }

    /// Sets `offset` on `cache`, or on each child of a `CacheList`, when the saved values of
    /// that cache do not hold its offset.
    ///
    /// - Parameters:
    ///   - offset: The recorded offset of the layer.
    ///   - cache: The restored cache of the layer, or a child of it.
    private static func setOffset(_ offset: Int, ofSettableCachesIn cache: KVCache) {
        if let list = cache as? CacheList {
            for child in list.children {
                setOffset(offset, ofSettableCachesIn: child)
            }
        } else if let settable = offsetSettableCache(cache) {
            settable.offset = offset
        }
    }

    /// Gives the cache as a `BaseKVCache` when its saved values do not hold its offset.
    ///
    /// - Parameter cache: A restored cache.
    /// - Returns: The cache for an `ArraysCache` (thus also a `MambaCache`) or a
    ///   `ChunkedKVCache`, and `nil` for each other class.
    private static func offsetSettableCache(_ cache: KVCache) -> BaseKVCache? {
        switch cache {
        case let arrays as ArraysCache: arrays
        case let chunked as ChunkedKVCache: chunked
        default: nil
        }
    }
}

private func promptCacheStateArrays(
    _ state: LMOutput.State?, userMetadata: [String: String]
) throws -> [(key: String, value: MLXArray)] {
    guard !userMetadata.keys.contains(where: { $0.hasPrefix(promptCacheStateMetadataPrefix) })
    else {
        throw KVCacheError(message: "User metadata uses the reserved prompt cache state namespace")
    }
    guard let state else { return [] }
    return try state.serializedArrays().sorted { $0.key < $1.key }
}

private func addPromptCacheState(
    _ stateArrays: [(key: String, value: MLXArray)],
    flattenedData: inout [String: MLXArray], flattenedMetadata: inout [String: String]
) {
    guard !stateArrays.isEmpty else { return }

    flattenedMetadata["1.\(promptCacheStateVersionKey)"] = promptCacheStateFormatVersion
    flattenedMetadata["1.\(promptCacheStateCountKey)"] = String(stateArrays.count)
    for (index, entry) in stateArrays.enumerated() {
        flattenedData[promptCacheStateTensorKey(index)] = entry.value
        flattenedMetadata["1.\(promptCacheStateEntryKey(index))"] = entry.key
    }
}

private func loadPromptCacheState(
    arrays: inout [String: MLXArray], metadata: [String: String]
) throws -> (LMOutput.State?, [String: String]) {
    let stateMetadata = metadata.filter { $0.key.hasPrefix(promptCacheStateMetadataPrefix) }
    let stateTensorKeys = arrays.keys.filter { $0.hasPrefix(promptCacheStateTensorPrefix) }
    guard !stateMetadata.isEmpty || !stateTensorKeys.isEmpty else { return (nil, metadata) }

    guard metadata[promptCacheStateVersionKey] == promptCacheStateFormatVersion else {
        throw KVCacheError(message: "Unsupported prompt cache state format")
    }
    guard let countValue = metadata[promptCacheStateCountKey],
        let count = Int(countValue), count > 0
    else {
        throw KVCacheError(message: "Invalid prompt cache state count")
    }

    var serializedArrays: [String: MLXArray] = [:]
    for index in 0 ..< count {
        let tensorKey = promptCacheStateTensorKey(index)
        guard let key = metadata[promptCacheStateEntryKey(index)],
            let array = arrays.removeValue(forKey: tensorKey)
        else {
            throw KVCacheError(message: "Invalid prompt cache state entry at index \(index)")
        }
        guard serializedArrays.updateValue(array, forKey: key) == nil else {
            throw KVCacheError(message: "Duplicate prompt cache state key: \(key)")
        }
    }

    // The loop proved every expected entry is present; counting rejects the leftovers — a
    // reserved-namespace key this reader does not know about means the file was written by
    // something else, so refuse it rather than silently dropping part of the state.
    guard stateMetadata.count == count + 2 else {
        throw KVCacheError(message: "Unexpected prompt cache state metadata")
    }
    guard stateTensorKeys.count == count else {
        throw KVCacheError(message: "Unexpected prompt cache state tensors")
    }

    let userMetadata = metadata.filter { !$0.key.hasPrefix(promptCacheStateMetadataPrefix) }
    return (LMOutput.State(serializedArrays: serializedArrays), userMetadata)
}

/// Prefixes the stored cache class name when the file carries model state.
///
/// The marker is redundant for this reader — the state metadata already says whether state is
/// present — and exists for older ones. A reader predating model state would ignore the
/// `__mlx_lm_state_` metadata entirely and restore the cache without its continuation anchor,
/// which positions new tokens as if the cached prefix held no images. Poisoning the class name
/// makes that reader fail with "Unknown cache class" instead of continuing at the wrong offsets.
private func promptCacheClassName(_ className: String, hasState: Bool) -> String {
    hasState ? "\(promptCacheStateClassPrefix)\(className)" : className
}

private func loadPromptCacheClasses(_ classNames: [String], hasState: Bool) throws -> [String] {
    if hasState {
        guard !classNames.isEmpty,
            classNames.allSatisfy({ $0.hasPrefix(promptCacheStateClassPrefix) })
        else {
            throw KVCacheError(
                message: "Prompt cache model state is missing its compatibility marker")
        }
        return classNames.map { String($0.dropFirst(promptCacheStateClassPrefix.count)) }
    }

    guard !classNames.contains(where: { $0.hasPrefix(promptCacheStateClassPrefix) }) else {
        throw KVCacheError(message: "Prompt cache compatibility marker has no model state")
    }
    return classNames
}

private func promptCacheStateEntryKey(_ index: Int) -> String {
    "\(promptCacheStateMetadataPrefix)\(index)_key"
}

private func promptCacheStateTensorKey(_ index: Int) -> String {
    "\(promptCacheStateTensorPrefix)\(index)"
}

private let promptCacheStateFormatVersion = "1"
private let promptCacheStateMetadataPrefix = "__mlx_lm_state_"
private let promptCacheStateVersionKey = "__mlx_lm_state_version"
private let promptCacheStateCountKey = "__mlx_lm_state_count"
private let promptCacheStateTensorPrefix = "__mlx_lm_state_tensor_"
// Derived so the format version lives in exactly one place.
private let promptCacheStateClassPrefix =
    "__mlx_lm_state_v\(promptCacheStateFormatVersion)__:"

/// Unflatten arrays from tree_flatten format (e.g., "0.1", "1.0") to nested structure
private func unflattenArrays(
    _ flatArrays: [String: MLXArray],
    cacheCount: Int
) throws -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        guard components.count == 2,
            let i = Int(components[0]),
            let j = Int(components[1]),
            (0 ..< cacheCount).contains(i),
            j >= 0
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid array key '\(key)'.")
        }
        arrayMap[i, default: [:]][j] = array
    }

    return try (0 ..< cacheCount).map { cacheIndex in
        guard let arrays = arrayMap[cacheIndex], !arrays.isEmpty else { return [] }
        var result: [MLXArray] = []
        result.reserveCapacity(arrays.count)
        for arrayIndex in 0 ..< arrays.count {
            guard let array = arrays[arrayIndex] else {
                throw KVCacheError(
                    message:
                        "Corrupt prompt cache: cache \(cacheIndex) has non-contiguous array indices."
                )
            }
            result.append(array)
        }
        return result
    }
}

/// Unflatten metadata from tree_flatten format to nested structure
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}
