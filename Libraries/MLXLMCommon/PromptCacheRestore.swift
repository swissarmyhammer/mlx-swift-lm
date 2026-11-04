// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Checks the saved `state` and `metaState` of one built-in cache class before a setter
/// reads them.
///
/// A cache type outside `MLXLMCommon` that holds a built-in cache (for example a
/// `RotatingKVCache` window) calls this from
/// ``PromptCacheRestorable/validatePromptCacheRestore(state:metaState:)``. The built-in
/// setters stop the process on bad input, thus the check must come first.
///
/// - Parameters:
///   - state: The saved arrays of the built-in part.
///   - metaState: The saved meta state of the built-in part.
///   - cache: The built-in cache that will receive the values. Its class selects the rules.
/// - Throws: ``KVCacheError`` when the values do not fit the class of `cache`.
public func validateBuiltInPromptCache(
    state: [MLXArray], metaState: [String], for cache: KVCache
) throws {
    guard let className = type(of: cache).serializedClassName,
        builtInLeafClassNames.contains(className)
    else {
        throw KVCacheError(
            message: "\(type(of: cache)) is not a built-in cache class that has a check.")
    }
    try validateBuiltInCache(className: className, state: state, metaState: metaState)
}

/// The saved values of one layer of a prompt cache file, or of one child of a saved `CacheList`.
struct SavedPromptCacheLayer {
    /// The class name that the save wrote for the layer.
    let className: String

    /// The saved arrays of the layer.
    let state: [MLXArray]

    /// The saved meta state of the layer.
    let metaState: [String]
}

/// The steps of ``loadPromptCacheSnapshot(url:into:)`` for one layer.
enum PromptCacheTemplateRestore {
    /// A restore step that runs after every layer passed its check. It writes the saved values
    /// and gives back the cache that holds them.
    typealias Step = () -> KVCache

    /// The saved classes that can take the place of a `KVCacheSimple` template. Generation
    /// makes them from a `KVCacheSimple` layer, and a model never makes them itself.
    private static let convertedClassNames: Set<String> = ["QuantizedKVCache", "TurboQuantKVCache"]

    /// The meta-state places that hold configuration that the model sets in `init` of a cache.
    /// The saved values must equal the values of the template:
    ///
    /// - No setter writes some of these values, thus a different saved value gives saved data
    ///   in a cache of another configuration.
    /// - The `metaState` setter of a `RotatingKVCache` writes its window and its kept prefix.
    ///   A different saved value thus changes the ring that the model made into a ring of
    ///   another configuration, and the model does not get the window that it asked for.
    /// - The `metaState` setter of a `QuantizedKVCache` writes its group size and its bit
    ///   width. A different saved value thus changes the cache that the model made into a cache
    ///   of another quantization, and the model does not get the quantization that it asked for.
    private static let fixedConfigurationIndices: [String: [Int]] = [
        "RotatingKVCache": [SavedMetaStateIndex.rotatingKeep, SavedMetaStateIndex.rotatingMaxSize],
        "QuantizedKVCache": [
            SavedMetaStateIndex.quantizedGroupSize, SavedMetaStateIndex.quantizedBits,
        ],
        "VarianceNormalizedKVCache": [
            VarianceNormalizedSavedLayout.tileSizeIndex, VarianceNormalizedSavedLayout.keyBitsIndex,
            VarianceNormalizedSavedLayout.valueBitsIndex,
            VarianceNormalizedSavedLayout.sinkhornIterationsIndex,
        ],
        "TurboQuantKVCache": [
            SavedMetaStateIndex.turboQuantBits, SavedMetaStateIndex.turboQuantKeyBits,
            SavedMetaStateIndex.turboQuantValueBits, SavedMetaStateIndex.turboQuantSeed,
        ],
        "ArraysCache": [SavedMetaStateIndex.arraysSlotCount],
        "MambaCache": [SavedMetaStateIndex.arraysSlotCount],
    ]

    /// Checks one saved layer against its template, and makes the step that restores it.
    ///
    /// No setter runs here, thus a throw leaves the template as it was.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - template: The fresh cache that the model made for this layer.
    /// - Returns: The step that writes the values.
    /// - Throws: ``KVCacheError`` when the layer does not fit the template.
    static func prepare(_ layer: SavedPromptCacheLayer, into template: KVCache) throws -> Step {
        guard let templateClassName = type(of: template).serializedClassName else {
            throw unrestorableTemplate(template)
        }
        guard layer.className == templateClassName else {
            return try prepareConverted(layer, into: template, templateClassName: templateClassName)
        }
        if let list = template as? CacheList {
            return try prepareCacheList(layer, into: list)
        }
        if let restorable = template as? PromptCacheRestorable {
            try restorable.validatePromptCacheRestore(
                state: layer.state, metaState: layer.metaState)
            return {
                restorable.restorePromptCache(state: layer.state, metaState: layer.metaState)
                return restorable
            }
        }
        guard builtInLeafClassNames.contains(templateClassName) else {
            throw unrestorableTemplate(template)
        }
        try validateBuiltInCache(
            className: layer.className, state: layer.state, metaState: layer.metaState)
        try validateFixedConfiguration(layer, template: template)
        return {
            applySavedValues(state: layer.state, metaState: layer.metaState, to: template)
            return template
        }
    }

    /// The error for a template that no restore step can write: a class outside the library
    /// that is not ``PromptCacheRestorable``, or a class with no
    /// ``KVCache/serializedClassName``.
    ///
    /// - Parameter template: The fresh cache that the model made for the layer.
    /// - Returns: The error.
    private static func unrestorableTemplate(_ template: KVCache) -> KVCacheError {
        KVCacheError(message: "\(type(of: template)) cannot receive a restore into a template.")
    }

    /// Makes the step for a saved layer whose class is not the class of its template.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - template: The fresh cache that the model made for this layer.
    ///   - templateClassName: The class name of `template`.
    /// - Returns: The step, which gives back a new cache of the saved class.
    /// - Throws: ``KVCacheError`` unless the template is a `KVCacheSimple` and the saved class
    ///   is a class that generation converts it to.
    private static func prepareConverted(
        _ layer: SavedPromptCacheLayer, into template: KVCache, templateClassName: String
    ) throws -> Step {
        guard templateClassName == "KVCache", convertedClassNames.contains(layer.className)
        else {
            throw KVCacheError(
                message:
                    "The prompt cache holds a \(layer.className) layer and the model gave a \(templateClassName) cache."
            )
        }
        let converted = try restoreCacheFromMetaState(
            className: layer.className, state: layer.state, metaState: layer.metaState)
        return { converted }
    }

    /// Makes the step for a saved `CacheList` layer. Each child restores into the child of the
    /// template at the same place.
    ///
    /// - Parameters:
    ///   - layer: The saved layer.
    ///   - list: The template list.
    /// - Returns: The step, which gives back `list`.
    /// - Throws: ``KVCacheError`` when the children do not fit the children of `list`.
    private static func prepareCacheList(
        _ layer: SavedPromptCacheLayer, into list: CacheList
    ) throws -> Step {
        let savedChildren = try CacheList.savedChildren(
            state: layer.state, metaState: layer.metaState)
        let templateChildren = list.children
        guard savedChildren.count == templateChildren.count else {
            throw KVCacheError(
                message:
                    "The prompt cache holds a CacheList of \(savedChildren.count) children and the model gave \(templateChildren.count)."
            )
        }
        let childSteps = try zip(savedChildren, templateChildren).map { saved, child in
            try prepare(saved, into: child)
        }
        return {
            list.replaceChildren(with: childSteps.map { $0() })
            return list
        }
    }

    /// Checks that the saved configuration values equal the values of the template.
    ///
    /// - Parameters:
    ///   - layer: The saved layer. Its values passed the check of its class.
    ///   - template: The fresh cache that the model made for this layer.
    /// - Throws: ``KVCacheError`` when a configuration value is different.
    private static func validateFixedConfiguration(
        _ layer: SavedPromptCacheLayer, template: KVCache
    ) throws {
        guard let indices = fixedConfigurationIndices[layer.className] else { return }
        let templateMetaState = template.metaState
        let matches = indices.allSatisfy { index in
            index < layer.metaState.count && index < templateMetaState.count
                && layer.metaState[index] == templateMetaState[index]
        }
        guard matches else {
            throw KVCacheError(
                message:
                    "The saved \(layer.className) configuration is not the configuration of the model cache."
            )
        }
    }
}

/// Reconstruct a single cache from its class name, state arrays, and metaState.
///
/// Like Python's `globals()[className].from_state(state, meta_state)`, each cache type
/// encodes enough info in `metaState` to reconstruct itself.
func restoreCacheFromMetaState(
    className: String,
    state: [MLXArray],
    metaState: [String]
) throws -> KVCache {
    guard let type = KVCacheTypeRegistry.shared.type(named: className) else {
        throw KVCacheError(
            message:
                "Unknown cache class: \(className). Register custom cache types with registerKVCacheType(_:)."
        )
    }
    return try type.restore(state: state, metaState: metaState)
}

/// Rebuilds one built-in leaf cache from its saved values: the restore of each built-in class.
///
/// The check runs before any setter, because the setters stop the process on bad input.
///
/// - Parameters:
///   - className: The saved class name, one of ``builtInLeafClassNames``.
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Returns: The cache, with the saved values.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
func restoreBuiltInCache(
    className: String, state: [MLXArray], metaState: [String]
) throws -> KVCache {
    try validateBuiltInCache(className: className, state: state, metaState: metaState)
    let cache = try makeEmptyBuiltInCache(className: className, metaState: metaState)
    applySavedValues(state: state, metaState: metaState, to: cache)
    return cache
}

/// The class names of the built-in caches that hold no child cache. `KVCacheSimple` is the
/// name that older files wrote for `KVCache`.
private let builtInLeafClassNames: Set<String> = [
    "KVCache", "KVCacheSimple", "RotatingKVCache", "QuantizedKVCache",
    "VarianceNormalizedKVCache", "ChunkedKVCache", "MambaCache", "ArraysCache",
    "TurboQuantKVCache",
]

/// The places of the values in the saved meta state of the built-in classes.
enum SavedMetaStateIndex {
    /// The number of leading tokens that a `RotatingKVCache` keeps in its window.
    static let rotatingKeep = 0
    /// The window size of a `RotatingKVCache`.
    static let rotatingMaxSize = 1
    /// The growth step of a `RotatingKVCache`.
    static let rotatingStep = 2
    /// The offset of a `RotatingKVCache`.
    static let rotatingOffset = 3
    /// The write index of a `RotatingKVCache`.
    static let rotatingIndex = 4
    /// The number of integer values at the start of a `RotatingKVCache` meta state.
    static let rotatingIntegerCount = 5
    /// The capacity origin of a `RotatingKVCache`, when the meta state holds one.
    static let rotatingCapacityOrigin = 5
    /// The wrapped flag of a `RotatingKVCache`, when the meta state holds one.
    static let rotatingWrapped = 6
    /// The rewind capacity of a `RotatingKVCache`, when the meta state holds one.
    static let rotatingRewindCapacity = 7
    /// The offset of a `QuantizedKVCache`.
    static let quantizedOffset = 1
    /// The group size of a `QuantizedKVCache`.
    static let quantizedGroupSize = 2
    /// The bit width of a `QuantizedKVCache`.
    static let quantizedBits = 3
    /// The chunk size of a `ChunkedKVCache`.
    static let chunkedChunkSize = 0
    /// The start position of a `ChunkedKVCache`.
    static let chunkedStartPosition = 1
    /// The bit width of a `TurboQuantKVCache`.
    static let turboQuantBits = 1
    /// The key bit width of a `TurboQuantKVCache`.
    static let turboQuantKeyBits = 2
    /// The value bit width of a `TurboQuantKVCache`.
    static let turboQuantValueBits = 3
    /// The seed of a `TurboQuantKVCache`.
    static let turboQuantSeed = 4
    /// The slot count of an `ArraysCache`.
    static let arraysSlotCount = 0
    /// The present slots of an `ArraysCache`.
    static let arraysPresentSlots = 1
    /// The left padding of an `ArraysCache`, when the meta state holds it.
    static let arraysLeftPadding = 2
    /// The lengths of an `ArraysCache`, when the meta state holds them.
    static let arraysLengths = 3
}

/// The array counts and meta-state counts that the built-in classes save.
enum SavedValueCounts {
    /// The array count of a cache that holds no value yet.
    private static let noArrays = 0
    /// The array count of the keys and the values.
    static let keysAndValues = 2
    /// The array count of quantized keys and values without biases: the packed values and
    /// the scales of each.
    static let quantizedWithoutBiases = 4
    /// The array count of quantized keys and values with biases.
    static let quantizedWithBiases = 6
    /// The meta-state count of a `QuantizedKVCache`: the step, the offset, the group size and
    /// the bit width.
    static let quantizedMetaStateCount = 4
    /// The meta-state count of a `ChunkedKVCache`: the chunk size and the start position.
    static let chunkedMetaStateCount = 2
    /// The smallest meta-state count of an `ArraysCache`: the slot count and the present slots.
    static let arraysMinimumMetaStateCount = 2
    /// The largest meta-state count of an `ArraysCache`: the left padding and the lengths
    /// follow.
    private static let arraysMaximumMetaStateCount = 4
    /// The rank of a saved key, value or quantized tensor: `[batch, heads, tokens, width]`.
    static let tensorRank = 4

    /// No arrays, or the keys and the values.
    static let keyValueStates: Set<Int> = [noArrays, keysAndValues]
    /// The one empty placeholder of a `KVCacheSimple` meta state.
    static let simpleMetaStates: Set<Int> = [1]
    /// Five integers, then an optional capacity origin, an optional wrapped flag and an
    /// optional rewind capacity.
    static let rotatingMetaStates: Set<Int> = [
        SavedMetaStateIndex.rotatingIntegerCount, SavedMetaStateIndex.rotatingCapacityOrigin + 1,
        SavedMetaStateIndex.rotatingWrapped + 1, SavedMetaStateIndex.rotatingRewindCapacity + 1,
    ]
    /// No arrays, or the packed values and the scales of keys and values, with or without
    /// biases.
    static let quantizedStates: Set<Int> = [noArrays, quantizedWithoutBiases, quantizedWithBiases]
    /// The step, the offset, the group size and the bit width.
    static let quantizedMetaStates: Set<Int> = [quantizedMetaStateCount]
    /// The chunk size and the start position.
    static let chunkedMetaStates: Set<Int> = [chunkedMetaStateCount]
    /// The slot count, the present slots, then optional left padding and lengths.
    static let arraysMetaStates = arraysMinimumMetaStateCount ... arraysMaximumMetaStateCount
}

/// Checks the saved values of one built-in cache class before any setter reads them.
///
/// - Parameters:
///   - className: The saved class name.
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class, or the class is unknown.
private func validateBuiltInCache(
    className: String, state: [MLXArray], metaState: [String]
) throws {
    switch className {
    case "KVCache", "KVCacheSimple":
        try validateSimpleCache(state: state, metaState: metaState)
    case "RotatingKVCache":
        try validateRotatingCache(state: state, metaState: metaState)
    case "QuantizedKVCache":
        try validatePromptCache(
            className: className, state: state, stateCounts: SavedValueCounts.quantizedStates,
            metadata: metaState, metadataCounts: SavedValueCounts.quantizedMetaStates)
        _ = try promptCacheIntegers(metaState, className: className)
    case "VarianceNormalizedKVCache":
        try validateVarianceNormalizedCache(state: state, metaState: metaState)
    case "ChunkedKVCache":
        try validateChunkedCache(state: state, metaState: metaState)
    case "MambaCache", "ArraysCache":
        try validateArraysCache(className: className, state: state, metaState: metaState)
    case "TurboQuantKVCache":
        try validateTurboQuantCache(state: state, metaState: metaState)
    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

/// Checks the saved values of a `KVCacheSimple`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateSimpleCache(state: [MLXArray], metaState: [String]) throws {
    try validatePromptCache(
        className: "KVCacheSimple", state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.simpleMetaStates)
    guard metaState == [""] else {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: KVCacheSimple metadata must contain its single empty placeholder."
        )
    }
}

/// Checks the saved values of a `RotatingKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateRotatingCache(state: [MLXArray], metaState: [String]) throws {
    let className = "RotatingKVCache"
    try validatePromptCache(
        className: className, state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.rotatingMetaStates)
    let values = try promptCacheIntegers(
        metaState.prefix(SavedMetaStateIndex.rotatingIntegerCount), className: className)
    let reserveIndex = SavedMetaStateIndex.rotatingRewindCapacity
    if metaState.count > reserveIndex {
        // The setter adds the reserve to the window, thus the sum must not overflow.
        guard let reserve = Int(metaState[reserveIndex]), reserve >= 0,
            values[SavedMetaStateIndex.rotatingMaxSize] <= Int.max - reserve
        else {
            throw KVCacheError(message: "Corrupt prompt cache: invalid rewind capacity.")
        }
    }
    let originIndex = SavedMetaStateIndex.rotatingCapacityOrigin
    if metaState.count > originIndex,
        RotatingKVCache.CapacityOrigin(rawValue: metaState[originIndex]) == nil
    {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: invalid RotatingKVCache capacity origin '\(metaState[originIndex])'."
        )
    }
    let wrappedIndex = SavedMetaStateIndex.rotatingWrapped
    if metaState.count > wrappedIndex, Bool(metaState[wrappedIndex]) == nil {
        throw KVCacheError(
            message:
                "Corrupt prompt cache: invalid RotatingKVCache wrapped flag '\(metaState[wrappedIndex])'."
        )
    }
}

/// Checks the saved values of a `VarianceNormalizedKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateVarianceNormalizedCache(state: [MLXArray], metaState: [String]) throws {
    let layout = try VarianceNormalizedSavedLayout(metaState: metaState)
    let tailStateCount =
        layout.tailLength > 0 ? VarianceNormalizedKVCache.tailStateCount : 0
    let tileStateCount = state.count - min(state.count, tailStateCount)
    let hasValidTileStateCount =
        if layout.tileCount == 0 {
            tileStateCount == 0
        } else {
            tileStateCount.isMultiple(of: layout.tileCount)
                && [
                    VarianceNormalizedKVCache.compactTileStateCount,
                    VarianceNormalizedKVCache.legacyTileStateCount,
                ].contains(tileStateCount / layout.tileCount)
        }
    guard
        layout.hasConsistentOffset,
        state.count >= tailStateCount,
        hasValidTileStateCount,
        state.allSatisfy({ $0.ndim == SavedValueCounts.tensorRank })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid VarianceNormalizedKVCache state."
        )
    }
}

/// The numbers of a saved `VarianceNormalizedKVCache` meta state.
private struct VarianceNormalizedSavedLayout {
    /// The meta-state count of the legacy layout.
    private static let legacyCount = 7
    /// The meta-state count of the versioned layout.
    private static let versionedCount = 10
    /// The place of the metadata version in the versioned layout.
    private static let versionIndex = 7
    /// The place of the key element type in the versioned layout.
    private static let keyDTypeIndex = 8
    /// The place of the value element type in the versioned layout.
    private static let valueDTypeIndex = 9
    /// The name that the versioned layout writes for "no element type yet".
    private static let noDType = "none"
    /// The place of the tile size.
    fileprivate static let tileSizeIndex = 0
    /// The place of the offset.
    private static let offsetIndex = 1
    /// The place of the key bit width.
    fileprivate static let keyBitsIndex = 2
    /// The place of the value bit width.
    fileprivate static let valueBitsIndex = 3
    /// The place of the Sinkhorn iteration count.
    fileprivate static let sinkhornIterationsIndex = 4
    /// The place of the complete tile count.
    private static let tileCountIndex = 5
    /// The place of the raw tail length.
    private static let tailLengthIndex = 6

    /// The number of tokens in one tile.
    let tileSize: Int
    /// The number of tokens that the cache holds.
    let offset: Int
    /// The key bit width.
    let keyBits: Int
    /// The value bit width.
    let valueBits: Int
    /// The Sinkhorn iteration count.
    let sinkhornIterations: Int
    /// The number of complete tiles.
    let tileCount: Int
    /// The number of tokens in the raw tail.
    let tailLength: Int

    /// Reads the numbers and checks the configuration and the element-type names.
    ///
    /// - Parameter metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when the meta state is not a valid layout.
    init(metaState: [String]) throws {
        let className = "VarianceNormalizedKVCache"
        guard metaState.count == Self.legacyCount || metaState.count == Self.versionedCount
        else {
            throw KVCacheError(
                message:
                    "Corrupt prompt cache: VarianceNormalizedKVCache metadata must contain 7 legacy or 10 versioned values."
            )
        }
        let values = try promptCacheIntegers(
            Array(metaState.prefix(Self.legacyCount)), className: className)
        tileSize = values[Self.tileSizeIndex]
        offset = values[Self.offsetIndex]
        keyBits = values[Self.keyBitsIndex]
        valueBits = values[Self.valueBitsIndex]
        sinkhornIterations = values[Self.sinkhornIterationsIndex]
        tileCount = values[Self.tileCountIndex]
        tailLength = values[Self.tailLengthIndex]
        guard
            (try? VarianceNormalizedKVCacheConfiguration(
                keyBits: keyBits,
                valueBits: valueBits,
                tileSize: tileSize,
                sinkhornIterations: sinkhornIterations)) != nil,
            offset >= 0,
            tileCount >= 0,
            (0 ..< tileSize).contains(tailLength),
            metaState.count == Self.legacyCount || Self.hasValidVersionFields(metaState)
        else {
            throw KVCacheError(
                message: "Corrupt prompt cache: invalid VarianceNormalizedKVCache metadata."
            )
        }
    }

    /// True when the offset equals the tokens of the tiles and the tail, with no overflow.
    var hasConsistentOffset: Bool {
        let (tiledLength, tileLengthOverflow) = tileCount.multipliedReportingOverflow(
            by: tileSize)
        let (representedLength, offsetOverflow) = tiledLength.addingReportingOverflow(tailLength)
        return !tileLengthOverflow && !offsetOverflow && offset == representedLength
    }

    /// Checks the version and the element-type names of a versioned layout.
    ///
    /// - Parameter metaState: The saved meta state, which holds the versioned count.
    /// - Returns: True when the version is known and each element-type name is supported.
    private static func hasValidVersionFields(_ metaState: [String]) -> Bool {
        let isSupportedName = { (name: String) in
            name == noDType
                || varianceNormalizedDType(named: name).map(isSupportedVarianceNormalizedDType)
                    == true
        }
        return Int(metaState[versionIndex]) == VarianceNormalizedKVCache.metadataVersion
            && isSupportedName(metaState[keyDTypeIndex])
            && isSupportedName(metaState[valueDTypeIndex])
    }
}

/// Checks the saved values of a `ChunkedKVCache`.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateChunkedCache(state: [MLXArray], metaState: [String]) throws {
    let className = "ChunkedKVCache"
    try validatePromptCache(
        className: className, state: state, stateCounts: SavedValueCounts.keyValueStates,
        metadata: metaState, metadataCounts: SavedValueCounts.chunkedMetaStates)
    _ = try savedChunkSize(metaState)
    _ = try promptCacheInteger(
        metaState[SavedMetaStateIndex.chunkedStartPosition], className: className)
}

/// Reads the chunk size of a saved `ChunkedKVCache`.
///
/// - Parameter metaState: The saved meta state, which holds two values.
/// - Returns: The chunk size, or `nil` for a cache with no chunk size.
/// - Throws: ``KVCacheError`` when the value is not an integer or `None`.
private func savedChunkSize(_ metaState: [String]) throws -> Int? {
    let value = metaState[SavedMetaStateIndex.chunkedChunkSize]
    return value == "None" ? nil : try promptCacheInteger(value, className: "ChunkedKVCache")
}

/// Checks the saved values of an `ArraysCache` or a `MambaCache`.
///
/// The legacy meta state `[""]` holds no slot list. The restore then writes the arrays into
/// the slots in order, thus any array count fits it.
///
/// - Parameters:
///   - className: The saved class name.
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateArraysCache(
    className: String, state: [MLXArray], metaState: [String]
) throws {
    guard metaState != [""] else { return }
    let slotIndex = SavedMetaStateIndex.arraysSlotCount
    let presentIndex = SavedMetaStateIndex.arraysPresentSlots
    guard SavedValueCounts.arraysMetaStates.contains(metaState.count) else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid \(className) state or metadata shape.")
    }
    let slotCount = try promptCacheInteger(metaState[slotIndex], className: className)
    let presentSlots =
        metaState[presentIndex].isEmpty
        ? []
        : try metaState[presentIndex].split(separator: ",").map {
            try promptCacheInteger(String($0), className: className)
        }
    guard slotCount >= 0, presentSlots.count == state.count,
        Set(presentSlots).count == presentSlots.count,
        presentSlots.allSatisfy({ (0 ..< slotCount).contains($0) })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: \(className) slots do not fit its arrays.")
    }
}

/// Checks the saved values of a `TurboQuantKVCache`.
///
/// The array count depends on the key mode that the key bit width selects: 2 raw arrays in
/// every mode, 3 in raw-key mode, 5 in affine-key mode, and 4 or 5 in the standard mode.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
/// - Throws: ``KVCacheError`` when the values do not fit the class.
private func validateTurboQuantCache(state: [MLXArray], metaState: [String]) throws {
    let configuration = try TurboQuantSavedConfiguration(metaState: metaState)
    let acceptedCounts = configuration.acceptedStateCounts
    // The setter reads the token axis of the first array, and of the second array in
    // raw-key mode.
    let firstRank = 4
    let secondMinimumRank = 3
    guard
        state.isEmpty
            || (acceptedCounts.contains(state.count)
                && state[0].ndim == firstRank && state[1].ndim >= secondMinimumRank)
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid TurboQuantKVCache state or metadata shape.")
    }
}

/// The configuration numbers of a saved `TurboQuantKVCache` meta state.
private struct TurboQuantSavedConfiguration {
    /// The meta-state count: the offset, the bit width, the key and value bit widths, and the
    /// seed.
    private static let metaStateCount = 5

    /// The array count of a cache that holds raw keys and raw values, in every key mode.
    private static let rawStateCount = 2

    /// The key bit width that selects raw-key mode.
    private static let rawKeyBits = 0

    /// The key bit width that selects affine-key mode.
    private static let affineKeyBits = 8

    /// The compressed array count of raw-key mode: the raw keys, the packed values and the
    /// value norms.
    private static let rawKeyCompressedCount = 3

    /// The compressed array count of the standard mode: the packed keys, the key norms, the
    /// packed values and the value norms.
    private static let standardCompressedCount = 4

    /// The compressed array count of affine-key mode, and of the standard mode with the key
    /// calibration scale.
    private static let fiveArrayCompressedCount = 5

    /// The compressed array counts of the modes that a key bit width selects.
    private static let compressedStateCountsByKeyBits: [Int: Set<Int>] = [
        rawKeyBits: [rawKeyCompressedCount], affineKeyBits: [fiveArrayCompressedCount],
    ]

    /// The compressed array counts of the standard mode, with and without the key calibration
    /// scale.
    private static let standardCompressedStateCounts: Set<Int> = [
        standardCompressedCount, fiveArrayCompressedCount,
    ]

    /// The bit width.
    let bits: Int
    /// The key bit width.
    let keyBits: Int
    /// The value bit width.
    let valueBits: Int
    /// The seed of the rotation.
    let seed: UInt64

    /// Reads the configuration numbers.
    ///
    /// - Parameter metaState: The saved meta state.
    /// - Throws: ``KVCacheError`` when the meta state is not a valid layout.
    init(metaState: [String]) throws {
        guard metaState.count == Self.metaStateCount,
            Int(metaState[0]) != nil,
            let bits = Int(metaState[SavedMetaStateIndex.turboQuantBits]),
            let keyBits = Int(metaState[SavedMetaStateIndex.turboQuantKeyBits]),
            let valueBits = Int(metaState[SavedMetaStateIndex.turboQuantValueBits]),
            let seed = UInt64(metaState[SavedMetaStateIndex.turboQuantSeed])
        else {
            throw KVCacheError(message: "Invalid TurboQuantKVCache metaState")
        }
        (self.bits, self.keyBits, self.valueBits, self.seed) = (bits, keyBits, valueBits, seed)
    }

    /// The array counts that the state setter accepts for the key mode of this configuration.
    var acceptedStateCounts: Set<Int> {
        let compressed =
            Self.compressedStateCountsByKeyBits[keyBits] ?? Self.standardCompressedStateCounts
        return compressed.union([Self.rawStateCount])
    }
}

/// Makes the empty built-in cache that a saved layer describes. The values passed the check
/// of ``validateBuiltInCache(className:state:metaState:)``.
///
/// - Parameters:
///   - className: The saved class name.
///   - metaState: The saved meta state.
/// - Returns: The empty cache, with the configuration of the saved meta state.
/// - Throws: ``KVCacheError`` when a configuration value cannot be read.
private func makeEmptyBuiltInCache(className: String, metaState: [String]) throws -> KVCache {
    switch className {
    case "RotatingKVCache":
        return RotatingKVCache(
            maxSize: try promptCacheInteger(
                metaState[SavedMetaStateIndex.rotatingMaxSize], className: className))
    case "QuantizedKVCache":
        let values = try promptCacheIntegers(metaState, className: className)
        return QuantizedKVCache(
            groupSize: values[SavedMetaStateIndex.quantizedGroupSize],
            bits: values[SavedMetaStateIndex.quantizedBits])
    case "VarianceNormalizedKVCache":
        let layout = try VarianceNormalizedSavedLayout(metaState: metaState)
        return VarianceNormalizedKVCache(
            tileSize: layout.tileSize, keyBits: layout.keyBits, valueBits: layout.valueBits,
            sinkhornIterations: layout.sinkhornIterations)
    case "ChunkedKVCache":
        return ChunkedKVCache(chunkSize: try savedChunkSize(metaState))
    case "MambaCache":
        return MambaCache()
    case "ArraysCache":
        return ArraysCache(size: 0)
    case "TurboQuantKVCache":
        let configuration = try TurboQuantSavedConfiguration(metaState: metaState)
        return TurboQuantKVCache(
            bits: configuration.bits, keyBits: configuration.keyBits,
            valueBits: configuration.valueBits, seed: configuration.seed)
    default:
        return KVCacheSimple()
    }
}

/// Writes checked saved values into a built-in cache, in the order that its setters need.
///
/// - Parameters:
///   - state: The saved arrays.
///   - metaState: The saved meta state.
///   - cache: The built-in cache that receives the values.
private func applySavedValues(state: [MLXArray], metaState: [String], to cache: KVCache) {
    if let arrays = cache as? ArraysCache {
        arrays.restoreFromMetaState(state: state, savedMetaState: metaState)
        return
    }
    var target = cache
    if cache is VarianceNormalizedKVCache {
        // The state setter reads the tile and tail counts that the meta-state setter writes.
        target.metaState = metaState
        target.state = state
        return
    }
    if !state.isEmpty {
        target.state = state
    }
    target.metaState = metaState
}

private func validatePromptCache(
    className: String,
    state: [MLXArray],
    stateCounts: Set<Int>,
    metadata: [String],
    metadataCounts: Set<Int>
) throws {
    guard stateCounts.contains(state.count), metadataCounts.contains(metadata.count),
        state.allSatisfy({ $0.ndim == SavedValueCounts.tensorRank })
    else {
        throw KVCacheError(
            message: "Corrupt prompt cache: invalid \(className) state or metadata shape."
        )
    }
}

private func promptCacheInteger(_ value: String, className: String) throws -> Int {
    guard let value = Int(value) else {
        throw KVCacheError(
            message: "Corrupt prompt cache: \(className) metadata must contain integers."
        )
    }
    return value
}

private func promptCacheIntegers(
    _ metadata: some Collection<String>, className: String
) throws -> [Int] {
    try metadata.map { try promptCacheInteger($0, className: className) }
}
