// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Checkpoint tensors and the metadata and quantization settings that describe them.
///
/// Models can normalize a checkpoint in ``BaseLanguageModel/prepareCheckpoint(_:)`` before
/// quantization and strict parameter validation. Name mappings apply to tensors and layer
/// settings together, so a renamed layer keeps its declared precision.
public struct ModelCheckpoint {
    public var weights: [String: MLXArray]
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    /// The first nonempty file metadata, for compatibility with existing sanitizers.
    public let metadata: [String: String]
    private var weightMetadata: [String: [String: String]]

    public init(
        weights: [String: MLXArray], metadata: [String: String] = [:],
        weightMetadata: [String: [String: String]] = [:],
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.weights = weights
        self.metadata = metadata
        self.weightMetadata = weightMetadata
        self.perLayerQuantization = perLayerQuantization
    }

    /// Metadata from the file that supplied this tensor, including an empty metadata block.
    public func metadata(forWeight name: String) -> [String: String] {
        weightMetadata[name] ?? metadata
    }

    /// Rename or exclude complete module paths, including their quantization settings.
    /// A `nil` name excludes the tensor or layer. Aliases must not overwrite each other.
    public func mapNames(_ transform: (String) -> String?) throws -> Self {
        var result = self
        result.weights = [:]
        result.weightMetadata = [:]
        var sources = [String: String]()
        for name in weights.keys.sorted() {
            guard let mapped = transform(name) else { continue }
            if let previous = sources.updateValue(name, forKey: mapped) {
                throw MappingError.collision(previous, name, destination: mapped)
            }
            result.weights[mapped] = weights[name]
            result.weightMetadata[mapped] = metadata(forWeight: name)
        }
        if let perLayerQuantization {
            var settings = [String: BaseConfiguration.QuantizationOption]()
            sources = [:]
            for name in perLayerQuantization.perLayerQuantization.keys.sorted() {
                guard let mapped = transform(name) else { continue }
                if let previous = sources.updateValue(name, forKey: mapped) {
                    throw MappingError.collision(previous, name, destination: mapped)
                }
                settings[mapped] = perLayerQuantization.perLayerQuantization[name]
            }
            result.perLayerQuantization = .init(
                quantization: perLayerQuantization.quantization, perLayerQuantization: settings)
        }
        return result
    }

    public enum MappingError: LocalizedError {
        case collision(String, String, destination: String)

        public var errorDescription: String? {
            switch self {
            case .collision(let first, let second, let destination):
                "Checkpoint names '\(first)' and '\(second)' both map to '\(destination)'."
            }
        }
    }
}

extension ModelCheckpoint {
    /// The bytes of tensors that ``materializeWeights(groupByteLimit:)`` evaluates together
    /// by default.
    static let materializationGroupBytes = 1 << 30

    /// Evaluates the tensors in groups of at most `groupByteLimit` bytes, and clears the MLX
    /// buffer cache after each group.
    ///
    /// ``BaseLanguageModel/prepareCheckpoint(_:)`` can replace loaded tensors with lazy
    /// results of them, as Flint joins its projections. A lazy result holds its inputs until
    /// it evaluates, thus one evaluation of the whole model would hold every input beside
    /// every result. A group frees the inputs that only it reads, before the next group
    /// allocates its results. A tensor larger than the limit forms a group of its own, and
    /// a tensor that is already evaluated costs nothing.
    ///
    /// - Parameter groupByteLimit: The most bytes of tensors that evaluate together.
    func materializeWeights(groupByteLimit: Int = materializationGroupBytes) {
        var group = [MLXArray]()
        var groupBytes = 0
        for tensor in weights.values {
            if !group.isEmpty, groupBytes + tensor.nbytes > groupByteLimit {
                Self.evaluate(group)
                group.removeAll(keepingCapacity: true)
                groupBytes = 0
            }
            group.append(tensor)
            groupBytes += tensor.nbytes
        }
        Self.evaluate(group)
    }

    /// Evaluates `group`, then returns the buffers its freed inputs leave in the cache.
    private static func evaluate(_ group: [MLXArray]) {
        eval(group)
        Memory.clearCache()
    }
}
