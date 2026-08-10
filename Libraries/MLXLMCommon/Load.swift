// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The `model.safetensors.index.json` file, which maps each checkpoint key to
/// the weight file that holds it.
private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// A failure that stops the weight files of a model from being read.
package enum WeightLoadingError: LocalizedError, Equatable {
    /// The contents of the model directory cannot be listed.
    case unreadableModelDirectory(URL)

    package var errorDescription: String? {
        switch self {
        case .unreadableModelDirectory(let modelDirectory):
            return "Cannot list the contents of the model directory '\(modelDirectory.path)'."
        }
    }
}

/// Collects the URLs of the safetensor weight files of a model.
///
/// A directory that holds a `model.safetensors.index.json` file gives the file
/// names that the index names, with no repeat, in sorted order. A directory
/// with no such index file gives every `.safetensors` file in it and in its
/// subdirectories instead.
///
/// - Parameter modelDirectory: The directory that holds the model files.
/// - Returns: The URL of each weight file to load.
/// - Throws: An error when the index file cannot be read or decoded, or
///   ``WeightLoadingError/unreadableModelDirectory(_:)`` when the directory
///   cannot be listed.
package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return Set(index.weightMap.values)
            .sorted()
            .map { modelDirectory.appendingPathComponent($0) }
    }

    // A directory that does not exist still gives an enumerator, and that
    // enumerator gives no item. `nil` thus keeps its own meaning: Foundation
    // cannot make an enumerator for this URL at all.
    guard
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)
    else {
        throw WeightLoadingError.unreadableModelDirectory(modelDirectory)
    }
    return enumerator.compactMap { item -> URL? in
        guard let url = item as? URL, url.pathExtension == "safetensors" else {
            return nil
        }
        return url
    }
}

/// Decides how ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// quantizes one layer.
///
/// This is the filter that function hands to `quantize(model:filter:)`, and it
/// is a gate of two parts:
///
/// 1. The weight files must hold a `<path>.scales` array for the layer. A
///    layer with no such array keeps its high precision, whatever the
///    configuration states.
/// 2. A layer that passes part 1 takes its parameters from the per-layer plan,
///    which gives its own default to every path it does not name. A model with
///    no per-layer plan takes the single default quantization instead.
///
/// - Parameters:
///   - path: The flattened module path of the layer.
///   - weights: The loaded and sanitized weights, keyed by checkpoint key.
///   - quantization: The default quantization, when the configuration states one.
///   - perLayerQuantization: The per-layer plan, when the configuration states one.
/// - Returns: The group size, bit width and mode to quantize the layer with, or
///   `nil` to leave the layer in high precision.
package func quantizationParameters(
    forPath path: String,
    weights: [String: MLXArray],
    quantization: BaseConfiguration.Quantization?,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
    guard weights["\(path).scales"] != nil else { return nil }
    if let perLayerQuantization {
        return perLayerQuantization.quantization(layer: path)?.asTuple
    }
    return quantization?.asTuple
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights: [String: MLXArray] = [:]
    var metadata: [String: String] = [:]
    for url in try safetensorWeightURLs(in: modelDirectory) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, _ in
            quantizationParameters(
                forPath: path, weights: weights, quantization: quantization,
                perLayerQuantization: perLayerQuantization)
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
