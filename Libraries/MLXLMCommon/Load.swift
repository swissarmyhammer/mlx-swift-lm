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

    /// The URL of the model directory does not name a file.
    case modelDirectoryIsNotAFileURL(URL)

    /// An entry of the safetensors index names a file that is not in the model
    /// directory.
    case weightFileOutsideModelDirectory(entry: String, modelDirectory: URL)

    package var errorDescription: String? {
        switch self {
        case .unreadableModelDirectory(let modelDirectory):
            return "Cannot list the contents of the model directory '\(modelDirectory.path)'."
        case .modelDirectoryIsNotAFileURL(let modelDirectory):
            return "The model directory '\(modelDirectory.absoluteString)' is not a file URL."
        case .weightFileOutsideModelDirectory(let entry, let modelDirectory):
            return """
                The safetensors index entry '\(entry)' does not name a file in the model \
                directory '\(modelDirectory.path)'.
                """
        }
    }
}

/// Maps one entry of a safetensors index onto the model directory.
///
/// A `model.safetensors.index.json` file comes inside a model repository that a
/// person downloads, thus it is input from outside and this function does not
/// trust it. A good entry is the relative path of a file in the model
/// directory. An entry that starts at the root of the file system, and an entry
/// that holds a `..` component, can name a file outside that directory, and
/// this function rejects both. The examination is of the text of the entry
/// alone, thus it reads no file and it changes no good entry.
///
/// - Parameters:
///   - entry: One value of the `weight_map` of the index file.
///   - modelDirectory: The directory that holds the model files.
/// - Returns: The URL of the weight file in the model directory.
/// - Throws: ``WeightLoadingError/weightFileOutsideModelDirectory(entry:modelDirectory:)``
///   when the entry can name a file outside the model directory.
private func weightFileURL(forIndexEntry entry: String, in modelDirectory: URL) throws -> URL {
    let components = entry.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
    guard !entry.isEmpty, !entry.hasPrefix("/"), !components.contains("..") else {
        throw WeightLoadingError.weightFileOutsideModelDirectory(
            entry: entry, modelDirectory: modelDirectory)
    }
    return modelDirectory.appendingPathComponent(entry)
}

/// Collects the URLs of the safetensor weight files of a model.
///
/// A directory that holds a `model.safetensors.index.json` file gives the file
/// names that the index names, with no repeat, in sorted order. A directory
/// with no such index file gives every `.safetensors` file in it and in its
/// subdirectories instead.
///
/// Each path this function gives to `FileManager` must stay in the model
/// directory. `FileManager` reads the path of a URL and gives no attention to
/// the scheme or the host, thus a URL that does not name a file walks the local
/// file system, and an index entry that holds `..` leaves the model directory.
/// This function rejects both.
///
/// - Parameter modelDirectory: The directory that holds the model files.
/// - Returns: The URL of each weight file to load.
/// - Throws: ``WeightLoadingError/modelDirectoryIsNotAFileURL(_:)`` when the
///   directory does not name a file,
///   ``WeightLoadingError/weightFileOutsideModelDirectory(entry:modelDirectory:)``
///   when an index entry leaves the directory, an error when the index file
///   cannot be read or decoded, or
///   ``WeightLoadingError/unreadableModelDirectory(_:)`` when the directory
///   cannot be listed.
package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
    guard modelDirectory.isFileURL else {
        throw WeightLoadingError.modelDirectoryIsNotAFileURL(modelDirectory)
    }

    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return try Set(index.weightMap.values)
            .sorted()
            .map { try weightFileURL(forIndexEntry: $0, in: modelDirectory) }
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
