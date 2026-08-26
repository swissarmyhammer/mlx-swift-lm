// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Concurrent weight loading

// MLX's safetensors loader is lazy: `loadArraysAndMetadata` reads only the header, and a
// tensor's bytes are read when its array is evaluated. A single `eval` of everything
// serializes that I/O and the copies into unified memory no matter how fast the disk is.
// Splitting each file into contiguous byte-balanced ranges and evaluating the ranges from
// concurrent `eval` calls overlaps read, copy, and allocation. Measured on an M4 Pro
// (14 cores, 5 GB shards, NVMe at ~6.1 GB/s sequential): the serial loader moves ~3-4.5 GB/s
// while the concurrent one reaches the disk ceiling cold (~5.9 GB/s) and >10 GB/s from the
// page cache -- a 30-45% faster cold load, about 2x warm. `F_RDADVISE`/read-ahead variants
// measured *slower* than the serial baseline because the advised I/O competes with the
// loader's own reads.

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// The tensors of the safetensors file at `url`, ordered by their position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorSpansInFileOrder(url: URL) throws -> [SafetensorSpan] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    // a header bigger than this is not a header
    guard headerLength > 0, headerLength <= 512 * 1024 * 1024 else { throw Malformed() }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, byteCount: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = (offsets[0] as? NSNumber)?.int64Value,
            let end = (offsets[1] as? NSNumber)?.int64Value,
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end - begin))
    }
    spans.sort { $0.begin < $1.begin }
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.byteCount) }
}

/// Contiguous index ranges of `byteCounts` whose byte totals are balanced around
/// `total / groupCount`, preserving order.
func contiguousLoadGroups(byteCounts: [Int64], groupCount: Int) -> [Range<Int>] {
    guard !byteCounts.isEmpty else { return [] }
    let total = byteCounts.reduce(0, +)
    guard groupCount > 1, total > 0 else { return [0 ..< byteCounts.count] }

    let groups = Int64(groupCount)
    var ranges = [Range<Int>]()
    var start = 0
    var cumulative: Int64 = 0
    var boundary: Int64 = 1
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= total * boundary / groups {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// How many concurrent evaluations to spread a model's weight loading across.
///
/// Throughput rises with concurrent readers until the disk (cold) or the memory system (warm)
/// saturates -- around 8-16 in-flight readers on Apple silicon. More workers than cores only
/// adds contention.
func weightLoadConcurrency(processorCount: Int = ProcessInfo.processInfo.activeProcessorCount)
    -> Int
{
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Lock-guarded shared state for the concurrent load.
private final class ConcurrentLoadState: @unchecked Sendable {
    private let lock = NSLock()
    private var perFile: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var firstError: Error?

    init(fileCount: Int) {
        perFile = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
    }

    func merge(file: Int, weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        perFile[file].merge(weights) { _, new in new }
        perFileMetadata[file] = metadata
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    /// Weights merged in file order (a later file overwrites a duplicate name, matching the
    /// serial loader) and the first file's metadata.
    func result() throws -> (weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        var weights = [String: MLXArray]()
        for fileWeights in perFile {
            weights.merge(fileWeights) { _, new in new }
        }
        let metadata = perFileMetadata.first { !$0.isEmpty } ?? [:]
        return (weights, metadata)
    }
}

/// Load and materialize the weights of every file in `urls`, evaluating contiguous byte
/// ranges of each file concurrently.
///
/// Each work item lazily opens its file, evaluates only its assigned tensors (forcing that
/// range's I/O inside the work item), and the results are merged in file order. A file whose
/// header cannot be parsed is loaded whole by one work item, which is exactly the serial
/// loader's behavior for that file.
func loadWeightArrays(urls: [URL]) throws -> (
    weights: [String: MLXArray], metadata: [String: String]
) {
    struct WorkItem {
        let file: Int
        let url: URL
        /// tensors this item evaluates; nil evaluates the whole file
        let names: [String]?
    }

    let items: [WorkItem] = {
        var spansPerFile = [[SafetensorSpan]?]()
        var totalBytes: Int64 = 0
        for url in urls {
            let spans = try? safetensorSpansInFileOrder(url: url)
            spansPerFile.append(spans)
            totalBytes += spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        }

        let concurrency = weightLoadConcurrency()
        let groupBytes = max(minimumBytesPerLoadGroup, totalBytes / Int64(concurrency))
        var items = [WorkItem]()
        for (file, url) in urls.enumerated() {
            if let spans = spansPerFile[file], !spans.isEmpty {
                let bytes = spans.reduce(0) { $0 + $1.byteCount }
                let groupCount = max(1, Int(bytes / groupBytes))
                for range in contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount)
                {
                    items.append(
                        WorkItem(file: file, url: url, names: spans[range].map(\.name)))
                }
            } else {
                items.append(WorkItem(file: file, url: url, names: nil))
            }
        }
        return items
    }()

    let state = ConcurrentLoadState(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: items.count) { index in
        let item = items[index]
        do {
            // Explicitly the CPU stream: `Load` has no GPU implementation and the arrays
            // land in unified memory either way. The concurrency comes from evaluating
            // disjoint groups from many threads, not from the stream itself.
            let (all, metadata) = try loadArraysAndMetadata(url: item.url, stream: .cpu)

            var selected = [String: MLXArray]()
            if let names = item.names {
                for name in names {
                    if let array = all[name] { selected[name] = array }
                }
            } else {
                selected = all
            }

            // force this range's I/O here, on this stream, in file-offset order
            if !selected.isEmpty { eval(Array(selected.values)) }
            state.merge(file: item.file, weights: selected, metadata: metadata)
        } catch {
            state.record(error: error)
        }
    }
    return try state.result()
}

// MARK: - Weight file selection

/// The `model.safetensors.index.json` file, which maps each checkpoint key to
/// the weight file that holds it.
private struct SafetensorsIndex: Decodable {
    /// The name of the weight file that holds each checkpoint key.
    let weightMap: [String: String]

    /// The name each property has in the index file.
    enum CodingKeys: String, CodingKey {
        /// The `weight_map` object of the index file.
        case weightMap = "weight_map"
    }
}

/// A failure that stops the weight files of a model from being read.
package enum WeightLoadingError: LocalizedError, Equatable {
    /// The URL of the model directory does not name a file.
    case modelDirectoryIsNotAFileURL(URL)

    /// An entry of the safetensors index names a file that is not in the model
    /// directory.
    case weightFileOutsideModelDirectory(entry: String, modelDirectory: URL)

    /// The localized description of the error, which this type gives through
    /// its `LocalizedError` conformance.
    ///
    /// Each case gives one sentence that names the directory, or the index
    /// entry, in the failure.
    package var errorDescription: String? {
        switch self {
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

/// How the safetensors files holding a model's weights are chosen.
///
/// ## See Also
/// - ``ModelConfiguration/weightFileSelection``
public enum WeightFileSelection: Sendable, Equatable {
    /// Use `model.safetensors.index.json` when it names files that exist, otherwise the
    /// conventional `model*.safetensors` (then `weight*.safetensors`) names.
    ///
    /// This is what a well-packaged checkpoint wants and it is the default.
    case automatic

    /// Load every safetensors file in the model directory.
    ///
    /// This is an escape hatch for a checkpoint whose index is known to be wrong in a way
    /// ``automatic`` cannot detect -- an index that names files that all exist but omits
    /// weights the model needs. It loads files that may not belong to this model, and a
    /// stray tensor whose name collides with one the model's `sanitize(weights:)` rewrites
    /// is loaded silently rather than reported, so prefer a model that declares its own
    /// extra files (see ``AdditionalWeightFilesProviding``) where that is possible.
    case allFilesPresent
}

/// The safetensors files in `modelDirectory` that hold the model's weights.
///
/// Only the top level of the directory is considered. Checkpoints keep auxiliary weights that
/// belong to a different module in subdirectories (for example `mlx-community/Qwen3.5-4B-OptiQ-4bit`
/// and its `optiq/mtp.safetensors`), and a nested Hugging Face snapshot cache under a local
/// checkpoint directory would otherwise be pulled in as well.
///
/// With ``WeightFileSelection/automatic`` the files are chosen in this order:
///
/// 1. The files named by `model.safetensors.index.json`, when it exists and every file it names
///    exists. The index is precise about which of several weight files belong to the model, which
///    matters for a repo that ships both a consolidated file and shards.
/// 2. The conventional `model*.safetensors` names, matching `mlx_lm.utils.load_model`. Uploads
///    regularly ship an index carried over from an unquantized source repo that names shards the
///    repo does not contain, and the convention is what those repos actually follow.
/// 3. `weight*.safetensors`, then every safetensors file present, so a directory that follows no
///    convention at all still loads.
///
/// `additionalFiles` names files the model requires that no rule above selects, for example the
/// Jina reranker's `projector.safetensors`. They are appended, so a file the index already names
/// is not loaded twice, and names that are not present are ignored.
///
/// The index comes from outside, thus each path this function gives to `FileManager` must stay
/// in the model directory. `FileManager` reads the path of a URL and gives no attention to the
/// scheme or the host, thus a URL that does not name a file walks the local file system, and an
/// index entry that holds `..` leaves the model directory. This function rejects both.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - selection: how to choose the files, see ``WeightFileSelection``
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     selected ones. See ``AdditionalWeightFilesProviding/additionalWeightFiles``.
/// - Throws: ``WeightLoadingError/modelDirectoryIsNotAFileURL(_:)`` when the directory does not
///   name a file,
///   ``WeightLoadingError/weightFileOutsideModelDirectory(entry:modelDirectory:)`` when an index
///   entry leaves the directory, or an error when the index file cannot be read or decoded.
package func safetensorWeightURLs(
    in modelDirectory: URL,
    selection: WeightFileSelection = .automatic,
    additionalFiles: [String] = []
) throws -> [URL] {
    guard modelDirectory.isFileURL else {
        throw WeightLoadingError.modelDirectoryIsNotAFileURL(modelDirectory)
    }

    let present = topLevelSafetensorURLs(in: modelDirectory)

    let selected: [URL]
    switch selection {
    case .allFilesPresent:
        selected = present
    case .automatic:
        selected = try indexedWeightURLs(in: modelDirectory) ?? conventionalWeightURLs(in: present)
    }

    var seen = Set(selected.map(\.standardizedFileURL.path))
    var urls = selected
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }
    return urls
}

/// The files named by `model.safetensors.index.json`, or `nil` when there is no index or it
/// names a file the directory does not contain.
///
/// Existence is checked against the file system rather than the top-level listing: an index may
/// legitimately map weights into a subdirectory, and that is a deliberate statement about where
/// this model's weights live rather than an unrelated file that happens to be nearby.
///
/// An entry that can name a file outside the model directory is rejected before any file is
/// read, through ``weightFileURL(forIndexEntry:in:)``.
private func indexedWeightURLs(in modelDirectory: URL) throws -> [URL]? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return nil
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let urls = try Set(index.weightMap.values)
        .sorted()
        .map { try weightFileURL(forIndexEntry: $0, in: modelDirectory) }

    guard !urls.isEmpty,
        urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return nil
    }
    return urls
}

/// The conventionally named weight files among `present`, matching `mlx_lm.utils.load_model`'s
/// `model*.safetensors` glob, with `weight*.safetensors` and then everything as fallbacks.
private func conventionalWeightURLs(in present: [URL]) -> [URL] {
    for prefix in ["model", "weight"] {
        let matches = present.filter { $0.lastPathComponent.hasPrefix(prefix) }
        if !matches.isEmpty {
            return matches
        }
    }
    return present
}

private func topLevelSafetensorURLs(in modelDirectory: URL) -> [URL] {
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    return
        contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// Decides how ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:)``
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

/// The total size of the given weight files, in bytes.
///
/// The sizes come from the file system, thus this reads no weight into memory
/// and it can run before the first weight buffer is made. A file whose size the
/// file system does not report counts as zero.
///
/// Each path is resolved first, because a `huggingface_hub` snapshot holds a
/// symbolic link to a blob for each weight file. `URLResourceValues` reads the
/// LINK, thus an unresolved path answers the size of the link and not the size
/// of the weights: measured on a snapshot of
/// `mlx-community/Llama-3.2-1B-Instruct-4bit`, 76 bytes against 745,270,382.
///
/// - Parameter weightURLs: The URL of each weight file.
/// - Returns: The sum of the sizes of the files, in bytes.
package func weightFileBytes(of weightURLs: [URL]) -> Int {
    weightURLs.reduce(0) { total, url in
        let values = try? url.resolvingSymlinksInPath()
            .resourceValues(forKeys: [.fileSizeKey])
        return total + (values?.fileSize ?? 0)
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
///
/// The function first raises the Metal wired-memory limit so that it covers the
/// weight files, through ``ModelWeightResidency``. A weight buffer joins the
/// Metal residency set when it is made, thus the limit must stand before the
/// first file is read. Without that, each decode step makes the whole weight
/// set resident again, which costs 2.10 s for each token of a 141 GiB model
/// against 0.068 s with the limit raised.
///
/// The weight files are chosen from `model.safetensors.index.json` when it names files that
/// exist, and otherwise by the conventional `model*.safetensors` names. A model can name extra
/// files it needs by conforming to ``AdditionalWeightFilesProviding``, and a caller can override
/// the choice with ``ModelConfiguration/weightFileSelection``.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic
) async throws {
    let additionalFiles = (model as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
    let weightURLs = try safetensorWeightURLs(
        in: modelDirectory,
        selection: weightFileSelection,
        additionalFiles: additionalFiles ?? [])
    await ModelWeightResidency.shared.raise(
        toCoverWeightBytes: weightFileBytes(of: weightURLs))

    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    (weights, metadata) = try loadWeightArrays(urls: weightURLs)

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
