// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXScriptedLM

/// Runs a prefill and greedy decode steps, and writes the last-position logits of each step.
///
/// Launched with `MLX_RANK` (plus `MLX_HOSTFILE` for ring, or `MLX_IBV_DEVICES` and
/// `MLX_JACCL_COORDINATOR` for JACCL), it joins that group and shards the model first;
/// otherwise it runs the whole model. `scripts/shard-check.sh` compares the two.
///
/// Usage: `ShardCheck <model-directory> <output.npy> [--tokens 1,2,3] [--decode-steps 8]`
@main
struct ShardCheck {

    /// Runs the check, and exits with status 1 on any error.
    static func main() async {
        do {
            let arguments = try Arguments(CommandLine.arguments.dropFirst())
            try await run(arguments)
        } catch {
            FileHandle.standardError.write(Data("ShardCheck: \(error)\n".utf8))
            exit(1)
        }
    }

    private static func run(_ arguments: Arguments) async throws {
        let group = try joinGroup()
        let rank = try group?.rank ?? 0
        let clock = ContinuousClock()

        let loadStart = clock.now
        let context = try await LLMModelFactory.shared.load(
            from: arguments.modelDirectory, using: PseudoWordTokenizerLoader())
        let loadTime = clock.now - loadStart

        let shardStart = clock.now
        if let group {
            guard let model = context.model as? TensorParallelShardable else {
                throw ShardCheckError.notShardable(String(describing: type(of: context.model)))
            }
            try model.shard(group: group)
        }
        let shardTime = clock.now - shardStart

        let run = try GreedyRun(
            model: context.model, prompt: arguments.tokens, decodeSteps: arguments.decodeSteps)
        guard rank == 0 else { return }

        try NumpyFile.write(run.logits, to: arguments.output)
        let report = Report(
            groupSize: try group?.size ?? 1, prompt: arguments.tokens, tokens: run.tokens,
            loadSeconds: loadTime.seconds, shardSeconds: shardTime.seconds,
            prefillSeconds: run.prefillTime.seconds,
            decodeSecondsPerToken: run.decodeTime.seconds / Double(arguments.decodeSteps))
        try report.write(besides: arguments.output)
        print(report.summary)
    }

    /// The group named by the environment, or nil when `MLX_RANK` is not set: JACCL when
    /// `MLX_IBV_DEVICES` is set (with `MLX_JACCL_COORDINATOR`), otherwise ring (`MLX_HOSTFILE`).
    private static func joinGroup() throws -> MLXDistributed.Group? {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MLX_RANK"] != nil || environment["MLX_HOSTFILE"] != nil else {
            return nil
        }
        let backend: MLXDistributed.Backend =
            environment["MLX_IBV_DEVICES"] != nil ? .jaccl : .ring
        return try MLXDistributed.initialize(backend: backend, strict: true)
    }
}

/// The command line of ``ShardCheck``.
struct Arguments {

    /// The prompt when the command line gives none: 32 ids well inside any Qwen vocabulary.
    static let defaultTokens = (0 ..< 32).map { 1000 + 37 * $0 }

    /// The greedy decode steps when the command line gives none.
    static let defaultDecodeSteps = 8

    /// The local Hugging Face snapshot to load.
    let modelDirectory: URL
    /// The `.npy` file of the logits; the report goes beside it as `.json`.
    let output: URL
    /// The prompt token ids.
    let tokens: [Int]
    /// The greedy decode steps after the prefill.
    let decodeSteps: Int

    /// Parses the command line, without the program name.

    init(_ arguments: ArraySlice<String>) throws {
        var positional: [String] = []
        var tokens = Self.defaultTokens
        var decodeSteps = Self.defaultDecodeSteps
        var remaining = arguments[...]
        while let argument = remaining.popFirst() {
            switch argument {
            case "--tokens":
                tokens = try Self.parseTokens(remaining.popFirst())
            case "--decode-steps":
                guard let value = remaining.popFirst().flatMap(Int.init), value > 0 else {
                    throw ShardCheckError.usage
                }
                decodeSteps = value
            default:
                positional.append(argument)
            }
        }
        guard positional.count == 2 else { throw ShardCheckError.usage }
        modelDirectory = URL(fileURLWithPath: positional[0])
        output = URL(fileURLWithPath: positional[1])
        self.tokens = tokens
        self.decodeSteps = decodeSteps
    }

    private static func parseTokens(_ value: String?) throws -> [Int] {
        let tokens = (value ?? "").split(separator: ",").compactMap { Int($0) }
        guard !tokens.isEmpty else { throw ShardCheckError.usage }
        return tokens
    }
}

/// A failure of ``ShardCheck``.
enum ShardCheckError: Error, CustomStringConvertible {
    /// The command line is wrong.
    case usage
    /// The model of this type cannot shard.
    case notShardable(String)

    var description: String {
        switch self {
        case .usage:
            "usage: ShardCheck <model-directory> <output.npy> [--tokens 1,2,3] [--decode-steps 8]"
        case .notShardable(let type):
            "\(type) does not conform to TensorParallelShardable"
        }
    }
}

/// One prefill and greedy decode steps through the cache of the model.
struct GreedyRun {

    /// The last-position logits of each step, as float32, shape `(steps, vocabulary)`.
    let logits: MLXArray

    /// The greedy token after each step.
    let tokens: [Int]

    /// The time of the prefill step.
    let prefillTime: Duration
    /// The time of every decode step together.
    let decodeTime: Duration

    /// Runs the prefill of `prompt`, then `decodeSteps` greedy steps.
    init(model: any LanguageModel, prompt: [Int], decodeSteps: Int) throws {
        let clock = ContinuousClock()
        let cache = try model.newCache(parameters: nil)
        var rows: [MLXArray] = []
        var tokens: [Int] = []
        var input = prompt
        var prefillTime = Duration.zero
        let start = clock.now
        for step in 0 ... decodeSteps {
            let text = LMInput.Text(tokens: MLXArray(input).expandedDimensions(axis: 0))
            let row = model(text, cache: cache, state: nil).logits[0, -1].asType(.float32)
            let next = argMax(row)
            eval(row, next)
            rows.append(row)
            tokens.append(next.item(Int.self))
            input = [tokens[step]]
            if step == 0 {
                prefillTime = clock.now - start
            }
        }
        self.logits = stacked(rows)
        self.tokens = tokens
        self.prefillTime = prefillTime
        self.decodeTime = clock.now - start - prefillTime
    }
}

/// The run summary written beside the logits, as JSON.
struct Report: Encodable {
    /// The processes the model ran across.
    let groupSize: Int
    /// The prompt token ids.
    let prompt: [Int]
    /// The greedy token after each step.
    let tokens: [Int]
    /// The time to load the model.
    let loadSeconds: Double
    /// The time to shard the model; zero when unsharded.
    let shardSeconds: Double
    /// The time of the prefill step.
    let prefillSeconds: Double
    /// The mean time of one decode step.
    let decodeSecondsPerToken: Double

    /// One line for the console.
    var summary: String {
        String(
            format:
                "group %d: load %.2fs, shard %.2fs, prefill %.3fs, decode %.1f ms/token, tokens %@",
            groupSize, loadSeconds, shardSeconds, prefillSeconds, decodeSecondsPerToken * 1000,
            tokens.map(String.init).joined(separator: ","))
    }

    /// Writes the report to the path of `logits` with a `.json` extension.
    func write(besides logits: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(
            to: logits.deletingPathExtension().appendingPathExtension("json"))
    }
}

/// Writes a float32 array in the `.npy` format of NumPy, version 1.0.
enum NumpyFile {

    /// The bytes the header and its length field align to.
    private static let headerAlignment = 64

    /// The magic string and version 1.0 of the format.
    private static let magic: [UInt8] = [0x93] + Array("NUMPY".utf8) + [1, 0]

    /// Writes `array`, which must be float32, to `url`.
    static func write(_ array: MLXArray, to url: URL) throws {
        let shape = array.shape.map(String.init).joined(separator: ", ")
        var header =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (\(shape)\(array.ndim == 1 ? "," : "")), }"
        // The magic, version and 2-byte length come first; the header ends with a newline.
        let prefixLength = magic.count + 2
        let padding =
            (headerAlignment - (prefixLength + header.utf8.count + 1) % headerAlignment)
            % headerAlignment
        header += String(repeating: " ", count: padding) + "\n"

        var data = Data(magic)
        let headerLength = UInt16(header.utf8.count)
        data.append(UInt8(headerLength & 0xff))
        data.append(UInt8(headerLength >> 8))
        data.append(Data(header.utf8))
        array.asArray(Float.self).withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try data.write(to: url)
    }
}

/// Loads a pseudo-word tokenizer: the check feeds token ids, so the text does not matter.
struct PseudoWordTokenizerLoader: TokenizerLoader {
    /// A pseudo-word tokenizer; `directory` is not read.
    func load(from directory: URL) async throws -> any Tokenizer {
        PseudoWordTokenizer()
    }
}

extension Duration {
    /// The duration in seconds.
    var seconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}
