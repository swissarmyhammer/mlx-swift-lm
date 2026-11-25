// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXScriptedLM
import MLXVLM

/// Runs a prefill and greedy decode steps, and writes the last-position logits of each step.
///
/// Launched with `MLX_RANK` (plus `MLX_HOSTFILE` for ring, or `MLX_IBV_DEVICES` and
/// `MLX_JACCL_COORDINATOR` for JACCL), it joins that group and shards the model first;
/// otherwise it runs the whole model. `scripts/shard-check.sh` compares the two.
///
/// With `--teacher-forcing <reference.json>` it feeds the greedy tokens of that reference run
/// as the next inputs instead of its own argmax, so one flipped near tie cannot change every
/// later context. It still records its own logits and argmax at each step.
///
/// With `--announce` it runs a greedy generation instead, and writes only the report. In a
/// group, rank 0 drives a ``ClusterLanguageModel`` whose transport announces each op to every
/// rank and runs it on its ``RemoteModelExecutor``; every other rank applies the announced ops
/// to its own shard, as a cluster worker does.
///
/// Without `--tokens` the prompt is ``DefaultPrompt``, or the prompt of the reference run.
///
/// Usage: `ShardCheck <model-directory> <output.npy> [--tokens 1,2,3] [--decode-steps 8]
/// [--teacher-forcing reference.json | --announce]`
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
        let context = try await loadContext(arguments.modelDirectory)
        let loadTime = clock.now - loadStart

        let shardStart = clock.now
        if let group {
            guard let model = context.model as? TensorParallelShardable else {
                throw ShardCheckError.notShardable(String(describing: type(of: context.model)))
            }
            try model.shard(group: group)
        }
        let shardTime = clock.now - shardStart
        let timing = (load: loadTime.seconds, shard: shardTime.seconds)

        let reference: ReferenceRun?
        switch arguments.mode {
        case .announce:
            let prompt = try resolvePrompt(arguments, reference: nil)
            try runGeneration(
                arguments, prompt: prompt, model: context.model, group: group, timing: timing)
            return
        case .freeRunning:
            reference = nil
        case .teacherForced(let url):
            reference = try ReferenceRun(contentsOf: url)
        }
        let prompt = try resolvePrompt(arguments, reference: reference)
        let forcedTokens = try reference?.forcedTokens(decodeSteps: arguments.decodeSteps)
        let run = try GreedyRun(
            model: context.model, prompt: prompt, decodeSteps: arguments.decodeSteps,
            forcedTokens: forcedTokens)
        guard rank == 0 else { return }

        try NumpyFile.write(run.logits, to: arguments.output)
        let report = Report(
            groupSize: try group?.size ?? 1, prompt: prompt, tokens: run.tokens,
            forcedTokens: forcedTokens, loadSeconds: timing.load, shardSeconds: timing.shard,
            prefillSeconds: run.prefillTime.seconds,
            decodeSecondsPerToken: run.decodeTime.seconds / Double(arguments.decodeSteps))
        try report.write(besides: arguments.output)
        print(report.summary)
    }

    /// The prompt of the command line; else that of the reference run, so a teacher-forced run
    /// sees the contexts of its reference; else the default prompt of the model family.
    private static func resolvePrompt(
        _ arguments: Arguments, reference: ReferenceRun?
    ) throws -> [Int] {
        if let reference, let tokens = arguments.tokens, tokens != reference.prompt {
            throw ShardCheckError.promptDiffersFromReference
        }
        if let tokens = arguments.tokens ?? reference?.prompt {
            return tokens
        }
        return try DefaultPrompt.tokens(forModelIn: arguments.modelDirectory)
    }

    /// Runs a greedy generation: through the announce loop in a group, else on the whole model.
    /// Rank 0, or the single process, writes the report.
    private static func runGeneration(
        _ arguments: Arguments, prompt: [Int], model: any LanguageModel,
        group: MLXDistributed.Group?, timing: (load: Double, shard: Double)
    ) throws {
        let rankCount = try group?.size ?? 1
        let run: GenerationRun
        if let group {
            let executor = RemoteModelExecutor(
                model: model,
                configuration: try .init(
                    modelDirectory: arguments.modelDirectory, rankCount: rankCount))
            guard try group.rank == 0 else {
                let summary = try AnnounceFollower.follow(executor, group: group)
                print(
                    "rank \(try group.rank): \(summary.opCount) ops, \(summary.parkCount) parks, "
                        + "\(summary.failedOpCount) failed")
                return
            }
            let transport = AnnouncingTransport(executor: executor, group: group)
            run = try GenerationRun(
                model: ClusterLanguageModel.connect(over: transport), prompt: prompt,
                decodeSteps: arguments.decodeSteps)
            try transport.close()
        } else {
            run = try GenerationRun(
                model: model, prompt: prompt, decodeSteps: arguments.decodeSteps)
        }
        let report = Report(
            groupSize: rankCount, prompt: prompt, tokens: run.tokens, forcedTokens: nil,
            loadSeconds: timing.load, shardSeconds: timing.shard,
            prefillSeconds: run.prefillTime.seconds,
            decodeSecondsPerToken: run.decodeTime.seconds / Double(arguments.decodeSteps))
        try report.write(besides: arguments.output)
        print(report.summary)
    }

    /// The model of `directory` from the LLM factory, or from the VLM factory for a model type
    /// only it knows, such as Inkling 2. The check feeds token ids, so it runs text only.
    private static func loadContext(_ directory: URL) async throws -> ModelContext {
        do {
            return try await LLMModelFactory.shared.load(
                from: directory, using: PseudoWordTokenizerLoader())
        } catch ModelFactoryError.unsupportedModelType {
            return try await VLMModelFactory.shared.load(
                from: directory, using: PseudoWordTokenizerLoader())
        }
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

    /// The greedy decode steps when the command line gives none.
    static let defaultDecodeSteps = 8

    /// The local Hugging Face snapshot to load.
    let modelDirectory: URL
    /// The `.npy` file of the logits; the report goes beside it as `.json`.
    let output: URL
    /// The prompt token ids, or nil for the prompt of the reference run or of the model family.
    let tokens: [Int]?
    /// The greedy decode steps after the prefill.
    let decodeSteps: Int
    /// What the run feeds and records.
    let mode: Mode

    /// What a ``ShardCheck`` run feeds and records.
    enum Mode {
        /// The logits check, feeding the run's own argmax.
        case freeRunning
        /// The logits check, feeding the greedy tokens of the reference report at the URL.
        case teacherForced(reference: URL)
        /// A greedy generation through the announce loop; only the report is written.
        case announce
    }

    /// Parses the command line, without the program name.
    init(_ arguments: ArraySlice<String>) throws {
        var positional: [String] = []
        var tokens: [Int]?
        var decodeSteps = Self.defaultDecodeSteps
        var mode = Mode.freeRunning
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
            case "--teacher-forcing":
                guard let path = remaining.popFirst() else { throw ShardCheckError.usage }
                try mode.choose(.teacherForced(reference: URL(fileURLWithPath: path)))
            case "--announce":
                try mode.choose(.announce)
            default:
                positional.append(argument)
            }
        }
        guard positional.count == 2 else { throw ShardCheckError.usage }
        modelDirectory = URL(fileURLWithPath: positional[0])
        output = URL(fileURLWithPath: positional[1])
        self.tokens = tokens
        self.decodeSteps = decodeSteps
        self.mode = mode
    }

    private static func parseTokens(_ value: String?) throws -> [Int] {
        let tokens = (value ?? "").split(separator: ",").compactMap { Int($0) }
        guard !tokens.isEmpty else { throw ShardCheckError.usage }
        return tokens
    }
}

extension Arguments.Mode {
    /// Takes `mode` when no flag chose a mode yet: a generation feeds its own tokens, so it
    /// cannot be teacher-forced.
    ///
    /// - Throws: ``ShardCheckError/usage`` when a mode was already chosen.
    mutating func choose(_ mode: Self) throws {
        guard case .freeRunning = self else { throw ShardCheckError.usage }
        self = mode
    }
}

/// A failure of ``ShardCheck``.
enum ShardCheckError: Error, CustomStringConvertible {
    /// The command line is wrong.
    case usage
    /// The model of this type cannot shard.
    case notShardable(String)
    /// No default prompt is known for the tokenizer of this model type.
    case noDefaultPrompt(modelType: String)
    /// The prompt of the command line is not the prompt of the reference run.
    case promptDiffersFromReference
    /// The reference run has fewer tokens than the decode steps need.
    case referenceTooShort(tokenCount: Int, decodeSteps: Int)

    var description: String {
        switch self {
        case .usage:
            "usage: ShardCheck <model-directory> <output.npy> [--tokens 1,2,3] [--decode-steps 8] "
                + "[--teacher-forcing reference.json | --announce]"
        case .notShardable(let type):
            "\(type) does not conform to TensorParallelShardable"
        case .noDefaultPrompt(let modelType):
            "no default prompt for model type \(modelType); pass --tokens"
        case .promptDiffersFromReference:
            "--tokens differs from the prompt of the --teacher-forcing reference"
        case .referenceTooShort(let tokenCount, let decodeSteps):
            "the reference has \(tokenCount) tokens; \(decodeSteps) decode steps need that many"
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
    ///
    /// - Parameters:
    ///   - model: The model, sharded or whole.
    ///   - prompt: The prompt token ids.
    ///   - decodeSteps: The decode steps after the prefill.
    ///   - forcedTokens: The input after each step but the last, or nil to feed the argmax.
    init(model: any LanguageModel, prompt: [Int], decodeSteps: Int, forcedTokens: [Int]?) throws {
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
            // After the last step the input goes unused, and the forced tokens have ended.
            if let forcedTokens, step < forcedTokens.count {
                input = [forcedTokens[step]]
            } else {
                input = [tokens[step]]
            }
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
    /// The input fed after each step but the last when teacher-forced; absent when the run fed
    /// its own argmax.
    let forcedTokens: [Int]?
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

/// The report of a reference run, read to teacher-force another run.
struct ReferenceRun: Decodable {
    /// The prompt token ids of the reference.
    let prompt: [Int]
    /// The greedy token after each step of the reference.
    let tokens: [Int]

    /// Reads the report at `url`.
    init(contentsOf url: URL) throws {
        self = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }

    /// The inputs after each of the first `decodeSteps` steps.
    ///
    /// - Throws: ``ShardCheckError/referenceTooShort(tokenCount:decodeSteps:)`` when the
    ///   reference ran fewer steps.
    func forcedTokens(decodeSteps: Int) throws -> [Int] {
        guard tokens.count >= decodeSteps else {
            throw ShardCheckError.referenceTooShort(
                tokenCount: tokens.count, decodeSteps: decodeSteps)
        }
        return Array(tokens.prefix(decodeSteps))
    }
}

/// The default prompt: one English sentence, pre-tokenized for the tokenizer of each model
/// family that can shard. Real text keeps the greedy distributions peaked; made-up ids invite
/// near ties that bf16 noise flips.
enum DefaultPrompt {

    /// The text the ids spell.
    static let text =
        "The lighthouse keeper climbed the narrow stairs every evening to light the lamp, and "
        + "from the top he watched the fishing boats return safely to the harbor before the "
        + "storm arrived."

    /// ``text`` as encoded by the `tokenizer.json` of each model type, without special tokens.
    private static let tokensByModelType: [String: [Int]] = [
        // Qwen3.5-2B and Qwen3.8-27B (4bit, mxfp4) encode it the same.
        "qwen3_5": [
            760, 324, 55965, 51624, 42942, 279, 14590, 30129, 1396, 11129, 310, 3000, 279, 27056,
            11, 321, 494, 279, 1844, 551, 14936, 279, 19340, 30592, 460, 20359, 310, 279, 54517,
            1518, 279, 13071, 11136, 13,
        ],
        // Inkling-2-Mini (4bit, oQ4e-mtp).
        "inkling2_loop": [
            371, 186077, 133997, 220, 118570, 158314, 2537, 149876, 161422, 220, 31604, 82,
            150710, 171498, 149882, 150932, 149876, 169167, 11, 149880, 149920, 149876, 150769,
            150026, 182047, 149876, 184221, 167627, 82, 150583, 188256, 149882, 149876, 220,
            36598, 150463, 149876, 168566, 167617, 13,
        ],
    ]

    /// The ids of ``text`` for the model type in the `config.json` of `directory`.
    ///
    /// - Throws: ``ShardCheckError/noDefaultPrompt(modelType:)`` for a model type whose
    ///   tokenizer has no ids here, or the error of reading the configuration.
    static func tokens(forModelIn directory: URL) throws -> [Int] {
        let configuration = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let tokens = tokensByModelType[configuration.modelType] else {
            throw ShardCheckError.noDefaultPrompt(modelType: configuration.modelType)
        }
        return tokens
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
