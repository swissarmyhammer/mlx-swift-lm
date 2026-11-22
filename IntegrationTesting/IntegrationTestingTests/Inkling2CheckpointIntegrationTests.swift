// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of Inkling 2 Mini (`inkling2_loop`, upstream PR
// ml-explore/mlx-swift-lm#821): greedy coherence, prefill and decode
// throughput at a short and a 4k-token context, prompt cache reuse across the
// turns of a `ChatSession`, a tool call through the `.inkling` format, and the
// peak memory of each.
//
// Each test downloads the pinned checkpoint through the Hugging Face
// downloader (or reuses the cached copy), thus none of them skips. The
// checkpoint is 16.1 GiB.
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`. Build for Release, or the numbers measure the debug
// build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2CheckpointIntegrationTests`

import BenchmarkHelpers
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - Pinned checkpoint

/// The revision of 2026-11-20 of Inkling 2 Mini at affine 4 bits.
let inkling2Mini4BitRevision = "9856dc53edead71405f8348930937c3638ff21a2"

/// Inkling 2 Mini at affine 4 bits, at ``inkling2Mini4BitRevision``. The tool-loop suites
/// load the same checkpoint.
let inkling2Mini4Bit = ModelConfiguration(
    id: "mlx-community/Inkling-2-Mini-4bit", revision: inkling2Mini4BitRevision)

/// The bytes of one GiB.
private let bytesPerGiB = 1_073_741_824.0

/// A checkpoint, loaded at most once per test process, on the first
/// ``container()`` call. The tool-loop suites share ``shared``, the 4bit checkpoint.
actor Inkling2MiniCheckpoint {
    static let shared = Inkling2MiniCheckpoint(configuration: inkling2Mini4Bit)

    /// The checkpoint this instance loads.
    private let configuration: ModelConfiguration

    /// The load the first ``container()`` call started.
    private var loading: Task<ModelContainer, Error>?

    /// An instance that loads `configuration`.
    ///
    /// - Parameter configuration: the checkpoint and its revision.
    init(configuration: ModelConfiguration) {
        self.configuration = configuration
    }

    /// The loaded checkpoint: the first call starts the download and the load,
    /// and every call shares it.
    func container() async throws -> ModelContainer {
        if let loading {
            return try await loading.value
        }
        let configuration = configuration
        let load = Task {
            try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration)
        }
        loading = load
        return try await load.value
    }
}

/// The peak memory since the last reset, in GiB.
private var peakGiB: Double { Double(Memory.peakMemory) / bytesPerGiB }

// MARK: - Tests

@Suite(.serialized, .timeLimit(.minutes(60)))
struct Inkling2CheckpointIntegrationTests {

    /// The thinking effort of the turns that measure text. Low effort keeps
    /// the answers short enough to read in the log.
    private static let lowEffort: [String: any Sendable] = ["reasoning_effort": "low"]

    /// A prompt of about 4k tokens: a generated Swift listing and a question about it. The
    /// suite makes it locally, so the measurement downloads no benchmark text. The MTP suite
    /// measures at the same prompt.
    static let longPrompt: String = {
        let functions = (0 ..< 240).map { index in
            "func step\(index)(_ x: Int) -> Int { x * \(index % 17 + 2) + \(index * 7 % 31) }"
        }
        return "Here is a Swift file:\n\n" + functions.joined(separator: "\n")
            + "\n\nWhich function multiplies by 18 and adds 3? Explain briefly."
    }()

    @Test
    func greedyCodingAnswerIsCoherent() async throws {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        Memory.peakMemory = 0
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 768, temperature: 0),
            additionalContext: Self.lowEffort)

        let turn = try await CollectedGeneration.collect(
            session.streamDetails(
                to: "Write a Python function `is_prime(n)` that returns whether n is prime. "
                    + "Reply with only the code."))

        Inkling2ResultLog.report("INKLING2", "coding reasoning: \(turn.reasoning)")
        Inkling2ResultLog.report("INKLING2", "coding answer: \(turn.text)")
        Inkling2ResultLog.report(
            "INKLING2", "coding turn: \(turn.summary), peak \(String(format: "%.2f", peakGiB)) GiB")
        #expect(turn.text.contains("def is_prime"), "answer: \(turn.text)")
        #expect(turn.text.contains("return"), "answer: \(turn.text)")
    }

    /// Prefill and decode throughput at a short prompt and at a prompt of
    /// about 4k tokens, each the median of three greedy runs after a warm-up.
    @Test
    func throughputAtShortAndLongContext() async throws {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        Memory.peakMemory = 0
        let short = try await Benchmark.llmGeneration(
            container: container,
            prompt: "Write a long, detailed story about a lighthouse keeper and a storm.",
            maxTokens: 256, runs: 3)
        short.logSummary(label: "Inkling-2-Mini-4bit short context")
        Inkling2ResultLog.report(
            "INKLING2",
            "short: prompt \(short.promptTokenCount) tok, "
                + "prefill \(String(format: "%.1f", short.prefillTokensPerSecond)) tok/s, "
                + "decode \(short.generationTokenCount) tok "
                + "\(String(format: "%.1f", short.decodeTokensPerSecond)) tok/s, "
                + "peak \(String(format: "%.2f", peakGiB)) GiB")

        Memory.peakMemory = 0
        let long = try await Benchmark.llmGeneration(
            container: container, prompt: Self.longPrompt, maxTokens: 128, runs: 3)
        long.logSummary(label: "Inkling-2-Mini-4bit 4k context")
        Inkling2ResultLog.report(
            "INKLING2",
            "long: prompt \(long.promptTokenCount) tok, "
                + "prefill \(String(format: "%.1f", long.prefillTokensPerSecond)) tok/s, "
                + "decode \(long.generationTokenCount) tok "
                + "\(String(format: "%.1f", long.decodeTokensPerSecond)) tok/s, "
                + "peak \(String(format: "%.2f", peakGiB)) GiB")

        for (label, stats) in [("short", short), ("long", long)] {
            #expect(stats.generationTokenCount > 0, "\(label) generated no tokens")
            #expect(
                stats.decodeTokensPerSecond.isFinite && stats.decodeTokensPerSecond > 0,
                "\(label) decode \(stats.decodeTokensPerSecond) tok/s")
            #expect(
                stats.prefillTokensPerSecond.isFinite && stats.prefillTokensPerSecond > 0,
                "\(label) prefill \(stats.prefillTokensPerSecond) tok/s")
        }
        #expect(long.promptTokenCount > 3_000, "long prompt \(long.promptTokenCount) tok")
    }

    /// Each follow-up turn of a `ChatSession` serves part of its prompt from
    /// the cache of the turns before it, at the default thinking effort and
    /// with thinking off.
    @Test(arguments: ["default", "none"])
    func chatSessionReusesThePromptCache(_ effort: String) async throws {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        Memory.peakMemory = 0
        let context: [String: any Sendable]? =
            effort == "default" ? nil : ["reasoning_effort": effort]
        let session = ChatSession(
            container,
            instructions: "You are a concise assistant. Answer in one or two sentences.",
            generateParameters: GenerateParameters(maxTokens: 512, temperature: 0),
            additionalContext: context)

        let questions = [
            "My name is Ada and my favourite number is 8214. Say hello.",
            "What is my favourite number?",
            "And what is my name?",
        ]
        var turns: [CollectedGeneration] = []
        for question in questions {
            let turn = try await CollectedGeneration.collect(session.streamDetails(to: question))
            Inkling2ResultLog.report(
                "INKLING2", "chat[\(effort)] turn \(turns.count): \(turn.summary) -> \(turn.text)")
            turns.append(turn)
        }
        Inkling2ResultLog.report(
            "INKLING2", "chat[\(effort)] peak \(String(format: "%.2f", peakGiB)) GiB")

        for (index, turn) in turns.enumerated().dropFirst() {
            let cached = turn.info?.cachedPromptTokenCount ?? 0
            #expect(
                cached > 0, "[\(effort)] turn \(index) reused no prompt tokens: \(turn.summary)")
        }
        #expect(turns[1].text.contains("8214"), "[\(effort)] turn 1: \(turns[1].text)")
        #expect(turns[2].text.contains("Ada"), "[\(effort)] turn 2: \(turns[2].text)")
    }

    /// The model calls a declared tool, and the `.inkling` format parses the
    /// call into a `ToolCall` with its arguments.
    @Test
    func toolCallParsesThroughTheInklingFormat() async throws {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        #expect(await container.configuration.toolCallFormat == .inkling)
        Memory.peakMemory = 0
        let weatherTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string", "description": "The city, for example Paris",
                        ] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["location"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 512, temperature: 0),
            additionalContext: Self.lowEffort, tools: [weatherTool])

        let turn = try await CollectedGeneration.collect(
            session.streamDetails(to: "What is the weather in Paris right now?"))

        Inkling2ResultLog.report(
            "INKLING2", "tool turn: \(turn.summary), calls \(turn.toolCalls), text \(turn.text)")
        Inkling2ResultLog.report("INKLING2", "tool peak \(String(format: "%.2f", peakGiB)) GiB")
        let call = try #require(turn.toolCalls.first, "no tool call; text: \(turn.text)")
        #expect(call.function.name == "get_weather")
        guard case .string(let location) = call.function.arguments["location"] else {
            Issue.record("no string location in \(call.function.arguments)")
            return
        }
        #expect(location.lowercased().contains("paris"), "location \(location)")
    }
}
