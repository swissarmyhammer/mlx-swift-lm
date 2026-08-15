// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

// MARK: - Scripted model doubles
//
// These doubles replace weights, not behavior: the executor runs its real
// tool-calling path over them. A round is one call of `respond`, thus one
// generation. The script of a round is the exact text the model emits.

/// One byte for each token, so a script is exactly its UTF-8 bytes.
///
/// The tool-call syntax the executor parses is plain ASCII, thus a byte
/// tokenizer reproduces it token for token with no vocabulary file.
private struct ScriptedByteTokenizer: MLXLMCommon.Tokenizer {

    /// Byte that ends a generation round.
    ///
    /// `0x03` (ASCII end-of-text) encodes as one UTF-8 byte, thus
    /// ``convertTokenToId(_:)`` resolves it to exactly one token ID. A
    /// multi-byte scalar would resolve to none and the generation loop would
    /// never stop.
    static let endOfTextByte = 3

    /// The token IDs that make the model emit `text` and then stop.
    static func tokenIDs(for text: String) -> [Int] {
        Array(text.utf8).map { Int($0) } + [endOfTextByte]
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let bytes = tokenIds.filter { $0 != Self.endOfTextByte }.map { UInt8($0 & 0xFF) }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < ScriptedLanguageModel.vocabularySize else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { String(UnicodeScalar(UInt8(Self.endOfTextByte))) }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A model that replays one scripted token sequence for each generation round.
///
/// Each round starts at ``prepare(_:cache:state:prefill:)``, which the token
/// iterator calls exactly one time for each generation.
private final class ScriptedLanguageModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{

    /// One token for each byte value.
    static let vocabularySize = 256

    /// Logit of the scripted token. The value only has to win `argmax`, so any
    /// pair with a clear gap works.
    private static let selectedLogit: Float = 100

    /// Logit of every token the script does not select.
    private static let rejectedLogit: Float = -100

    /// No attention layers, thus no KV cache to allocate.
    var kvHeads: [Int] { [] }

    private let rounds: [[Int]]
    private var roundIndex = -1
    private var step = 0

    init(rounds: [[Int]]) {
        self.rounds = rounds
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        roundIndex += 1
        step = 0
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = Swift.max(inputs.size, 1)
        var logits = Array(
            repeating: Self.rejectedLogit,
            count: positions * Self.vocabularySize)
        // Only the last row is read as the next-token distribution.
        logits[(positions - 1) * Self.vocabularySize + nextToken()] = Self.selectedLogit
        return MLXArray(logits, [1, positions, Self.vocabularySize])
    }

    /// The token this round emits next, or end-of-text once the script is spent.
    private func nextToken() -> Int {
        defer { step += 1 }
        guard rounds.indices.contains(roundIndex) else {
            return ScriptedByteTokenizer.endOfTextByte
        }
        let script = rounds[roundIndex]
        guard script.indices.contains(step) else {
            return ScriptedByteTokenizer.endOfTextByte
        }
        return script[step]
    }
}

/// A processor that returns the same prompt tokens for every input.
///
/// The scripted model ignores its prompt, thus the only requirement is a
/// non-empty token array of the rank an LLM processor produces.
private struct FixedPromptInputProcessor: UserInputProcessor {

    /// Three arbitrary token IDs. The count only has to be more than zero so
    /// the prefill path runs its normal course.
    private static let promptTokens: [Int32] = [1, 2, 3]

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray(Self.promptTokens))
    }
}

/// Builds a container over the scripted doubles. No download, no weights.
private func makeScriptedContainer(modelID: String, rounds: [String]) -> ModelContainer {
    let context = ModelContext(
        configuration: ModelConfiguration(id: modelID),
        model: ScriptedLanguageModel(
            rounds: rounds.map { ScriptedByteTokenizer.tokenIDs(for: $0) }),
        processor: FixedPromptInputProcessor(),
        tokenizer: ScriptedByteTokenizer())
    return ModelContainer(context: context)
}

// MARK: - Tool double

/// Records what the tool body produced from its own generation, and the names
/// of the tool calls the executor emitted.
private final class ContainerReentryLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: String?
    private var emittedNames: [String] = []

    func record(_ text: String) {
        lock.lock()
        storage = text
        lock.unlock()
    }

    func recordEmittedToolCall(named name: String) {
        lock.lock()
        emittedNames.append(name)
        lock.unlock()
    }

    /// The text the tool body generated, or `nil` when it never generated.
    var generatedText: String? {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    /// The names of the tool calls the executor sent into the channel.
    var emittedToolCallNames: [String] {
        lock.lock()
        defer { lock.unlock() }
        return emittedNames
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ContainerReentryArguments {
    @Guide(description: "Any text to look up.")
    var query: String
}

/// A tool whose body generates on the same model as the turn that called it,
/// which is what a host does when a tool ranks candidates with a model.
///
/// The body drives a second session over the same model, thus over the one
/// resident container. That is the shape the defect report describes.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct ContainerReentryTool: FoundationModels.Tool {
    let name = "probe"
    let description = "Looks up text with a second generation on the same model."

    let model: MLXLanguageModel
    let log: ContainerReentryLog

    func call(arguments: ContainerReentryArguments) async throws -> String {
        let nested = LanguageModelSession(model: model, tools: [], instructions: nil)
        let response = try await nested.respond(to: "Rank the candidates for \(arguments.query).")
        log.record(response.content)
        return response.content
    }
}

// MARK: - Turn outcome

/// What one session turn produced, carried out of the turn's own task.
private enum TurnOutcome: Sendable {
    case finished(String)
    case failed(String)
}

// MARK: - Tests

/// The executor must not hold the model container across the tool call it
/// emits. A host whose tool body generates on the same container would
/// otherwise never acquire it, and both sides would park forever.
@Suite("Tool body container re-entry")
struct ToolBodyContainerReentryTests {

    /// Upper bound for one scripted two-round turn.
    ///
    /// The turn loads no weights, downloads nothing, and generates about one
    /// hundred single-byte tokens, thus it completes far inside one second on
    /// any host. Thirty seconds is well beyond any scheduling delay or
    /// first-use Metal shader compilation, thus only a true park reaches it.
    /// The bound exists so this test FAILS instead of hanging: a hanging test
    /// blocks the whole suite.
    private static let turnTimeout = Duration.seconds(30)

    /// The three generations of the turn, in the order the model serves them,
    /// each one exactly what the model emits: the outer turn calls the tool,
    /// the tool body generates its own answer, and the outer turn concludes.
    private static let rounds = [
        #"<tool_call>{"name":"probe","arguments":{"query":"weather"}}</tool_call>"#,
        "Candidate one ranks highest.",
        "The probe answered.",
    ]

    @Test("A tool body may generate on the same model while its turn is in flight")
    func toolBodyGeneratesDuringItsTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let weights = try Self.makeWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        // A fresh identity for each run keeps the process-wide model cache and
        // its container lock out of every other test.
        let modelID = "probe/container-reentry-\(UUID().uuidString)"
        let rounds = Self.rounds
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [.toolCalling],
            weightsLocation: { _ in weights },
            load: { _, _ in makeScriptedContainer(modelID: modelID, rounds: rounds) })

        let log = ContainerReentryLog()
        let session = LanguageModelSession(
            model: model,
            tools: [ContainerReentryTool(model: model, log: log)],
            instructions: nil)

        // The turn runs in its own task and reports through a stream. A task
        // parked on the container lock cannot be cancelled, thus the timeout
        // must never await that task directly. The observer is task-local and
        // reaches that task, so it records the tool call the executor sends
        // from inside its `container.perform` block.
        let (outcomes, report) = AsyncStream<TurnOutcome>.makeStream()
        MLXLanguageModel.Executor.$generationObserver.withValue({ event in
            if case .toolCall(_, let name, _) = event {
                log.recordEmittedToolCall(named: name)
            }
        }) {
            Task {
                do {
                    let response = try await session.respond(to: "Use the probe tool.")
                    report.yield(.finished(response.content))
                } catch {
                    report.yield(.failed(String(describing: error)))
                }
                report.finish()
            }
        }

        guard let outcome = await Self.firstOutcome(from: outcomes, within: Self.turnTimeout) else {
            Issue.record(
                """
                The turn did not complete within \(Self.turnTimeout). The tool body \
                generates on the same container the turn holds, thus both sides are \
                parked: the executor emits its tool call while it still holds the \
                container, and the tool body cannot take it.
                """)
            return
        }

        switch outcome {
        case .finished(let content):
            #expect(
                log.emittedToolCallNames == ["probe"],
                "The executor did not take its tool-calling path.")
            #expect(
                log.generatedText == Self.rounds[1],
                "The tool body did not generate on the same model.")
            #expect(content == Self.rounds[2])
        case .failed(let description):
            Issue.record("The turn failed instead of completing: \(description)")
        }
    }

    /// A directory that satisfies ``MLXLanguageModel/availability``, which
    /// tests for `config.json` at the weights location.
    private static func makeWeightsDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("container-reentry-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data().write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    /// The first outcome the turn reports, or `nil` when `timeout` wins.
    ///
    /// Both children are cancellable, thus the group always finishes even when
    /// the turn itself never does.
    private static func firstOutcome(
        from outcomes: AsyncStream<TurnOutcome>, within timeout: Duration
    ) async -> TurnOutcome? {
        await withTaskGroup(of: TurnOutcome?.self) { group in
            group.addTask {
                for await outcome in outcomes { return outcome }
                return nil
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
