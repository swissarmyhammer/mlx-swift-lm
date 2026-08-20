// Copyright © 2026 Apple Inc.
//
// Scripted model doubles shared by the in-package `MLXFoundationModelsTests`
// target. These doubles replace weights, not behavior: the executor runs its
// real generation paths over them, on real MLX arrays, with no download and no
// network. `ToolBodyContainerReentryTests` and `CancelledGenerationDrainTests`
// both build their models from this file.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXFoundationModels

/// One byte for each token, so a script is exactly its UTF-8 bytes.
///
/// The tool-call syntax the executor parses is plain ASCII, thus a byte
/// tokenizer reproduces it token for token with no vocabulary file.
struct ScriptedByteTokenizer: MLXLMCommon.Tokenizer {

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

/// A thread-safe record of the forward passes a scripted model ran.
///
/// The model writes it on the generation thread, and a test reads it from
/// its own task, thus a lock guards the counts. `begun` and `completed`
/// bracket each pass, so a test can tell a pass that is still running from
/// a pass that finished.
final class ForwardStepCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var begunCount = 0
    private var completedCount = 0

    /// Records the start of one forward pass.
    func begin() {
        lock.lock()
        begunCount += 1
        lock.unlock()
    }

    /// Records the end of one forward pass.
    func end() {
        lock.lock()
        completedCount += 1
        lock.unlock()
    }

    /// The number of forward passes the model started so far.
    var begun: Int {
        lock.lock()
        defer { lock.unlock() }
        return begunCount
    }

    /// The number of forward passes that are running right now.
    var inFlight: Int {
        lock.lock()
        defer { lock.unlock() }
        return begunCount - completedCount
    }
}

/// A model that replays one scripted token sequence for each generation round.
///
/// Each round starts at ``prepare(_:cache:state:prefill:)``, which the token
/// iterator calls exactly one time for each generation.
final class ScriptedLanguageModel: Module, MLXLMCommon.LanguageModel,
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
    private let forwardSteps: ForwardStepCounter?
    private let forwardDelay: TimeInterval
    private var roundIndex = -1
    private var step = 0

    /// Makes a model that replays `rounds` and counts its forward passes
    /// into `forwardSteps` when a counter is given.
    ///
    /// `forwardDelay` slows each forward pass by that many seconds, so a
    /// test can reproduce the decode speed of a real model. The default of
    /// zero keeps the scripted decode as fast as the arrays allow.
    init(
        rounds: [[Int]], forwardSteps: ForwardStepCounter? = nil,
        forwardDelay: TimeInterval = 0
    ) {
        self.rounds = rounds
        self.forwardSteps = forwardSteps
        self.forwardDelay = forwardDelay
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
        forwardSteps?.begin()
        defer { forwardSteps?.end() }
        if forwardDelay > 0 {
            // Blocks the generation thread the way a real forward pass does.
            Thread.sleep(forTimeInterval: forwardDelay)
        }
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
struct FixedPromptInputProcessor: UserInputProcessor {

    /// Three arbitrary token IDs. The count only has to be more than zero so
    /// the prefill path runs its normal course.
    private static let promptTokens: [Int32] = [1, 2, 3]

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray(Self.promptTokens))
    }
}

/// Builds a container over the scripted doubles. No download, no weights.
///
/// `forwardSteps` and `forwardDelay` pass through to
/// ``ScriptedLanguageModel/init(rounds:forwardSteps:forwardDelay:)``.
func makeScriptedContainer(
    modelID: String, rounds: [String], forwardSteps: ForwardStepCounter? = nil,
    forwardDelay: TimeInterval = 0
) -> ModelContainer {
    let context = ModelContext(
        configuration: ModelConfiguration(id: modelID),
        model: ScriptedLanguageModel(
            rounds: rounds.map { ScriptedByteTokenizer.tokenIDs(for: $0) },
            forwardSteps: forwardSteps,
            forwardDelay: forwardDelay),
        processor: FixedPromptInputProcessor(),
        tokenizer: ScriptedByteTokenizer())
    return ModelContainer(context: context)
}

/// A directory that satisfies ``MLXLanguageModel/availability``, which
/// tests for `config.json` at the weights location.
func makeScriptedWeightsDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("scripted-weights-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: true)
    try Data().write(to: directory.appendingPathComponent("config.json"))
    return directory
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
