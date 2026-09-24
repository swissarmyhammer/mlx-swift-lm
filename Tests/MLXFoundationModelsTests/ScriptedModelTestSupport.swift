// Copyright © 2026 Apple Inc.
//
// Scripted model doubles shared by the in-package `MLXFoundationModelsTests`
// target. These doubles replace weights, not behavior: the executor runs its
// real generation paths over them, on real MLX arrays, with no download and no
// network. `ToolBodyContainerReentryTests` and `CancelledGenerationDrainTests`
// both build their models from this file.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Synchronization

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

    /// Mask that keeps the low byte of a token ID when decoding.
    private static let tokenByteMask = 0xFF

    /// The token IDs that make the model emit `text` and then stop.
    static func tokenIDs(for text: String) -> [Int] {
        Array(text.utf8).map { Int($0) } + [endOfTextByte]
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let bytes = tokenIds.filter { $0 != Self.endOfTextByte }
            .map { UInt8($0 & Self.tokenByteMask) }
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

    /// The key/value heads of each cache layer. One head is enough: the
    /// caches only have to count the positions the model saw.
    private static let cacheHeadCount = 1

    /// The width of each key and value the model writes into its caches.
    private static let cacheHeadDimension = 1

    /// One key/value head for each cache layer. A model with no cache layers
    /// allocates no KV cache.
    var kvHeads: [Int] { Array(repeating: Self.cacheHeadCount, count: cacheLayerCount) }

    private let rounds: [[Int]]
    private let forwardSteps: ForwardStepCounter?
    private let forwardDelay: TimeInterval
    private let cacheLayerCount: Int
    private var roundIndex = -1
    private var step = 0

    /// Makes a model that replays `rounds` and counts its forward passes
    /// into `forwardSteps` when a counter is given.
    ///
    /// `forwardDelay` slows each forward pass by that many seconds, so a
    /// test can reproduce the decode speed of a real model. The default of
    /// zero keeps the scripted decode as fast as the arrays allow.
    ///
    /// `cacheLayerCount` gives the model that many KV cache layers. Each
    /// forward pass writes one position into each layer for each input
    /// token, thus the executor can carry the caches of a pass into the next
    /// pass. The default of zero gives the model no cache.
    init(
        rounds: [[Int]], forwardSteps: ForwardStepCounter? = nil,
        forwardDelay: TimeInterval = 0, cacheLayerCount: Int = 0
    ) {
        self.rounds = rounds
        self.forwardSteps = forwardSteps
        self.forwardDelay = forwardDelay
        self.cacheLayerCount = cacheLayerCount
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
        write(tokenCount: inputs.size, into: cache ?? [])
        let positions = Swift.max(inputs.size, 1)
        var logits = Array(
            repeating: Self.rejectedLogit,
            count: positions * Self.vocabularySize)
        // Only the last row is read as the next-token distribution.
        logits[(positions - 1) * Self.vocabularySize + nextToken()] = Self.selectedLogit
        return MLXArray(logits, [1, positions, Self.vocabularySize])
    }

    /// Writes one zero key and one zero value for each of `tokenCount` tokens
    /// into each of `caches`, thus the offset of each cache counts the tokens
    /// the model saw.
    ///
    /// - Parameters:
    ///   - tokenCount: the number of tokens in the input of the forward pass.
    ///   - caches: the caches of the forward pass.
    private func write(tokenCount: Int, into caches: [KVCache]) {
        guard tokenCount > 0 else { return }
        let keyValues = MLXArray.zeros([
            1, Self.cacheHeadCount, tokenCount, Self.cacheHeadDimension,
        ])
        for cache in caches {
            _ = cache.update(keys: keyValues, values: keyValues)
        }
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

    /// The number of arbitrary prompt token IDs. The count only has to be
    /// more than zero so the prefill path runs its normal course.
    private static let promptTokenCount: Int32 = 3

    /// Arbitrary token IDs that stand in for a rendered prompt.
    private static let promptTokens: [Int32] = Array(1 ... promptTokenCount)

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray(Self.promptTokens))
    }
}

/// A processor that renders the text of the prompt as its UTF-8 bytes, one
/// token for each byte.
///
/// A later turn of a conversation renders the text of the earlier turns
/// first, thus its tokens start with the tokens of the earlier render, as a
/// chat template render does. The executor can then reuse the cache of the
/// earlier turn.
struct PromptBytesInputProcessor: UserInputProcessor {

    /// Renders the text of the prompt of `input` as one token for each byte.
    func prepare(input: UserInput) async throws -> LMInput {
        let tokens = ScriptedByteTokenizer().encode(
            text: input.prompt.description, addSpecialTokens: false)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

/// Builds a container over the scripted doubles. No download, no weights.
///
/// `forwardSteps`, `forwardDelay` and `cacheLayerCount` pass through to
/// ``ScriptedLanguageModel/init(rounds:forwardSteps:forwardDelay:cacheLayerCount:)``.
/// `processor` renders the prompt of each pass. The default renders the same
/// tokens for every prompt.
func makeScriptedContainer(
    modelID: String, rounds: [String], forwardSteps: ForwardStepCounter? = nil,
    forwardDelay: TimeInterval = 0, cacheLayerCount: Int = 0,
    processor: any UserInputProcessor = FixedPromptInputProcessor()
) -> ModelContainer {
    let context = ModelContext(
        configuration: ModelConfiguration(id: modelID),
        model: ScriptedLanguageModel(
            rounds: rounds.map { ScriptedByteTokenizer.tokenIDs(for: $0) },
            forwardSteps: forwardSteps,
            forwardDelay: forwardDelay,
            cacheLayerCount: cacheLayerCount),
        processor: processor,
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

/// Runs one executor pass of a scripted model inside `store`, for a
/// transcript whose first entry is `sessionID`.
///
/// The scripted model holds no key/value cache, thus the pass checks nothing
/// back in.
///
/// - Parameters:
///   - store: the prompt cache store the pass binds.
///   - modelID: the model of the pass. A fresh identity keeps the
///     process-wide model cache and the shared store out of every other test.
///   - sessionID: the first entry of the transcript, which names the session.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func respondOnce(
    inside store: ExecutorPromptCacheStore, modelID: String, sessionID: String
) async throws {
    let weights = try makeScriptedWeightsDirectory()
    defer { try? FileManager.default.removeItem(at: weights) }
    let model = MLXLanguageModel(
        configuration: ModelConfiguration(id: modelID),
        capabilities: [],
        weightsLocation: { _ in weights },
        load: { _, _ in makeScriptedContainer(modelID: modelID, rounds: ["A"]) })
    let prompt = Transcript.Prompt(
        id: sessionID, segments: [.text(Transcript.TextSegment(content: "first turn"))])
    try await ScriptedExecutorPass.run(
        over: Transcript(entries: [.prompt(prompt)]), model: model, inside: store)
}

/// Runs one executor pass directly, on the task that calls it, the way a host
/// that binds a task-local around its call to the executor runs it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ScriptedExecutorPass {

    /// Runs one executor pass of `model` over `transcript` inside `store`.
    ///
    /// The pass runs on the task that calls this method, thus every
    /// task-local the caller binds reaches the executor.
    ///
    /// - Parameters:
    ///   - transcript: the transcript of the request.
    ///   - model: the model of the pass.
    ///   - store: the prompt cache store the pass binds.
    /// - Returns: the prompt tokens the pass reused from the cache its session
    ///   carried, as the usage of the pass reports them.
    /// - Throws: the error of the executor.
    @discardableResult
    static func run(
        over transcript: Transcript, model: MLXLanguageModel,
        inside store: ExecutorPromptCacheStore
    ) async throws -> Int {
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(transcript: transcript)
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside the
        // executor or every send parks it.
        let consumer = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { consumer.cancel() }

        let usage = ReusedTokenLog()
        try await MLXLanguageModel.Executor.$generationObserver.withValue({ usage.record($0) }) {
            try await ExecutorPromptCacheStore.$current.withValue(store) {
                try await executor.respond(to: request, model: model, streamingInto: channel)
            }
        }
        return usage.reusedTokenCount
    }

    /// The prompt tokens that the usage events of one pass report as reused.
    ///
    /// The executor reports the usage on the generation path, and the test
    /// reads the count from its own task, thus a mutex guards the count.
    private final class ReusedTokenLog: Sendable {
        private let count = Mutex(0)

        /// Adds the reused prompt tokens of `event` when it is a usage event.
        ///
        /// - Parameter event: one event the executor streamed.
        func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
            guard case .updateUsage(let input, _, _) = event else { return }
            count.withLock { $0 += input.cachedTokenCount }
        }

        /// The reused prompt tokens of every usage event so far.
        var reusedTokenCount: Int { count.withLock { $0 } }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
