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
/// tokens for every prompt. `reasoningConfig` goes into the configuration of
/// the context. The default of `nil` gives a model that does not reason.
func makeScriptedContainer(
    modelID: String, rounds: [String], forwardSteps: ForwardStepCounter? = nil,
    forwardDelay: TimeInterval = 0, cacheLayerCount: Int = 0,
    processor: any UserInputProcessor = FixedPromptInputProcessor(),
    reasoningConfig: ReasoningConfig? = nil
) -> ModelContainer {
    let context = ModelContext(
        configuration: ModelConfiguration(id: modelID, reasoningConfig: reasoningConfig),
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

/// A scripted model with one KV cache layer, and the transcripts of the
/// turns of one session. A later turn renders the text of the earlier turns
/// first, thus it can reuse the cache of an earlier turn.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ScriptedSessionModel {

    /// The KV cache layers of the model. One layer is enough for a pass to
    /// check a cache in.
    static let cacheLayerCount = 1

    /// The most passes one model runs. Each pass replays one script round.
    static let maximumPassCount = 3

    /// The text each pass generates before it stops.
    static let scriptedResponse = "A"

    /// The thinking that each pass of a reasoning model writes before
    /// ``scriptedResponse``.
    static let scriptedThinking = "plan"

    /// Makes the model under a fresh identity, thus the process-wide model
    /// cache keeps it apart from every other test.
    ///
    /// - Parameters:
    ///   - weights: the directory that makes the model available.
    ///   - processor: renders the prompt of each pass.
    ///   - reasoningConfig: the reasoning protocol of the model. When it is
    ///     set, the model declares `.reasoning`, and each pass writes
    ///     ``scriptedThinking`` between the delimiters of the protocol before
    ///     ``scriptedResponse``. The default of `nil` gives a model that does
    ///     not reason.
    /// - Returns: the model.
    static func make(
        weights: URL, processor: any UserInputProcessor = PromptBytesInputProcessor(),
        reasoningConfig: ReasoningConfig? = nil
    ) -> MLXLanguageModel {
        make(
            weights: weights,
            scripts: Array(
                repeating: script(reasoningConfig: reasoningConfig), count: maximumPassCount),
            processor: processor, reasoningConfig: reasoningConfig)
    }

    /// Makes the model under a fresh identity, with one script for each pass.
    ///
    /// Pass `n` of the model replays `scripts[n]`. A pass past the last script
    /// generates nothing.
    ///
    /// - Parameters:
    ///   - weights: the directory that makes the model available.
    ///   - scripts: the text that each pass generates, in the order of the
    ///     passes.
    ///   - processor: renders the prompt of each pass.
    ///   - reasoningConfig: the reasoning protocol of the model. When it is
    ///     set, the model declares `.reasoning`. The default of `nil` gives a
    ///     model that does not reason.
    /// - Returns: the model.
    static func make(
        weights: URL, scripts: [String],
        processor: any UserInputProcessor = PromptBytesInputProcessor(),
        reasoningConfig: ReasoningConfig? = nil
    ) -> MLXLanguageModel {
        let modelID = "probe/scripted-session-\(UUID().uuidString)"
        let capabilities: [LanguageModelCapabilities.Capability] =
            reasoningConfig == nil ? [] : [.reasoning]
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID, reasoningConfig: reasoningConfig),
            capabilities: capabilities,
            weightsLocation: { _ in weights },
            load: { _, _ in
                makeScriptedContainer(
                    modelID: modelID, rounds: scripts, cacheLayerCount: cacheLayerCount,
                    processor: processor, reasoningConfig: reasoningConfig)
            })
    }

    /// The text that each pass of the model generates.
    ///
    /// - Parameter reasoningConfig: the reasoning protocol of the model, or
    ///   `nil` for a model that does not reason.
    /// - Returns: ``scriptedResponse``, after ``scriptedThinking`` between the
    ///   delimiters of `reasoningConfig` when it is set.
    private static func script(reasoningConfig: ReasoningConfig?) -> String {
        guard let reasoningConfig else { return scriptedResponse }
        return reasoningConfig.startDelimiter + scriptedThinking + reasoningConfig.endDelimiter
            + scriptedResponse
    }

    /// A transcript of `turns` prompts whose first entry identifier is
    /// `firstEntryID`.
    ///
    /// The render of a transcript of more turns starts with the render of a
    /// transcript of fewer turns, thus a later pass can reuse the cache of an
    /// earlier pass.
    ///
    /// - Parameters:
    ///   - firstEntryID: the identifier of the first entry.
    ///   - turns: the number of prompts.
    /// - Returns: the transcript.
    static func transcript(firstEntryID: String, turns: Int = 1) -> Transcript {
        let first = Transcript.Prompt(
            id: firstEntryID, segments: [.text(Transcript.TextSegment(content: "first turn"))])
        let later = (1 ..< turns).map { turn in
            Transcript.Entry.prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "turn \(turn)"))]))
        }
        return Transcript(entries: [.prompt(first)] + later)
    }
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
        try await respond(over: transcript, model: model, inside: store).reusedTokenCount
    }

    /// Runs one executor pass of `model` over `transcript` inside `store`, and
    /// gives what the pass streamed.
    ///
    /// The pass runs on the task that calls this method, thus every
    /// task-local the caller binds reaches the executor.
    ///
    /// - Parameters:
    ///   - transcript: the transcript of the request.
    ///   - model: the model of the pass.
    ///   - store: the prompt cache store the pass binds.
    /// - Returns: the reused prompt tokens, the response text, the reasoning
    ///   text and the tool calls of the pass.
    /// - Throws: the error of the executor.
    static func respond(
        over transcript: Transcript, model: MLXLanguageModel,
        inside store: ExecutorPromptCacheStore
    ) async throws -> ScriptedPassResult {
        try await respond(
            to: makeExecutorRequest(transcript: transcript), model: model, inside: store)
    }

    /// Runs one executor pass of `model` for `request` inside `store`, and
    /// gives what the pass streamed.
    ///
    /// The pass runs on the task that calls this method, thus every
    /// task-local the caller binds reaches the executor.
    ///
    /// - Parameters:
    ///   - request: the request of the pass: its transcript, and its schema,
    ///     tools and options when the pass needs them.
    ///   - model: the model of the pass.
    ///   - store: the prompt cache store the pass binds.
    /// - Returns: the reused prompt tokens, the response text, the reasoning
    ///   text and the tool calls of the pass.
    /// - Throws: the error of the executor.
    static func respond(
        to request: LanguageModelExecutorGenerationRequest, model: MLXLanguageModel,
        inside store: ExecutorPromptCacheStore
    ) async throws -> ScriptedPassResult {
        let executor = try makeMLXExecutor(for: model)
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside the
        // executor or every send parks it.
        let consumer = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { consumer.cancel() }

        let events = PassEventLog()
        try await MLXLanguageModel.Executor.$generationObserver.withValue({ events.record($0) }) {
            try await ExecutorPromptCacheStore.$current.withValue(store) {
                try await executor.respond(to: request, model: model, streamingInto: channel)
            }
        }
        return events.result
    }

    /// The reused prompt tokens, the response text, the reasoning text and the
    /// tool calls of the events of one pass.
    ///
    /// The executor reports the events on the generation path, and the test
    /// reads the result from its own task, thus a mutex guards the result.
    private final class PassEventLog: Sendable {
        private let collected = Mutex(ScriptedPassResult())

        /// Adds the reused prompt tokens of `event` when it is a usage event,
        /// its text when it is response text or reasoning text, and its tool
        /// name when it is a tool call.
        ///
        /// - Parameter event: one event the executor streamed.
        func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
            switch event {
            case .updateUsage(let input, _, _):
                collected.withLock { $0.reusedTokenCount += input.cachedTokenCount }
            case .appendText(let text, _, .response):
                collected.withLock { $0.responseText += text }
            case .appendText(let text, _, .reasoning):
                collected.withLock { $0.reasoningText += text }
            case .toolCall(_, let name, _):
                collected.withLock { $0.toolCallNames.append(name) }
            case .updateMetadata, .completion:
                break
            }
        }

        /// What the events so far give.
        var result: ScriptedPassResult { collected.withLock { $0 } }
    }
}

/// What one scripted executor pass streamed.
struct ScriptedPassResult: Sendable {

    /// The prompt tokens that the usage events of the pass report as reused.
    var reusedTokenCount = 0

    /// The response text of the pass, in the order the pass streamed it.
    var responseText = ""

    /// The reasoning text of the pass, in the order the pass streamed it.
    var reasoningText = ""

    /// The tool name of each tool call of the pass, in the order the pass
    /// streamed the calls.
    var toolCallNames: [String] = []
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
