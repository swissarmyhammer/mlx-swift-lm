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

    /// One token for each byte value.
    static let vocabularySize = 256

    /// Mask that keeps the low byte of a token ID when decoding.
    private static let tokenByteMask = 0xFF

    /// The token IDs that make the model emit `text` and then stop.
    static func tokenIDs(for text: String) -> [Int] {
        ScriptedByteTokenizer().encode(text: text, addSpecialTokens: false) + [endOfTextByte]
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
        guard id >= 0, id < Self.vocabularySize else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { convertIdToToken(Self.endOfTextByte) }
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
/// its own task, thus a mutex guards the counts. `begun` and `completed`
/// bracket each pass, so a test can tell a pass that is still running from
/// a pass that finished.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class ForwardStepCounter: Sendable {
    private let counts = Mutex((begun: 0, completed: 0))

    /// Records the start of one forward pass.
    func begin() {
        counts.withLock { $0.begun += 1 }
    }

    /// Records the end of one forward pass.
    func end() {
        counts.withLock { $0.completed += 1 }
    }

    /// The number of forward passes the model started so far.
    var begun: Int {
        counts.withLock { $0.begun }
    }

    /// The number of forward passes that are running right now.
    var inFlight: Int {
        counts.withLock { $0.begun - $0.completed }
    }
}

/// The shape of what the scripted models and the prompt-cache fixtures write
/// into a cache.
enum ScriptedCacheShape {

    /// The key/value heads of each cache layer. One head is enough: the
    /// caches only have to count the positions the model saw.
    static let headCount = 1

    /// The width of each key and value written into a cache.
    static let headDimension = 1

    /// The shape of the state written into each recurrent layer. One value is
    /// enough: the caches only have to count the positions the model saw.
    static let recurrentStateShape = [1, 1]
}

/// Writes `keyValues` as the keys and the values of each attention cache of
/// `caches`, and a new zero state into each recurrent cache, thus the offset of
/// each cache moves past the tokens of `keyValues`.
///
/// - Parameters:
///   - keyValues: the keys and values, of shape
///     `(1, ScriptedCacheShape.headCount, tokenCount, ScriptedCacheShape.headDimension)`.
///   - caches: the caches to write into.
/// - Returns: all the keys that the first attention cache holds after the
///   write, or nil when `caches` holds no attention cache.
@discardableResult
func writeScriptedCaches(keyValues: MLXArray, into caches: [KVCache]) -> MLXArray? {
    let tokenCount = keyValues.dim(2)
    var firstKeys: MLXArray?
    for cache in caches {
        if let recurrent = cache as? MambaCache {
            // A linear layer puts new state arrays in its slots and does
            // not write into the old ones, as Qwen 3.5 does.
            recurrent[0] = MLXArray.zeros(ScriptedCacheShape.recurrentStateShape)
            recurrent[1] = MLXArray.zeros(ScriptedCacheShape.recurrentStateShape)
            recurrent.advance(tokenCount)
        } else {
            let keys = cache.update(keys: keyValues, values: keyValues).0
            firstKeys = firstKeys ?? keys
        }
    }
    return firstKeys
}

/// Moves each cache of `caches` past `tokenCount` more tokens, the way a
/// forward pass does: one zero key and one zero value for each token into each
/// attention cache, and a new zero state into each recurrent cache.
///
/// A ledger reads the POSITION of a cache and nothing else, thus a key of the
/// right shape moves that position exactly as a prompt of the same length does.
///
/// - Parameters:
///   - caches: the caches to feed.
///   - tokenCount: the number of tokens to feed.
func feed(_ caches: [KVCache], tokenCount: Int) {
    guard tokenCount > 0 else { return }
    writeScriptedCaches(
        keyValues: MLXArray.zeros([
            1, ScriptedCacheShape.headCount, tokenCount, ScriptedCacheShape.headDimension,
        ]),
        into: caches)
}

/// A model that replays one scripted token sequence for each generation round.
///
/// Each round starts at ``prepare(_:cache:state:prefill:)``, which the token
/// iterator calls exactly one time for each generation.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class ScriptedLanguageModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{

    /// One token for each byte value.
    static let vocabularySize = ScriptedByteTokenizer.vocabularySize

    /// Logit of the scripted token. The value only has to win `argmax`, so any
    /// pair with a clear gap works.
    private static let selectedLogit: Float = 100

    /// Logit of every token the script does not select.
    private static let rejectedLogit: Float = -100

    /// One key/value head for each cache layer. A model with no cache layers
    /// allocates no KV cache.
    var kvHeads: [Int] {
        Array(repeating: ScriptedCacheShape.headCount, count: cacheLayerCount)
    }

    private let rounds: [[Int]]
    private let forwardSteps: ForwardStepCounter?
    private let forwardDelay: TimeInterval
    private let cacheLayerCount: Int
    private let recurrentLayerCount: Int
    private let weights: MLXArray?
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
    ///
    /// `recurrentLayerCount` gives the model that many recurrent layers
    /// after the KV cache layers, the way the linear layers of a Qwen 3.5
    /// model sit beside its attention layers. A recurrent layer cannot
    /// rewind, thus a model with one is a hybrid model. The default of zero
    /// gives the model no recurrent layer.
    ///
    /// `weights` is an array the model holds for as long as it lives, the way
    /// a real model holds its parameters. No forward pass reads it. The
    /// default of `nil` gives the model no weights.
    init(
        rounds: [[Int]], forwardSteps: ForwardStepCounter? = nil,
        forwardDelay: TimeInterval = 0, cacheLayerCount: Int = 0,
        recurrentLayerCount: Int = 0, weights: MLXArray? = nil
    ) {
        self.rounds = rounds
        self.forwardSteps = forwardSteps
        self.forwardDelay = forwardDelay
        self.cacheLayerCount = cacheLayerCount
        self.recurrentLayerCount = recurrentLayerCount
        self.weights = weights
        super.init()
    }

    /// One attention cache for each KV cache layer, then one recurrent cache
    /// for each recurrent layer.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let attention = try (0 ..< cacheLayerCount).map { _ in
            try makeAttentionKVCache(parameters: parameters)
        }
        let recurrent: [KVCache] = (0 ..< recurrentLayerCount).map { _ in MambaCache() }
        return attention + recurrent
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
        feed(cache ?? [], tokenCount: inputs.size)
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

    /// The number of arbitrary prompt token IDs. The count only has to be
    /// more than zero so the prefill path runs its normal course.
    private static let promptTokenCount: Int32 = 3

    /// Arbitrary token IDs that stand in for a rendered prompt.
    private static let promptTokens: [Int32] = Array(1 ... promptTokenCount)

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray(Self.promptTokens))
    }
}

/// The model input that renders `text` as its UTF-8 bytes, one token for each byte.
private func bytesInput(_ text: String) -> LMInput {
    let tokens = ScriptedByteTokenizer().encode(text: text, addSpecialTokens: false)
    return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
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
        bytesInput(input.prompt.description)
    }
}

/// A processor that renders the text of the prompt as its UTF-8 bytes, one
/// token for each byte, and then a generation prompt on a line of its own.
///
/// A chat template writes a generation prompt after the last message (for
/// Qwen 3.5, `<|im_start|>assistant\n<think>\n\n</think>\n\n`). The next
/// render writes the messages again but not that generation prompt, thus
/// the next render agrees with the earlier prompt only up to the start of
/// the generation prompt. The line break before the generation prompt
/// stands for `<|im_start|>`: the next message starts with it too.
struct GenerationPromptBytesInputProcessor: UserInputProcessor {

    /// The text that the render writes after the last message.
    let generationPrompt: String

    /// Renders the text of the prompt of `input`, a line break and
    /// ``generationPrompt``, as one token for each byte.
    func prepare(input: UserInput) async throws -> LMInput {
        bytesInput(input.prompt.description + "\n" + generationPrompt)
    }
}

/// Builds a container over the scripted doubles. No download, and no weights
/// unless `weights` gives an array.
///
/// `forwardSteps`, `forwardDelay`, `cacheLayerCount`, `recurrentLayerCount` and
/// `weights` pass through to
/// ``ScriptedLanguageModel/init(rounds:forwardSteps:forwardDelay:cacheLayerCount:recurrentLayerCount:weights:)``.
/// `processor` renders the prompt of each pass. The default renders the same
/// tokens for every prompt. `reasoningConfig` goes into the configuration of
/// the context. The default of `nil` gives a model that does not reason.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func makeScriptedContainer(
    modelID: String, rounds: [String], forwardSteps: ForwardStepCounter? = nil,
    forwardDelay: TimeInterval = 0, cacheLayerCount: Int = 0, recurrentLayerCount: Int = 0,
    weights: MLXArray? = nil,
    processor: any UserInputProcessor = FixedPromptInputProcessor(),
    reasoningConfig: ReasoningConfig? = nil
) -> ModelContainer {
    let context = ModelContext(
        configuration: ModelConfiguration(id: modelID, reasoningConfig: reasoningConfig),
        model: ScriptedLanguageModel(
            rounds: rounds.map { ScriptedByteTokenizer.tokenIDs(for: $0) },
            forwardSteps: forwardSteps,
            forwardDelay: forwardDelay,
            cacheLayerCount: cacheLayerCount,
            recurrentLayerCount: recurrentLayerCount,
            weights: weights),
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
    ///   - recurrentLayerCount: the recurrent layers of the model beside its
    ///     KV cache layer. The default of zero gives a model that is not
    ///     hybrid.
    /// - Returns: the model.
    static func make(
        weights: URL, scripts: [String],
        processor: any UserInputProcessor = PromptBytesInputProcessor(),
        reasoningConfig: ReasoningConfig? = nil, recurrentLayerCount: Int = 0
    ) -> MLXLanguageModel {
        make(
            modelID: "probe/scripted-session-\(UUID().uuidString)", weights: weights,
            scripts: scripts, cacheLayerCount: cacheLayerCount, processor: processor,
            reasoningConfig: reasoningConfig, recurrentLayerCount: recurrentLayerCount)
    }

    /// Makes the model under `modelID`, with one script for each pass.
    ///
    /// - Parameters:
    ///   - modelID: the identity of the model.
    ///   - weights: the directory that makes the model available.
    ///   - scripts: the text that each pass generates, in the order of the passes.
    ///   - cacheLayerCount: the KV cache layers of the model.
    ///   - processor: renders the prompt of each pass.
    ///   - reasoningConfig: the reasoning protocol of the model. When it is set, the model
    ///     declares `.reasoning`.
    ///   - recurrentLayerCount: the recurrent layers of the model beside its KV cache layers.
    /// - Returns: the model.
    static func make(
        modelID: String, weights: URL, scripts: [String], cacheLayerCount: Int,
        processor: any UserInputProcessor, reasoningConfig: ReasoningConfig? = nil,
        recurrentLayerCount: Int = 0
    ) -> MLXLanguageModel {
        let capabilities: [LanguageModelCapabilities.Capability] =
            reasoningConfig == nil ? [] : [.reasoning]
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID, reasoningConfig: reasoningConfig),
            capabilities: capabilities,
            weightsLocation: { _ in weights },
            load: { _, _ in
                makeScriptedContainer(
                    modelID: modelID, rounds: scripts, cacheLayerCount: cacheLayerCount,
                    recurrentLayerCount: recurrentLayerCount, processor: processor,
                    reasoningConfig: reasoningConfig)
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

/// Runs one executor pass directly, on the task that calls it, the way a host
/// that binds a task-local around its call to the executor runs it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ScriptedExecutorPass {
    /// Reports the failures of a pass that survive a release build.
    private static let logger = MLXLogger(label: "ScriptedExecutorPass")

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
    static func respondOnce(
        inside store: ExecutorPromptCacheStore, modelID: String, sessionID: String
    ) async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(
            modelID: modelID, weights: weights, scripts: ["A"], cacheLayerCount: 0,
            processor: FixedPromptInputProcessor())
        try await run(
            over: ScriptedSessionModel.transcript(firstEntryID: sessionID), model: model,
            inside: store)
    }

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
            do {
                for try await _ in channel {}
            } catch is CancellationError {
                // The pass ended and took its consumer down with it.
            } catch {
                assertionFailure("The channel of a scripted pass failed: \(error)")
                logger.error("The channel of a scripted pass failed: \(error)")
            }
        }
        defer { consumer.cancel() }

        let events = PassEventLog()
        let observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void = {
            events.record($0)
        }
        try await ExecutorPromptCacheStore.$current.withValue(store) {
            try await MLXLanguageModel.Executor.$generationObserver.withValue(observer) {
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
                collected.withLock {
                    $0.reusedTokenCount += input.cachedTokenCount
                    $0.promptTokenCount += input.totalTokenCount
                }
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

    /// Every prompt token that the usage events of the pass report: the
    /// reused tokens and the fed tokens together.
    var promptTokenCount = 0

    /// The response text of the pass, in the order the pass streamed it.
    var responseText = ""

    /// The reasoning text of the pass, in the order the pass streamed it.
    var reasoningText = ""

    /// The tool name of each tool call of the pass, in the order the pass
    /// streamed the calls.
    var toolCallNames: [String] = []
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
