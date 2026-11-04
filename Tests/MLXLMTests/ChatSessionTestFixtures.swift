// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMTestSupport
import MLXNN
import MLXOptimizers
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

/// The models, tokenizers, processors and helpers that the ChatSession test classes share.
///
/// A test class adopts it to reach the helpers with `Self.` and `self.`, as the tests were
/// written when they shared one class.
protocol ChatSessionFixtures {}

/// The largest number of tokens a fixture generation produces.
private let fixtureMaxTokens = 50
/// The quantization group size of the fixture model.
private let fixtureQuantizationGroupSize = 64
/// The quantization bit width of the fixture model.
private let fixtureQuantizationBits = 4

/// The end-of-sequence and unknown token ids of the fixture tokenizers.
enum FixtureSpecialToken {
    /// The end-of-sequence token id.
    static let endOfSequence = 101
    /// The unknown token id.
    static let unknown = 102
}

extension ChatSessionFixtures {
    static func makeStateProducingModel() -> ModelContext {
        let base = makeModel()
        guard let inner = base.model as? Gemma3TextModel else {
            fatalError("expected the test model to be a Gemma3TextModel")
        }
        return .init(
            configuration: base.configuration,
            model: StateProducingModel(inner),
            processor: base.processor,
            tokenizer: base.tokenizer)
    }

    static func makeModel(
        processor: any UserInputProcessor,
        configuration: ModelConfiguration,
        tokenizer: any Tokenizer
    )
        -> ModelContext
    {
        let config = Gemma3TextConfiguration.smallConfiguration(
            vocabularySize: 100)
        let model = Gemma3TextModel(config)
        quantize(
            model: model, groupSize: fixtureQuantizationGroupSize, bits: fixtureQuantizationBits)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        return .init(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)
    }

    static func makeModel(processor: TestInputProcessor = TestInputProcessor())
        -> ModelContext
    {
        makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        Self.makeModel(processor: processor)
    }

    func model(processor: MediaAwareInputProcessor) -> ModelContext {
        Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    func model(
        processor: MediaAwareInputProcessor, splitting: Bool, declinesToSplit: Bool = false
    ) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        if splitting {
            context.model = SplittingLanguageModel(
                context.model, declinesToSplit: declinesToSplit)
        }
        return context
    }

    func model(processor: MaskedInputProcessor) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        context.model = MaskTolerantLanguageModel(context.model)
        return context
    }

    func collectGeneration(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var completionInfo: GenerateCompletionInfo?
        for try await item in stream {
            if let chunk = item.chunk {
                text += chunk
            }
            if let info = item.info {
                completionInfo = info
            }
        }
        return (text, try XCTUnwrap(completionInfo))
    }

    /// The values a realized cache holds, read out so they can leave the session.
    static func cachedValues(_ cache: [KVCache]?) -> [[Float]] {
        (cache ?? []).flatMap(\.state).map { $0.asType(.float32).asArray(Float.self) }
    }

    var generationParameters: GenerateParameters {
        GenerateParameters(maxTokens: fixtureMaxTokens)
    }

    var targetLength: Int { 1 }

    func rejectionSession(
        output: String,
        messageGenerator: any MessageGenerator = DefaultMessageGenerator(),
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) -> ChatSession {
        let tokenizer = LiteralTokenizer(output: output)
        let configuration = ModelConfiguration(
            id: "rejected-tool-call-test",
            eosTokenIds: [LiteralTokenizer.endOfSequence],
            toolCallFormat: .json)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: messageGenerator)
        let outputTokens = tokenizer.tokens(for: output) + [LiteralTokenizer.endOfSequence]
        let components = GenerationComponents {
            ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: outputTokens))
        }
        return ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(
                maxTokens: outputTokens.count + 1, temperature: 0),
            components: components,
            tools: tools,
            toolDispatch: toolDispatch)
    }

}

struct RecordedMessage: Equatable, Sendable {
    var role: Chat.Message.Role
    var content: String
}

struct RecordingMessageGenerator: MessageGenerator {
    let continuation: AsyncStream<[RecordedMessage]>.Continuation

    func generate(messages: [Chat.Message]) -> [Message] {
        continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

        return DefaultMessageGenerator().generate(messages: messages)
    }
}

/// Produces a transcript token stream where a rendered follow-up is an
/// exact extension of the prompt and generated tokens from the prior turn.
struct PrefixPreservingTokenizer: Tokenizer {
    /// The token ids of the transcript markers, and the scalars of the text encoding.
    private enum Marker {
        /// The first private-use scalar, which encodes token id 0.
        static let privateUseBase = 0xE000
        /// The scalar after the last private-use scalar that decodes to a token id.
        static let privateUseEnd = 0xE064
        /// The number of token ids that ordinary text maps onto.
        static let textTokenCount = 80
        /// The token that opens an assistant message, and the generation prompt.
        static let assistant = 94
        /// The token that closes a message that is not from the assistant.
        static let endOfMessage = 95
        /// The token that opens a system message.
        static let system = 96
        /// The token that opens a user message.
        static let user = 97
        /// The token that opens a tool message.
        static let tool = 98
        /// The token that opens a message of any other role.
        static let otherRole = 99
        /// The prefix that a rewrite writes for the first turn.
        static let firstTurnRewrite = 92
        /// The prefix that a rewrite writes for a continuation.
        static let continuationRewrite = 93
    }

    let renderedLengthContinuation: AsyncStream<Int>.Continuation
    var rewritesPrefixOnContinuation = false
    var rewritesCachedTailOnContinuation = false

    var bosToken: String? = nil
    var eosToken: String? = nil
    var unknownToken: String? = nil
    var eosTokenId: Int? { FixtureSpecialToken.endOfSequence }
    var unknownTokenId: Int? { FixtureSpecialToken.unknown }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            if (Marker.privateUseBase ..< Marker.privateUseEnd).contains(Int(scalar.value)) {
                return Int(scalar.value) - Marker.privateUseBase
            }
            return Int(scalar.value) % Marker.textTokenCount + 1
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(
            String.UnicodeScalarView(
                tokenIds.compactMap { UnicodeScalar(Marker.privateUseBase + $0) }))
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? {
        UnicodeScalar(Marker.privateUseBase + id).map(String.init)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens: [Int] = []
        for message in messages {
            let role = message["role"] as? String
            let content = message["content"] as? String ?? ""
            switch role {
            case "system":
                tokens.append(Marker.system)
            case "user":
                tokens.append(Marker.user)
            case "tool":
                tokens.append(Marker.tool)
            case "assistant":
                tokens.append(Marker.assistant)
            default:
                tokens.append(Marker.otherRole)
            }
            tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
            if role != "assistant" {
                tokens.append(Marker.endOfMessage)
            }
        }
        tokens.append(Marker.assistant)  // assistant generation marker
        if rewritesPrefixOnContinuation {
            tokens.insert(
                messages.count > 1 ? Marker.continuationRewrite : Marker.firstTurnRewrite, at: 0)
        }
        if rewritesCachedTailOnContinuation, let marker = tokens.firstIndex(of: Marker.assistant) {
            tokens[marker] =
                messages.count > 1 ? Marker.continuationRewrite : Marker.firstTurnRewrite
        }

        renderedLengthContinuation.yield(tokens.count)
        return tokens
    }
}

struct MediaAwareInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration = ModelConfiguration(id: "test")
    let messageGenerator = DefaultMessageGenerator()

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext)
        let image =
            input.images.isEmpty
            ? nil : LMInput.ProcessedImage(pixels: MLXArray([Float(0)]))
        return LMInput(text: .init(tokens: MLXArray(tokens)), image: image)
    }
}

struct MaskedInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration = ModelConfiguration(id: "test")
    let messageGenerator = DefaultMessageGenerator()

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext)
        let mask = MLXArray(Array(repeating: 1, count: tokens.count))
            .reshaped(1, tokens.count)
        return LMInput(text: .init(tokens: MLXArray(tokens), mask: mask))
    }
}

/// The synthetic text model does not consume attention masks itself.
/// Strip the mask only after `ChatSession` has made its cache-reuse
/// decision so the regression test can exercise that decision with a
/// valid masked processor input.
final class MaskTolerantLanguageModel: Module, LanguageModel {
    let base: any LanguageModel

    init(_ base: any LanguageModel) {
        self.base = base
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try base.prepare(
            LMInput(tokens: input.text.tokens),
            cache: cache,
            state: state,
            prefill: prefill)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters)
    }
}

/// A model that opts into append-only media cache reuse.
///
/// The synthetic text model underneath cannot consume a media payload, so the
/// split keeps the token suffix and the payload is stripped in `prepare` -- the
/// same shape as `MaskTolerantLanguageModel`. What these tests measure is
/// `ChatSession`'s decision, which is made before the model is called.
final class SplittingLanguageModel: Module, LanguageModel, PreparedInputSplitting {
    let base: any LanguageModel
    let declinesToSplit: Bool

    init(_ base: any LanguageModel, declinesToSplit: Bool = false) {
        self.base = base
        self.declinesToSplit = declinesToSplit
        super.init()
    }

    func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
        -> LMInput?
    {
        guard !declinesToSplit else { return nil }
        let ids = input.text.tokens.asArray(Int.self)
        guard prefixTokenCount > 0, prefixTokenCount < ids.count else { return nil }
        return LMInput(
            text: .init(tokens: MLXArray(Array(ids[prefixTokenCount...]))),
            image: input.image)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try base.prepare(
            LMInput(tokens: input.text.tokens),
            cache: cache,
            state: state,
            prefill: prefill)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters)
    }
}

/// A conformer that honors the protocol's shape but not its contract: it returns
/// a suffix starting `extraDrop` tokens past the boundary it was handed. It never
/// returns `nil`, so a nil-only check would let it through.
final class MisSplittingLanguageModel: Module, LanguageModel,
    PreparedInputSplitting
{
    let base: any LanguageModel
    let extraDrop: Int

    init(_ base: any LanguageModel, extraDrop: Int) {
        self.base = base
        self.extraDrop = extraDrop
        super.init()
    }

    func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
        -> LMInput?
    {
        let ids = input.text.tokens.asArray(Int.self)
        let wrongStart = prefixTokenCount + extraDrop
        guard wrongStart > 0, wrongStart < ids.count else { return nil }
        return LMInput(
            text: .init(tokens: MLXArray(Array(ids[wrongStart...]))),
            image: input.image)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try base.prepare(
            LMInput(tokens: input.text.tokens),
            cache: cache,
            state: state,
            prefill: prefill)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters)
    }
}

/// Counts `KVCache.copy()` calls across every cache a model creates.
final class CopyCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.withLock { count += 1 }
    }

    var value: Int {
        lock.withLock { count }
    }
}

final class CopyCountingCache: KVCacheSimple {
    let copies: CopyCounter

    init(copies: CopyCounter) {
        self.copies = copies
        super.init()
    }

    override func copy() -> any KVCache {
        copies.increment()
        return super.copy()
    }
}

/// Serves `base` over caches that count their copies.
final class CopyCountingLanguageModel: Module, LanguageModel {
    let base: any LanguageModel
    let copies: CopyCounter

    init(_ base: any LanguageModel, copies: CopyCounter) {
        self.base = base
        self.copies = copies
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try base.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters).map { _ in
            CopyCountingCache(copies: copies)
        }
    }
}

struct EmptyChatTemplateTokenizer: Tokenizer {
    var bosToken: String? = nil
    var eosToken: String? = nil
    var unknownToken: String? = nil
    var eosTokenId: Int? { FixtureSpecialToken.endOfSequence }
    var unknownTokenId: Int? { FixtureSpecialToken.unknown }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

struct UnexpectedDraftModelLoadError: Error {}

struct PreparedImageInputProcessor: UserInputProcessor {
    private let base = TestInputProcessor()

    func prepare(input: UserInput) throws -> LMInput {
        let text = try base.prepare(input: input).text
        return LMInput(
            text: text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
    }
}

/// Attaches media to the first prepared turn only. That turn refuses speculation and drops
/// the draft cache; later text-only turns are what should be able to pick it back up.
///
/// `@unchecked Sendable` because the counter is serialised by the lock.
final class FirstTurnImageInputProcessor: UserInputProcessor, @unchecked Sendable {
    private let base: TestInputProcessor
    private let lock = NSLock()
    private var prepared = 0

    init(base: TestInputProcessor) {
        self.base = base
    }

    private func consumeFirst() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        prepared += 1
        return prepared == 1
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let text = try base.prepare(input: input).text
        guard consumeFirst() else { return LMInput(text: text) }
        return LMInput(
            text: text,
            image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
    }
}

/// A model that hands back model state on every prefill, the way the Qwen
/// vision families and GLM-OCR hand back an M-RoPE anchor. The state is
/// what a session has to carry across turns and persist with its cache;
/// this stands in for a VLM without needing image inputs.
final class StateProducingModel: Module, LanguageModel, KVCacheDimensionProvider {
    static let anchorKey = LMOutput.Key<MLXArray>("test.ropeDeltas")

    @ModuleInfo var inner: Gemma3TextModel

    var kvHeads: [Int] { (inner as? KVCacheDimensionProvider)?.kvHeads ?? [] }

    init(_ inner: Gemma3TextModel) {
        self.inner = inner
    }

    private func batched(_ tokens: MLXArray) -> MLXArray {
        tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let logits = inner(batched(input.text.tokens), cache: cache.isEmpty ? nil : cache)
        // Carry a seeded anchor forward, or start one at zero — the shape
        // of every wired model's cold prefill.
        var produced = LMOutput.State()
        produced[Self.anchorKey] = state?[Self.anchorKey] ?? MLXArray([Int32(0)])
        let total = input.text.tokens.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits, state: produced))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: inner(batched(input.tokens), cache: cache), state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }
}

actor DraftModelLoadCounter {
    private var count = 0

    func increment() {
        count += 1
    }

    var value: Int {
        count
    }
}

/// Processor that masks every logit except one, forcing that token to be sampled.
struct ForceTokenProcessor: LogitProcessor {
    let token: Int32

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let indices = MLXArray(0 ..< Int32(logits.dim(-1)))
        return MLX.where(indices .== MLXArray(token), logits, MLXArray(-Float.infinity))
    }

    func didSample(token: MLXArray) {}
}

struct LiteralTokenizer: Tokenizer {
    /// The end-of-sequence token id, which ends each scripted output.
    static let endOfSequence = 99
    /// The unknown token id.
    static let unknown = 98
    /// The one token of every encoded prompt.
    static let prompt = 97

    let tokenByCharacter: [Character: Int]
    let characterByToken: [Int: Character]

    init(output: String) {
        let characters = Array(Set(output)).sorted { String($0) < String($1) }
        tokenByCharacter = Dictionary(
            uniqueKeysWithValues: characters.enumerated().map { ($0.element, $0.offset + 1) })
        characterByToken = Dictionary(
            uniqueKeysWithValues: tokenByCharacter.map { ($0.value, $0.key) })
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var eosTokenId: Int? { Self.endOfSequence }
    var unknownTokenId: Int? { Self.unknown }

    func tokens(for output: String) -> [Int] {
        output.compactMap { tokenByCharacter[$0] }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [Self.prompt] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(tokenIds.compactMap { characterByToken[$0] })
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.count == 1 ? token.first.flatMap { tokenByCharacter[$0] } : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        characterByToken[id].map(String.init)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [Self.prompt]
    }
}

final class TokenSequenceState: @unchecked Sendable {
    private let lock = NSLock()
    private let tokens: [Int32]
    private var index = 0

    init(tokens: [Int]) {
        self.tokens = tokens.map(Int32.init)
    }

    var current: Int32 {
        lock.withLock { tokens[min(index, tokens.count - 1)] }
    }

    func advance() {
        lock.withLock { index = min(index + 1, tokens.count - 1) }
    }
}

struct ForceTokenSequenceProcessor: LogitProcessor {
    let state: TokenSequenceState

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let indices = MLXArray(0 ..< Int32(logits.dim(-1)))
        return MLX.where(
            indices .== MLXArray(state.current), logits, MLXArray(-Float.infinity))
    }

    func didSample(token: MLXArray) {
        state.advance()
    }
}

/// Thread-safe counter for asserting how many times a `@Sendable` factory runs.
final class CallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.withLock { count += 1 }
    }

    var value: Int {
        lock.withLock { count }
    }
}
