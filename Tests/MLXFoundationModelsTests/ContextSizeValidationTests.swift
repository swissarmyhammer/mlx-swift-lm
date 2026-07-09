// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Tool-argument schema for the tool-calling regression test. Named
/// `ContextSizeWeatherArgs` (not the more obvious `WeatherArgs`) because
/// `@Generable`'s macro-generated conformances collide across files on the
/// type name, not just within one file -- see
/// `EagerFallbackPrepareOrderingTests.swift`'s `EagerFallbackWeatherArgs` for
/// the same idiom.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ContextSizeWeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

/// Developer-schema stand-in for the guided-generation regression test.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ContextSizeSchemaArgs {
    @Guide(description: "A single word answer.")
    var answer: String
}

/// Thrown the instant real generation would start, so these tests never need
/// a real model's weights or vocabulary-sized logits. If `Executor.respond`
/// ever reached this, the context-size check either didn't run or didn't
/// fire when it should have.
private struct ContextSizeGenerationProbeError: Error {}

/// Minimal `LanguageModel` stand-in for `context.model`. Throws from
/// `prepare` before any `MLXArray` computation runs, proving the dispatch
/// path reached real generation without needing real weights.
private final class ContextSizeGenerationProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        throw ContextSizeGenerationProbeError()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

/// A `UserInputProcessor` that always returns a fixed-length `LMInput`,
/// regardless of the transcript's actual text -- lets each test dictate the
/// prompt's token count directly instead of depending on a real tokenizer's
/// output length.
private struct ContextSizeFixedLengthProcessor: UserInputProcessor {
    let tokenCount: Int

    func prepare(input: UserInput) async throws -> LMInput {
        LMInput(tokens: MLXArray(Array(repeating: Int32(1), count: tokenCount)))
    }
}

/// A `UserInputProcessor` that reports a different fixed token count
/// depending on whether the render carries `additionalContext`. Lets a test
/// give the reasoning-primed prompt (rendered via `Self.preparedInput` with a
/// non-nil `additionalContext` for a `.templateFlag` strategy) an
/// independently-controllable length from the baseline prompt (rendered via
/// the eager `context.processor.prepare(input: UserInput(chat: messages))`
/// call, which carries no `additionalContext`) -- reproduces the two
/// distinct prompts `prepareRespondSetup` can render for one request.
private struct ContextSizeReasoningAwareProcessor: UserInputProcessor {
    let baselineTokenCount: Int
    let reasoningTokenCount: Int

    func prepare(input: UserInput) async throws -> LMInput {
        let count = input.additionalContext == nil ? baselineTokenCount : reasoningTokenCount
        return LMInput(tokens: MLXArray(Array(repeating: Int32(1), count: count)))
    }
}

/// A `Tokenizer` whose `applyChatTemplate` always returns a fixed-length
/// token array. The tool-calling path re-tokenizes independently via
/// `context.tokenizer.applyChatTemplate` (bypassing `UserInputProcessor`
/// entirely -- see `runToolCalling`), so this is what the tool-calling
/// regression test uses to control that path's prompt length directly.
private struct ContextSizeFixedLengthTokenizer: MLXLMCommon.Tokenizer {
    let tokenCount: Int

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        Array(repeating: 1, count: tokenCount)
    }
}

/// A SentencePiece-style byte-fallback tokenizer (`<0xNN>` per-byte vocab --
/// the same shape as `EagerFallbackPrepareOrderingTests.EagerFallbackByteFallbackTokenizer`
/// / `ToolCallingSchemaTests.makeByteTokenizer()`, both proven to compile a
/// real `GrammarConstraint`). Unlike `ContextSizeFixedLengthTokenizer` (which
/// can't round-trip real text), this one actually encodes/decodes real UTF-8
/// bytes, so it doubles as both `applyChatTemplate`'s tool-aware prompt AND
/// the tokenizer `ReasoningTokenCollector`'s `NaiveStreamingDetokenizer`
/// decodes Phase 1's generated tokens through -- required for the
/// think-then-call regression test below, the only test in this file whose
/// prompt actually reaches real xgrammar/`GrammarConstraint` compilation
/// (every other test's context-size check fires before `prepareConstraintSetup`
/// ever runs).
private struct ContextSizeThinkThenCallByteTokenizer: MLXLMCommon.Tokenizer {
    private static let vocabSize = 256

    /// Rendered as-is (byte-encoded) for `applyChatTemplate` -- the test
    /// controls this directly instead of deriving it from `messages`/`tools`.
    let promptText: String

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        return UInt8(token.dropFirst(3).dropLast(), radix: 16).map(Int.init)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0 && id < Self.vocabSize else { return nil }
        return String(format: "<0x%02X>", id)
    }

    var bosToken: String? { nil }
    var eosToken: String? { String(format: "<0x%02X>", Self.vocabSize - 1) }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(text: promptText, addSpecialTokens: false)
    }
}

/// Thrown from `ContextSizeReasoningCloseModel.prepare` on any call after the
/// first, proving Phase 2 (`executeToolCallingPhase2`/`GuidedGenerationLoop.run`,
/// which starts its own `TokenIterator` and so calls `prepare` again) was
/// reached. A dedicated error (rather than reusing
/// `ContextSizeGenerationProbeModel`/`ContextSizeGenerationProbeError`) because
/// this model's *first* `prepare` call must succeed -- it needs to actually
/// drive Phase 1's reasoning-token generation loop -- so "generation was
/// reached" here specifically means "generation reached a SECOND round",
/// i.e. Phase 2 starting.
private struct ContextSizeReasoningPhase2ReachedError: Error {}

/// A `LanguageModel` whose `prepare` succeeds exactly once (see
/// `ContextSizeReasoningPhase2ReachedError`) and whose `callAsFunction`
/// deterministically emits a planned token sequence via one-hot logits at
/// each generated position -- the same idiom as `MockMainModel` in
/// `MTPSpeculativeTokenIteratorTests.swift`, minus the MTP-specific state
/// hooks this test doesn't need. Drives Phase 1's real (if tiny) token-by-token
/// generation loop so it closes normally -- having generated a controlled,
/// non-zero number of reasoning tokens -- rather than being cut off at
/// `maxTokens` (a cut-off Phase 1 returns before `phase2Input` is ever built,
/// which would prove nothing about this test's fix).
private final class ContextSizeReasoningCloseModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [1] }

    /// Token values returned in increasing generated-position order (index 0
    /// covers the prompt's first position). Only the LAST position of each
    /// forward call is ever sampled (`TokenIterator.convertToToken` reads
    /// `logits[..., -1, ...]`), so entries covering the prompt's earlier
    /// positions are never consulted and can be filler.
    private let plannedTokens: [Int32]
    private var position = 0
    private var prepareCallCount = 0

    init(plannedTokens: [Int32]) {
        self.plannedTokens = plannedTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepareCallCount += 1
        guard prepareCallCount == 1 else {
            throw ContextSizeReasoningPhase2ReachedError()
        }
        return .tokens(input.text)
    }

    /// Returns deterministic one-hot logits at each position so the default
    /// (non-zero-temperature) categorical sampler picks the planned token: the
    /// gap between the planned token's logit and every other's is large enough
    /// that softmax collapses every other token's probability to (numerically)
    /// zero, making the sample deterministic in practice without needing
    /// `.greedy`/argmax sampling to be forced explicitly.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        let vocab = 256
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0 ..< positions {
            let tokIdx = position + i
            let tok = tokIdx < plannedTokens.count ? Int(plannedTokens[tokIdx]) : 0
            data[i * vocab + tok] = 100
        }
        position += positions
        return MLXArray(data, [1, positions, vocab])
    }
}

/// Builds an `MLXLanguageModel` backed by a real on-disk `config.json`
/// (so `Executor.respond`'s `context.configuration.modelDirectory` resolves
/// and the context-length decode in `prepareRespondSetup` has something to
/// read), the given processor, and a model that throws the moment real
/// generation would start.
///
/// - Parameters:
///   - maxPositionEmbeddings: The `max_position_embeddings` value written
///     into the fake model's `config.json`.
///   - processor: The `UserInputProcessor` `context.processor` resolves to,
///     standing in for the transcript's real tokenized length(s).
///   - capabilities: Capabilities declared on the model (e.g. `.reasoning`
///     for the reasoning-path regression test).
///   - reasoningConfig: Threaded into the model's `ModelConfiguration` so
///     `resolved.reasoningConfig` is non-nil for the reasoning-path test.
///   - tokenizer: The `Tokenizer` `context.tokenizer` resolves to. Only the
///     tool-calling test needs to control this (`applyChatTemplate`'s
///     output length); every other path renders through `processor` instead.
///   - model: A factory for the `context.model` stand-in, invoked fresh
///     inside the (`@Sendable`) `load` closure below -- a factory rather
///     than a plain instance because `LanguageModel` isn't `Sendable`, so an
///     already-constructed instance can't be captured across the closure
///     boundary. Defaults to `ContextSizeGenerationProbeModel`, which throws
///     the instant real generation starts; only the think-then-call
///     regression test needs to override this with a model that actually
///     generates (Phase 1 must run for real to produce reasoning tokens).
/// - Returns: The constructed model, and a `cleanup` closure the caller must
///   run (via `defer`) to remove the temporary model directory.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeContextSizeTestModel(
    maxPositionEmbeddings: Int,
    processor: any UserInputProcessor,
    capabilities: [LanguageModelCapabilities.Capability] = [],
    reasoningConfig: ReasoningConfig? = nil,
    tokenizer: any MLXLMCommon.Tokenizer = ByteTokenizer(),
    model modelFactory: @escaping @Sendable () -> any MLXLMCommon.LanguageModel = {
        ContextSizeGenerationProbeModel()
    }
) throws -> (model: MLXLanguageModel, cleanup: () -> Void) {
    let modelDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("mlx-context-size-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: modelDirectory, withIntermediateDirectories: true)
    let configJSON = """
        {"model_type": "test", "max_position_embeddings": \(maxPositionEmbeddings)}
        """
    try configJSON.write(
        to: modelDirectory.appendingPathComponent("config.json"),
        atomically: true, encoding: .utf8)

    let model = MLXLanguageModel(
        configuration: ModelConfiguration(
            directory: modelDirectory, reasoningConfig: reasoningConfig),
        capabilities: capabilities,
        weightsLocation: { _ in modelDirectory },
        load: { configuration, _ in
            // Constructed fresh here (no outer capture beyond the
            // `Sendable`-conforming `processor` parameter) -- see
            // `EagerFallbackPrepareOrderingTests` for the same idiom.
            ModelContainer(
                context: ModelContext(
                    configuration: configuration,
                    model: modelFactory(),
                    processor: processor,
                    tokenizer: tokenizer))
        })
    return (model, { try? FileManager.default.removeItem(at: modelDirectory) })
}

/// Builds a single-prompt request with no tools/schema, so `dispatchGeneration`
/// takes the plain-text-generation branch.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeContextSizeTestRequest(
    schema: GenerationSchema? = nil,
    enabledTools: [Transcript.ToolDefinition] = []
) -> LanguageModelExecutorGenerationRequest {
    let prompt = Transcript.Prompt(
        segments: [.text(Transcript.TextSegment(content: "Does this exceed the context window?"))],
        responseFormat: nil)
    return makeExecutorRequest(
        transcript: Transcript(entries: [.prompt(prompt)]),
        enabledTools: enabledTools, schema: schema)
}

/// Shared assertion for the "exceeds" tests below: expects
/// `LanguageModelError.contextSizeExceeded` with the given `contextSize`/
/// `tokenCount`, and fails loudly if generation was reached instead (the
/// probe model/error never fires when the check works).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func expectContextSizeExceeded(
    model: MLXLanguageModel, request: LanguageModelExecutorGenerationRequest,
    expectedContextSize: Int, expectedTokenCount: Int
) async throws {
    let executor = try makeMLXExecutor(for: model)
    let channel = LanguageModelExecutorGenerationChannel()
    let drainer = Task<Void, Never> {
        do {
            for try await _ in channel {}
        } catch {
            // Draining only; the typed error surfaces via respond()'s throw below.
        }
    }
    defer { drainer.cancel() }

    do {
        try await executor.respond(to: request, model: model, streamingInto: channel)
        Issue.record("Expected LanguageModelError.contextSizeExceeded")
    } catch is ContextSizeGenerationProbeError {
        Issue.record(
            """
            Generation was reached even though the prompt exceeds the model's context \
            length -- the context-size check either didn't run or didn't fire on this path.
            """
        )
    } catch let error as LanguageModelError {
        guard case .contextSizeExceeded(let details) = error else {
            Issue.record("Expected .contextSizeExceeded, got \(error)")
            return
        }
        #expect(details.contextSize == expectedContextSize)
        #expect(details.tokenCount == expectedTokenCount)
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Suite("Executor.respond context-size validation before generation")
struct ContextSizeValidationTests {

    @Test("A transcript within the model's context length reaches generation normally")
    func withinContextLengthReachesGeneration() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: 100,
            processor: ContextSizeFixedLengthProcessor(tokenCount: 5))
        defer { cleanup() }
        let executor = try makeMLXExecutor(for: model)

        let channel = LanguageModelExecutorGenerationChannel()
        let drainer = Task<Void, Never> {
            do {
                for try await _ in channel {}
            } catch {
                // Draining only; the probe error surfaces via respond()'s throw below.
            }
        }
        defer { drainer.cancel() }

        do {
            try await executor.respond(
                to: makeContextSizeTestRequest(), model: model, streamingInto: channel)
            Issue.record("Expected ContextSizeGenerationProbeError once dispatch reached generation")
        } catch is ContextSizeGenerationProbeError {
            // Expected: within-limits prompt reaches real generation
            // unimpeded -- no regression to existing dispatch behavior.
        } catch let error as LanguageModelError {
            Issue.record("Expected generation to be reached, but got a typed error instead: \(error)")
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test(
        "A transcript exceeding the model's context length throws contextSizeExceeded before generation (plain text-generation path)"
    )
    func exceedingContextLengthThrowsBeforeGeneration() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: 4,
            processor: ContextSizeFixedLengthProcessor(tokenCount: 10))
        defer { cleanup() }

        try await expectContextSizeExceeded(
            model: model, request: makeContextSizeTestRequest(),
            expectedContextSize: 4, expectedTokenCount: 10)
    }

    @Test(
        """
        A transcript within the baseline render but exceeding the context length via the \
        reasoning-primed prompt still throws before generation
        """
    )
    func exceedingContextLengthViaReasoningPromptThrowsBeforeGeneration() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // `.templateFlag` always yields a non-nil `additionalContext` (see
        // `ReasoningPromptStrategy.additionalContext`), so the reasoning
        // render always differs from the baseline render for this strategy.
        let reasoningConfig = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: 4,
            processor: ContextSizeReasoningAwareProcessor(
                baselineTokenCount: 2, reasoningTokenCount: 10),
            capabilities: [.reasoning],
            reasoningConfig: reasoningConfig)
        defer { cleanup() }

        // No tools/schema, so `dispatchGeneration` takes the plain
        // text-generation branch, which must validate whichever prompt
        // `runTextGeneration` actually feeds to generation -- the
        // reasoning-primed prompt here, not the smaller baseline render.
        try await expectContextSizeExceeded(
            model: model, request: makeContextSizeTestRequest(),
            expectedContextSize: 4, expectedTokenCount: 10)
    }

    @Test(
        "A transcript exceeding the model's context length throws contextSizeExceeded before generation (guided-generation path)"
    )
    func exceedingContextLengthThrowsBeforeGenerationGuidedGeneration() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: 4,
            processor: ContextSizeFixedLengthProcessor(tokenCount: 10))
        defer { cleanup() }

        try await expectContextSizeExceeded(
            model: model,
            request: makeContextSizeTestRequest(schema: ContextSizeSchemaArgs.generationSchema),
            expectedContextSize: 4, expectedTokenCount: 10)
    }

    @Test(
        "A transcript exceeding the model's context length throws contextSizeExceeded before generation (tool-calling path)"
    )
    func exceedingContextLengthThrowsBeforeGenerationToolCalling() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // The tool-calling path re-tokenizes independently via
        // `context.tokenizer.applyChatTemplate`, not `context.processor` --
        // the fixed-length processor is irrelevant here, so the tokenizer
        // is what controls this path's (over-limit) prompt length.
        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: 4,
            processor: ContextSizeFixedLengthProcessor(tokenCount: 0),
            capabilities: [.toolCalling],
            tokenizer: ContextSizeFixedLengthTokenizer(tokenCount: 10))
        defer { cleanup() }

        let weatherTool = Transcript.ToolDefinition(
            name: "get_weather", description: "Get current weather",
            parameters: ContextSizeWeatherArgs.generationSchema)

        try await expectContextSizeExceeded(
            model: model,
            request: makeContextSizeTestRequest(enabledTools: [weatherTool]),
            expectedContextSize: 4, expectedTokenCount: 10)
    }

    @Test(
        """
        Phase 1's reasoning tokens pushing the tool-aware prompt over the context length \
        throws contextSizeExceeded before Phase 2 starts (think-then-call path)
        """
    )
    func exceedingContextLengthAfterThinkThenCallPhase1ThrowsBeforePhase2() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // `.templateFlag` is required for `makeThinkThenCallConfig` to resolve
        // think-then-call at all (see `runToolCalling`), alongside declaring
        // `.reasoning`. The tool-aware prompt text is exactly the reasoning
        // config's start delimiter, unclosed -- `reasoningPrimedInside`
        // (which decodes the prompt's tail) then reports the prompt as
        // already inside a reasoning span, mirroring a DeepSeek-R1-style
        // template that prefills `<think>` into the assistant turn. That
        // means Phase 1 only has to generate the CLOSING delimiter to close
        // cleanly, not the opening one too -- keeping the real generation
        // loop this test drives trivially short.
        let reasoningConfig = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
        let promptText = "<think>"
        let tokenizer = ContextSizeThinkThenCallByteTokenizer(promptText: promptText)
        let promptTokenCount = Array(promptText.utf8).count

        // Phase 1 must generate a real, small number of tokens before
        // closing `</think>` -- one byte-token per character of the closing
        // delimiter, via a model that deterministically emits them in order.
        // Entries before index `promptTokenCount - 1` are never read (only
        // the LAST position of each forward call is sampled), so they're
        // filler.
        let closingDelimiterTokens = Array("</think>".utf8).map { Int32($0) }
        let plannedTokens =
            Array(repeating: Int32(0), count: promptTokenCount - 1) + closingDelimiterTokens

        // Sits strictly between the prompt alone (7 tokens, within limits --
        // the INITIAL `toolAwareInput` check must pass) and prompt + Phase
        // 1's reasoning tokens (7 + 8 = 15, over limit) -- proving the fix's
        // re-validation of `phase2Input`, not the earlier, already-covered
        // `toolAwareInput` check.
        let contextLength = 10

        let (model, cleanup) = try makeContextSizeTestModel(
            maxPositionEmbeddings: contextLength,
            processor: ContextSizeFixedLengthProcessor(tokenCount: 0),
            capabilities: [.toolCalling, .reasoning],
            reasoningConfig: reasoningConfig,
            tokenizer: tokenizer,
            model: { ContextSizeReasoningCloseModel(plannedTokens: plannedTokens) })
        defer { cleanup() }

        let weatherTool = Transcript.ToolDefinition(
            name: "get_weather", description: "Get current weather",
            parameters: ContextSizeWeatherArgs.generationSchema)

        let executor = try makeMLXExecutor(for: model)
        let channel = LanguageModelExecutorGenerationChannel()
        let drainer = Task<Void, Never> {
            do {
                for try await _ in channel {}
            } catch {
                // Draining only; the typed error surfaces via respond()'s throw below.
            }
        }
        defer { drainer.cancel() }

        do {
            try await executor.respond(
                to: makeContextSizeTestRequest(enabledTools: [weatherTool]),
                model: model, streamingInto: channel)
            Issue.record("Expected LanguageModelError.contextSizeExceeded")
        } catch is ContextSizeReasoningPhase2ReachedError {
            Issue.record(
                """
                Phase 2 was reached even though Phase 1's reasoning tokens push the prompt \
                over the model's context length -- the Phase-2 re-validation either didn't \
                run or didn't fire.
                """
            )
        } catch let error as LanguageModelError {
            guard case .contextSizeExceeded(let details) = error else {
                Issue.record("Expected .contextSizeExceeded, got \(error)")
                return
            }
            #expect(details.contextSize == contextLength)
            #expect(details.tokenCount == promptTokenCount + closingDelimiterTokens.count)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
