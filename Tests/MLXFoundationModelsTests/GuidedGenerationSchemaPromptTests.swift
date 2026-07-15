// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Developer-schema stand-in for the guided-generation prompt-seam tests
/// below. Named `GuidedPromptSeamSchemaArgs` (not the more obvious
/// `SchemaArgs`) because `@Generable`'s macro-generated conformances
/// collide across files on the type name, not just within one file -- see
/// `ContextSizeValidationTests.ContextSizeSchemaArgs` for the same idiom.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct GuidedPromptSeamSchemaArgs {
    @Guide(description: "A single word answer.")
    var answer: String
}

/// Thrown the instant real generation would start, so the seam tests below
/// never need a real model's weights or vocabulary-sized logits -- they only
/// need to prove the prompt reached `UserInputProcessor.prepare` (captured by
/// `GuidedPromptSeamCaptureProcessor`) before `contextSizeExceeded` cuts
/// dispatch off.
private struct GuidedPromptSeamGenerationProbeError: Error {}

/// Minimal `LanguageModel` stand-in for `context.model`. Throws from
/// `prepare` before any `MLXArray` computation runs, proving (if ever
/// reached) that the context-size check didn't fire when it should have --
/// the same idiom as `ContextSizeGenerationProbeModel` in
/// `ContextSizeValidationTests.swift`.
private final class GuidedPromptSeamGenerationProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?) throws -> PrepareResult {
        throw GuidedPromptSeamGenerationProbeError()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

/// Records the `[Chat.Message]` array a real `UserInputProcessor.prepare`
/// call receives during `Executor.respond`'s guided-generation dispatch --
/// i.e. whatever `prepareRespondSetup`/`guidedGenerationMessages` actually
/// produced, not a hand-built array bypassing that production code path.
/// Returns a fixed, deliberately over-long token count so the caller's tiny
/// `max_position_embeddings` throws `contextSizeExceeded` immediately after
/// capturing, before any real generation is attempted.
///
/// `@unchecked Sendable`: mutated only from the single serialized
/// `container.perform` call each test below drives -- the same idiom as
/// `CountingTokenizer` in `TokenizerBiasCacheTests.swift`.
private final class GuidedPromptSeamCaptureProcessor: UserInputProcessor, @unchecked Sendable {
    private(set) var capturedMessages: [Chat.Message]?
    private let tokenCount: Int

    init(tokenCount: Int) {
        self.tokenCount = tokenCount
    }

    func prepare(input: UserInput) async throws -> LMInput {
        if case .chat(let messages) = input.prompt {
            capturedMessages = messages
        }
        return LMInput(tokens: MLXArray(Array(repeating: Int32(1), count: tokenCount)))
    }
}

/// Counts occurrences of `schemaJSON` across `messages`' `content` -- the
/// tokenizable text a `UserInputProcessor` renders into the prompt, so this
/// is a faithful proxy for "how many times the schema appears in the
/// tokenized prompt" without needing a real tokenizer. Shared by both test
/// suites below: `GuidedGenerationSchemaPromptTests` (isolated helper calls)
/// and `GuidedGenerationPromptSeamTests` (the real `Executor.respond` seam).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func countSchemaOccurrences(in messages: [Chat.Message], schemaJSON: String) -> Int {
    messages.reduce(0) { count, message in
        count + message.content.components(separatedBy: schemaJSON).count - 1
    }
}

/// Regression coverage for `ContextOptions.includeSchemaInPrompt` at the
/// guided-generation prompt-assembly seam.
///
/// This file has two suites:
///
/// - `GuidedGenerationSchemaPromptTests` (below): direct unit tests of the
///   pure helpers `Executor.guidedGenerationMessages` and
///   `Executor.shouldInjectSchemaIntoPrompt`, called directly with
///   hand-built `[Chat.Message]` arrays. These pin down the helpers' own
///   semantics in isolation but do NOT exercise `prepareRespondSetup`,
///   `dispatchGeneration`, `respond()`, `TranscriptConverter.mlxMessages`, or
///   any tokenizer/`UserInputProcessor` -- so passing here does not by
///   itself prove the helpers are wired into production.
/// - `GuidedGenerationPromptSeamTests` (further below): drives the real seam
///   end-to-end through `Executor.respond`, with a fake `UserInputProcessor`
///   that captures the actual messages `prepareRespondSetup` hands it and a
///   real `Transcript` rendered through `TranscriptConverter.mlxMessages` --
///   proving the wiring itself, not just the isolated helper.
///
/// Investigation established the adapter has no schema-in-prompt rendering
/// of its own to begin with (`schemaJSON` only ever feeds xgrammar's
/// constrained-decoding constraint, never prompt text), so "preserve current
/// behavior" for every value of the flag means "the seam never adds
/// anything" -- both suites assert exactly that.
@Suite("Guided-generation prompt-assembly helpers honor ContextOptions.includeSchemaInPrompt (isolated)")
struct GuidedGenerationSchemaPromptTests {

    private static let schemaJSON = #"{"type":"object","properties":{"answer":{"type":"string"}}}"#

    /// Counts occurrences of `schemaJSON` across the assembled messages'
    /// `content`. Delegates to the file-level `countSchemaOccurrences(in:
    /// schemaJSON:)` shared with `GuidedGenerationPromptSeamTests` below.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func schemaOccurrences(in messages: [Chat.Message]) -> Int {
        countSchemaOccurrences(in: messages, schemaJSON: schemaJSON)
    }

    @Test("includeSchemaInPrompt == true: the seam adds nothing extra")
    func trueAddsNothingExtra() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let messages: [Chat.Message] = [.user(content: "Describe today's weather.")]

        let result = MLXLanguageModel.Executor.guidedGenerationMessages(
            from: messages, schemaJSON: Self.schemaJSON, includeSchemaInPrompt: true)

        #expect(result.count == messages.count)
        #expect(Self.schemaOccurrences(in: result) == 0)
    }

    @Test("includeSchemaInPrompt == false: current behavior is unchanged -- still nothing added")
    func falseAddsNothingExtra() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let messages: [Chat.Message] = [.user(content: "Describe today's weather.")]

        let result = MLXLanguageModel.Executor.guidedGenerationMessages(
            from: messages, schemaJSON: Self.schemaJSON, includeSchemaInPrompt: false)

        #expect(result.count == messages.count)
        #expect(Self.schemaOccurrences(in: result) == 0)
    }

    @Test("includeSchemaInPrompt == nil: same as false -- current behavior is unchanged")
    func nilAddsNothingExtra() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let messages: [Chat.Message] = [.user(content: "Describe today's weather.")]

        let result = MLXLanguageModel.Executor.guidedGenerationMessages(
            from: messages, schemaJSON: Self.schemaJSON, includeSchemaInPrompt: nil)

        #expect(result.count == messages.count)
        #expect(Self.schemaOccurrences(in: result) == 0)
    }

    @Test(
        """
        A transcript that already embeds the schema's own rendering keeps exactly one copy \
        regardless of includeSchemaInPrompt's value -- the seam never doubles it, matching the \
        `true` contract even while the gate's `false`/`nil` branch is exercised
        """
    )
    func neverDuplicatesTranscriptsOwnRenderingRegardlessOfFlag() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Simulates the app having already put the schema in the prompt
        // text itself (whatever assembled the transcript -- the framework's
        // own default prompt construction, or the developer's own manual
        // `Prompt` segments).
        let messages: [Chat.Message] = [.user(content: "Schema: \(Self.schemaJSON)")]

        for includeSchemaInPrompt in [true, false, nil] {
            let result = MLXLanguageModel.Executor.guidedGenerationMessages(
                from: messages, schemaJSON: Self.schemaJSON,
                includeSchemaInPrompt: includeSchemaInPrompt)
            #expect(
                Self.schemaOccurrences(in: result) == 1,
                "includeSchemaInPrompt == \(String(describing: includeSchemaInPrompt)) duplicated the schema"
            )
        }
    }

    @Test("shouldInjectSchemaIntoPrompt gates true vs. false/nil")
    func shouldInjectGatesOnExactlyTrue() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: true)
                == false)
        #expect(
            MLXLanguageModel.Executor.shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: false)
                == true)
        #expect(
            MLXLanguageModel.Executor.shouldInjectSchemaIntoPrompt(includeSchemaInPrompt: nil)
                == true)
    }
}

/// Drives `Executor.respond` end-to-end -- through the real
/// `prepareRespondSetup` -> `guidedGenerationMessages` ->
/// `UserInputProcessor.prepare` seam, with a real `Transcript` rendered by
/// the real `TranscriptConverter.mlxMessages` -- to prove the wiring itself
/// honors `ContextOptions.includeSchemaInPrompt`, not just the isolated
/// helper (see `GuidedGenerationSchemaPromptTests` above).
///
/// Each test builds a schema-guided request (so `dispatchGeneration` takes
/// the guided-generation branch), optionally embeds the schema's own JSON
/// rendering in the transcript's prompt text (simulating the app having
/// already put it there -- the real contract behind
/// `ContextOptions(includeSchemaInPrompt: true)`, see
/// `shouldInjectSchemaIntoPrompt`'s doc comment), and drives a tiny
/// `max_position_embeddings` so `dispatchGeneration`'s guided-generation
/// branch throws `contextSizeExceeded` immediately after
/// `GuidedPromptSeamCaptureProcessor.prepare` captures the real, rendered
/// messages -- never reaching real generation.
@Suite("Executor.respond guided-generation prompt seam honors ContextOptions.includeSchemaInPrompt")
struct GuidedGenerationPromptSeamTests {

    /// Builds a schema-guided request whose transcript prompt text optionally
    /// embeds the schema's own JSON rendering.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeRequest(
        embedSchemaInPromptText: Bool,
        schemaJSON: String,
        includeSchemaInPrompt: Bool?
    ) -> LanguageModelExecutorGenerationRequest {
        let promptText =
            embedSchemaInPromptText
            ? "Respond matching this schema: \(schemaJSON)"
            : "Describe today's weather."
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: promptText))],
            responseFormat: nil)
        return makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(prompt)]),
            schema: GuidedPromptSeamSchemaArgs.generationSchema,
            contextOptions: ContextOptions(includeSchemaInPrompt: includeSchemaInPrompt))
    }

    /// Drives `Executor.respond` for the given flag/transcript combination
    /// and returns how many times the schema's own JSON rendering appears in
    /// the `[Chat.Message]` array actually captured at
    /// `UserInputProcessor.prepare` -- the real tokenizable content
    /// `prepareRespondSetup`'s guided-generation seam produced, not a
    /// reimplemented/isolated check.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func schemaOccurrencesAtRealSeam(
        embedSchemaInPromptText: Bool,
        includeSchemaInPrompt: Bool?
    ) async throws -> Int {
        let schemaJSON = try SchemaConverter.encodeToJSON(
            GuidedPromptSeamSchemaArgs.generationSchema)
        let request = makeRequest(
            embedSchemaInPromptText: embedSchemaInPromptText, schemaJSON: schemaJSON,
            includeSchemaInPrompt: includeSchemaInPrompt)

        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "mlx-guided-prompt-seam-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        // `max_position_embeddings: 1` guarantees the (deliberately
        // over-long) captured prompt trips `contextSizeExceeded` in
        // `dispatchGeneration`'s guided-generation branch immediately after
        // `UserInputProcessor.prepare` runs, before real generation.
        try #"{"model_type": "test", "max_position_embeddings": 1}"#.write(
            to: modelDirectory.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let processor = GuidedPromptSeamCaptureProcessor(tokenCount: 50)
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(directory: modelDirectory),
            weightsLocation: { _ in modelDirectory },
            load: { configuration, _ in
                // Constructed fresh here (no outer capture beyond the
                // `Sendable`-conforming `processor`) -- the same idiom as
                // `ContextSizeValidationTests.makeContextSizeTestModel`.
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: GuidedPromptSeamGenerationProbeModel(),
                        processor: processor,
                        tokenizer: ByteTokenizer()))
            })

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
            Issue.record(
                "Expected LanguageModelError.contextSizeExceeded once the real seam's prompt reached validation"
            )
        } catch is GuidedPromptSeamGenerationProbeError {
            Issue.record(
                """
                Generation was reached even though the captured prompt exceeds the model's \
                context length -- the context-size check either didn't run or didn't fire.
                """
            )
        } catch let error as LanguageModelError {
            guard case .contextSizeExceeded = error else {
                Issue.record("Expected .contextSizeExceeded, got \(error)")
                return -1
            }
            // Expected: the real seam's prompt reached validation and was
            // rejected before generation, exactly as intended.
        }

        guard let captured = processor.capturedMessages else {
            Issue.record(
                "UserInputProcessor.prepare was never called -- the real seam wasn't reached")
            return -1
        }
        return countSchemaOccurrences(in: captured, schemaJSON: schemaJSON)
    }

    @Test(
        "includeSchemaInPrompt == true, schema already in the transcript: the real prompt keeps exactly one copy"
    )
    func trueWithSchemaAlreadyInTranscriptKeepsExactlyOneCopy() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let occurrences = try await schemaOccurrencesAtRealSeam(
            embedSchemaInPromptText: true, includeSchemaInPrompt: true)
        #expect(occurrences == 1)
    }

    @Test(
        """
        includeSchemaInPrompt == false, schema already in the transcript: current behavior is \
        unchanged -- the real prompt still keeps exactly one copy, never duplicated
        """
    )
    func falseWithSchemaAlreadyInTranscriptKeepsExactlyOneCopy() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let occurrences = try await schemaOccurrencesAtRealSeam(
            embedSchemaInPromptText: true, includeSchemaInPrompt: false)
        #expect(occurrences == 1)
    }

    @Test(
        """
        includeSchemaInPrompt == nil, schema already in the transcript: same as false -- the \
        real prompt still keeps exactly one copy
        """
    )
    func nilWithSchemaAlreadyInTranscriptKeepsExactlyOneCopy() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let occurrences = try await schemaOccurrencesAtRealSeam(
            embedSchemaInPromptText: true, includeSchemaInPrompt: nil)
        #expect(occurrences == 1)
    }

    @Test(
        "includeSchemaInPrompt == true, no schema in the transcript: the real prompt has zero occurrences -- the seam adds nothing"
    )
    func trueWithNoSchemaInTranscriptAddsNothing() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let occurrences = try await schemaOccurrencesAtRealSeam(
            embedSchemaInPromptText: false, includeSchemaInPrompt: true)
        #expect(occurrences == 0)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
