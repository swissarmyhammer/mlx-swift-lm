// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Tool-argument schema for this test file. Named distinctly from every
/// other `@Generable` args struct in this test target -- `@Generable`'s
/// macro-generated conformances collide across files on the type name, not
/// just within one file (see `EagerFallbackPrepareOrderingTests.swift`'s
/// `EagerFallbackWeatherArgs` for the same note).
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ToolCallingImagePreservationArgs {
    @Guide(description: "Unused; present only so the tool has a valid parameters schema.")
    var note: String
}

/// Thrown by `ToolCallingImageProbeProcessor.prepare` immediately after
/// recording the `UserInput`'s image count, so the test never needs a real
/// model's weights or a real generation loop -- reaching this error at all
/// proves `runToolCalling`'s tool-aware retokenization routed through
/// `context.processor.prepare(input:)`.
private struct ToolCallingImageReachedProcessorError: Error {}

/// Thrown by `ToolCallingImageProbeGenerationModel.prepare` if real
/// generation is ever reached. Reaching THIS instead of
/// `ToolCallingImageReachedProcessorError` proves the bug this test guards
/// against: `runToolCalling` re-tokenized via a hand-rolled
/// `DefaultMessageGenerator` + raw `applyChatTemplate` call that never
/// touches `context.processor` at all, so no image ever had a chance to
/// reach it.
private struct ToolCallingImageReachedGenerationError: Error {}

/// Records the image count `UserInput` carried when
/// `ToolCallingImageProbeProcessor.prepare` was called -- `nil` if it was
/// never called at all. An `actor` because `UserInputProcessor.prepare` is
/// `async` and may run off the caller's isolation.
private actor ToolCallingImageProbeRecorder {
    private(set) var recordedImageCount: Int?

    func record(imageCount: Int) {
        recordedImageCount = imageCount
    }
}

/// A `UserInputProcessor` that records the image count it was handed, then
/// throws -- isolating "was this processor reached, and with how many
/// images" from any real tokenization or image-pixel-processing work.
private struct ToolCallingImageProbeProcessor: UserInputProcessor {
    let recorder: ToolCallingImageProbeRecorder

    func prepare(input: UserInput) async throws -> LMInput {
        await recorder.record(imageCount: input.images.count)
        throw ToolCallingImageReachedProcessorError()
    }
}

/// Minimal `LanguageModel` stand-in for `context.model`. Only reached if
/// `runToolCalling` bypasses `context.processor.prepare(input:)` entirely
/// (the bug), so throwing here -- instead of the processor's own error --
/// is itself diagnostic.
private final class ToolCallingImageProbeGenerationModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?) throws -> PrepareResult {
        throw ToolCallingImageReachedGenerationError()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

/// A byte-fallback tokenizer using the SentencePiece `<0xNN>` convention, so
/// that if `runToolCalling`'s xgrammar constraint setup IS reached (the bug
/// path, where retokenization happens via a raw `applyChatTemplate` call
/// with no processor involved), grammar compilation succeeds against a
/// known-compatible vocab instead of failing for an unrelated reason. See
/// `EagerFallbackPrepareOrderingTests.swift`'s
/// `EagerFallbackByteFallbackTokenizer` for the same rationale.
private struct ToolCallingImagePreservationByteFallbackTokenizer: MLXLMCommon.Tokenizer {
    private static let vocabSize = 256

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
    ) throws -> [Int] { [] }
}

@Suite("Tool-calling retokenization preserves image attachments end-to-end")
struct ToolCallingImagePreservationTests {

    @Test(
        """
        A .toolOutput entry's image attachment reaches context.processor.prepare(input:) \
        during runToolCalling's tool-aware retokenization, instead of being silently \
        dropped by a hand-rolled DefaultMessageGenerator + applyChatTemplate call that \
        never sees media attachments.
        """
    )
    func toolCallingPreservesToolOutputImageAttachment() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/tool-calling-image-preservation-\(UUID().uuidString)"
        let recorder = ToolCallingImageProbeRecorder()
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [.toolCalling, .vision],
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: { configuration, _ in
                // Constructed fresh here (no outer capture) so this
                // `@Sendable` closure never has to carry a non-Sendable
                // model/processor/tokenizer instance across the boundary.
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: ToolCallingImageProbeGenerationModel(),
                        processor: ToolCallingImageProbeProcessor(recorder: recorder),
                        tokenizer: ToolCallingImagePreservationByteFallbackTokenizer()))
            })
        let executor = try makeMLXExecutor(for: model)

        // A continuation round: TranscriptConverter.mlxMessages replays the
        // prior .toolCalls/.toolOutput round as assistant/tool-role
        // messages, ahead of the fresh prompt -- exactly the shape a real
        // tool-calling turn produces once a tool has returned an image.
        let toolCall = Transcript.ToolCall(
            id: "call-1", toolName: "take_photo",
            arguments: try GeneratedContent(json: #"{"note": "n/a"}"#))
        let toolCalls = Transcript.ToolCalls(id: "calls-1", [toolCall])
        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())), label: "photo")
        let toolOutput = Transcript.ToolOutput(
            id: "call-1", toolName: "take_photo",
            segments: [
                .text(Transcript.TextSegment(content: "Here is the photo")),
                .attachment(attachment),
            ])
        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "What's in the photo?"))],
            responseFormat: nil
        )
        let photoTool = Transcript.ToolDefinition(
            name: "take_photo", description: "Takes a photo",
            parameters: ToolCallingImagePreservationArgs.generationSchema
        )

        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .toolCalls(toolCalls), .toolOutput(toolOutput), .prompt(prompt),
            ]),
            enabledTools: [photoTool]
        )

        let channel = LanguageModelExecutorGenerationChannel()
        // respond() sends a metadata event on the channel before reaching
        // either throw point below; with no concurrent reader, that send
        // would suspend forever.
        let drainer = Task<Void, Never> {
            do {
                for try await _ in channel {}
            } catch {
                // Draining only; the probe error surfaces via respond()'s throw below.
            }
        }
        defer { drainer.cancel() }

        do {
            try await executor.respond(to: request, model: model, streamingInto: channel)
            Issue.record("Expected an error once the probe path was reached, but respond returned")
        } catch let error as LanguageModelError {
            // Expected: the tool-aware retokenization routed through
            // context.processor.prepare(input:), which recorded the image
            // count below before throwing ToolCallingImageReachedProcessorError.
            // Since the transcript carries an image,
            // preparedInputMappingImageFailures wraps that throw into
            // unsupportedTranscriptContent -- the same remapping
            // UnsupportedTranscriptContentMappingTests exercises for the
            // non-tool-calling path.
            guard case .unsupportedTranscriptContent = error else {
                Issue.record(
                    "Expected unsupportedTranscriptContent wrapping ToolCallingImageReachedProcessorError, got \(error)"
                )
                return
            }
        } catch is ToolCallingImageReachedGenerationError {
            Issue.record(
                """
                runToolCalling never called context.processor.prepare(input:) at all -- it \
                reached real generation via a hand-rolled DefaultMessageGenerator + \
                applyChatTemplate call, which cannot see media attachments. This is exactly \
                the silent-drop bug this test guards against.
                """
            )
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        let recordedImageCount = await recorder.recordedImageCount
        #expect(
            recordedImageCount == 1,
            "the tool-output image attachment must reach the model's UserInputProcessor during tool-calling retokenization"
        )
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
