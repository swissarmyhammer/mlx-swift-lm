// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Stands in for a concrete VLM's own image-specific processing failure
/// (mismatched image count, unsupported resolution, decode/resize failure --
/// `MLXVLM.VLMError` and its per-architecture siblings). MLXFoundationModels
/// can't catch those by type since it deliberately does not depend on MLXVLM
/// (see Package.swift's "runtime trampoline discovery" note), so
/// `preparedInputMappingImageFailures` maps by content shape instead --
/// this generic error is exactly what a real VLM processor failure would
/// look like from MLXFoundationModels' side of that boundary.
private struct ImageProcessingProbeError: Error {}

/// A `UserInputProcessor` that throws `ImageProcessingProbeError` whenever
/// the prepared `UserInput` carries image content, and otherwise succeeds --
/// isolating the image-failure path from any unrelated (tokenizer/config)
/// failure the same processor might also raise.
private struct ImageFailingProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        if !input.images.isEmpty {
            throw ImageProcessingProbeError()
        }
        return LMInput(tokens: MLXArray([Int32(0)]))
    }
}

/// A `UserInputProcessor` that always throws, regardless of image content --
/// used to prove a text-only failure passes through unchanged instead of
/// being misrepresented as a vision-content problem.
private struct AlwaysFailingProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        throw ImageProcessingProbeError()
    }
}

/// Minimal `LanguageModel` stand-in for `context.model`. Never reached by
/// either test here: `prepare(input:)` throws before any generation call
/// would touch it.
private final class TranscriptContentProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    var kvHeads: [Int] { [] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        fatalError("not reached -- processor.prepare(input:) throws first")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private func makeVisionModel(
    modelID: String, processor: any UserInputProcessor
) -> MLXLanguageModel {
    MLXLanguageModel(
        configuration: ModelConfiguration(id: modelID),
        capabilities: [.vision],
        weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
        load: { configuration, _ in
            ModelContainer(
                context: ModelContext(
                    configuration: configuration,
                    model: TranscriptContentProbeModel(),
                    processor: processor,
                    tokenizer: ByteTokenizer()))
        })
}

@Suite("MLXLanguageModel unsupportedTranscriptContent mapping")
struct UnsupportedTranscriptContentMappingTests {

    @Test("Image processing failure surfaces as unsupportedTranscriptContent, not a generic error")
    func imageProcessingFailureMapsToTypedError() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/vision-processing-failure-\(UUID().uuidString)"
        let model = makeVisionModel(modelID: modelID, processor: ImageFailingProcessor())
        let executor = try makeMLXExecutor(for: model)

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Describe this")),
                .attachment(attachment),
            ],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(prompt)]))
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
            try await executor.respond(to: request, model: model, streamingInto: channel)
            Issue.record("Expected unsupportedTranscriptContent, but respond returned")
        } catch let error as LanguageModelError {
            guard case .unsupportedTranscriptContent(let payload) = error else {
                Issue.record("Expected unsupportedTranscriptContent, got \(error)")
                return
            }
            #expect(payload.unsupportedContent.count == 1)
        } catch {
            Issue.record(
                "Expected LanguageModelError.unsupportedTranscriptContent, got: \(error)")
        }
    }

    @Test("Text-only prepare failure passes through unchanged, not remapped to unsupportedTranscriptContent")
    func textOnlyProcessingFailurePassesThroughUnchanged() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/text-only-processing-failure-\(UUID().uuidString)"
        let model = makeVisionModel(modelID: modelID, processor: AlwaysFailingProcessor())
        let executor = try makeMLXExecutor(for: model)

        let prompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "No images here"))],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [.prompt(prompt)]))
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
            try await executor.respond(to: request, model: model, streamingInto: channel)
            Issue.record("Expected ImageProcessingProbeError, but respond returned")
        } catch is ImageProcessingProbeError {
            // Expected: no image content this round, so the failure isn't
            // remapped to a vision-specific typed error.
        } catch {
            Issue.record(
                "Expected the original ImageProcessingProbeError to pass through unchanged, got: \(error)"
            )
        }
    }

    @Test(
        """
        A later, text-only turn's failure still maps to unsupportedTranscriptContent when an \
        EARLIER turn carried an image -- documents the deliberate whole-transcript heuristic \
        (see preparedInputMappingImageFailures's doc comment): every respond() call re-renders \
        the full transcript, so an older image is still part of what `prepare` processes this \
        round, not just brand-new attachments.
        """
    )
    func earlierTurnImageStillMapsLaterTextOnlyFailure() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let modelID = "test/multi-turn-earlier-image-\(UUID().uuidString)"
        let model = makeVisionModel(modelID: modelID, processor: AlwaysFailingProcessor())
        let executor = try makeMLXExecutor(for: model)

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let firstPrompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "What is this?")),
                .attachment(attachment),
            ],
            responseFormat: nil
        )
        let firstResponse = Transcript.Response(
            assetIDs: [],
            segments: [.text(Transcript.TextSegment(content: "It's a solid color square."))])
        let secondPrompt = Transcript.Prompt(
            segments: [.text(Transcript.TextSegment(content: "And now, unrelated: what's 2+2?"))],
            responseFormat: nil
        )
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(firstPrompt), .response(firstResponse), .prompt(secondPrompt),
            ]))
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
            try await executor.respond(to: request, model: model, streamingInto: channel)
            Issue.record("Expected unsupportedTranscriptContent, but respond returned")
        } catch let error as LanguageModelError {
            guard case .unsupportedTranscriptContent(let payload) = error else {
                Issue.record("Expected unsupportedTranscriptContent, got \(error)")
                return
            }
            // Only the first (image-bearing) entry is named -- the second,
            // text-only prompt never carried an image itself.
            #expect(payload.unsupportedContent.count == 1)
        } catch {
            Issue.record(
                "Expected LanguageModelError.unsupportedTranscriptContent, got: \(error)")
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
