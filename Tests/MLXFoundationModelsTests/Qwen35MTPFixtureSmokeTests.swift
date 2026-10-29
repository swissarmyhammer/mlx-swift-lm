// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// Smoke tests of the shared Qwen 3.5 + MTP fixture in this bundle: it builds, it decodes the
/// same greedy tokens with and without MTP, and a `LanguageModelSession` runs on its target.
@Suite(.serialized)
struct Qwen35MTPFixtureSmokeTests {

    /// The tokens of the prompt that each decode of these tests starts from.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode of these tests generates.
    private static let generatedTokenCount = 16

    @Test
    func greedyMTPStreamEqualsPlainDecode() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true)

        #expect(plain.tokens.count == Self.generatedTokenCount)
        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func languageModelSessionRespondsWithThePlainGreedyDecode() async throws {
        let fixture = try Qwen35MTPFixture()
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let session = LanguageModelSession(
            model: fixture.makeLanguageModel(weights: weights), tools: [], instructions: nil)

        let response = try await session.respond(
            to: fixture.text(for: Self.prompt),
            options: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount))
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)

        #expect(
            fixture.tokenizer.encode(text: response.content, addSpecialTokens: false)
                == plain.tokens)
    }
}

#endif
