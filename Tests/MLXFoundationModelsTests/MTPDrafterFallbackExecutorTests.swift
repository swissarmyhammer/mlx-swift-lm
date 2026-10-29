// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLLM
import MLXLMCommon
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// An `MLXLanguageModel` whose MTP drafter cannot draft for the loaded target writes the plain
/// output and says why, in its text turns and in its guided generations. A drafter registration
/// that matches the target loads the drafter family of the loaded target instead.
@Suite(.serialized)
struct MTPDrafterFallbackExecutorTests {

    /// The tokens of the prompt of each turn.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The name of the drafter class that drafts for a vision-language target.
    private static let visionLanguageDrafterName = "Qwen35VLMNextNDraftModel"

    /// The name of the class of the text target of the fixture.
    private static let textTargetName = "Qwen35TextModel"

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static let greedy = GenerationOptions(
        samplingMode: .greedy, maximumResponseTokens: generatedTokenCount)

    /// Whether `statedReason` names the vision-language drafter and the text target.
    private func namesTheIncompatiblePair(_ statedReason: String?) -> Bool {
        guard let statedReason else { return false }
        return statedReason.contains(Self.visionLanguageDrafterName)
            && statedReason.contains(Self.textTargetName)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a text turn with an incompatible drafter writes the plain text and says why")
    func aTextTurnWithAnIncompatibleDrafterSaysWhy() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let plain = fixture.makeLanguageModel(weights: weights)
        let speculative = fixture.makeLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeIncompatibleSpeculativeDecoding())

        let turns = try await MTPExecutorTurns.respond(
            plain: plain, speculative: speculative,
            to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(!turns.plain.responseText.isEmpty)
        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(!turns.speculative.speculated)
        #expect(!turns.speculative.completions.isEmpty)
        for completion in turns.speculative.completions {
            #expect(completion.speculativeDecodingFallbackReason == .incompatibleDrafter)
            #expect(namesTheIncompatiblePair(completion.passthroughReason))
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided generation with an incompatible drafter decodes plainly and says why")
    func aGuidedGenerationWithAnIncompatibleDrafterSaysWhy() throws {
        let fixture = try Qwen35MTPFixture()
        let schema = GuidedFixtureSchema.activity.json

        let plain = try GuidedFixtureRun.generate(
            schema: schema, prompt: Self.prompt, fixture: fixture, blockSize: nil)
        let run = try GuidedFixtureRun.generate(
            schema: schema, prompt: Self.prompt, fixture: fixture,
            blockSize: Qwen35MTPFixture.blockSize, drafter: try Qwen35VLMMTPFixture().drafter)

        #expect(!plain.text.isEmpty)
        #expect(run.text == plain.text)
        #expect(run.sink.draftedTokenCount == 0)
        #expect(namesTheIncompatiblePair(run.sink.speculationFallbackReason))
        #expect(plain.sink.speculationFallbackReason == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        "speculative decoding that matches the target loads the drafter of the loaded target",
        arguments: [false, true])
    func matchingSpeculativeDecodingLoadsTheDrafterOfTheTarget(vision: Bool) async throws {
        let checkpoint = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: true, family: "qwen3_8")
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let speculativeDecoding = try MLXSpeculativeDecoding(
            drafterConfiguration: ModelConfiguration(directory: checkpoint.directory),
            from: UnusedDownloader(), using: UnusedTokenizerLoader(),
            registration: .matchingTarget(.qwen35Text, .qwen35VLM))
        let target: any MLXLMCommon.LanguageModel =
            vision ? try Qwen35VLMMTPFixture().target : try Qwen35MTPFixture().target

        let drafter = try await speculativeDecoding.drafterLoader(for: target)(
            speculativeDecoding.drafterConfiguration)

        let model = await drafter.perform { SendableBox($0.model) }.consume()
        #expect(
            vision
                ? model is MLXVLM.Qwen35VLMNextNDraftModel : model is MLXLLM.Qwen35MTPDraftModel)
    }
}

#endif
