// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// The memory policy of an `MLXSpeculativeDecoding` admits the MTP drafter beside the model
/// before the drafter loads, as the memory policy of a `ChatSession` does: a drafter it does not
/// admit never loads, and the call decodes plainly and says why, or throws.
///
/// Each test runs the shared Qwen 3.5 + MTP fixture. The support types live in
/// `MTPExecutorTestSupport.swift` and `Qwen35MTPFixture+LanguageModel.swift`.
@Suite(.serialized)
struct MLXSpeculativeDecodingMemoryPolicyTests {

    /// The tokens of the prompt of each turn.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The greedy options of each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static let greedy = GenerationOptions(
        samplingMode: .greedy, maximumResponseTokens: generatedTokenCount)

    /// The byte estimate of the drafter the tests hand the policy.
    private static let drafterByteEstimate = 4096

    /// A budget no model fits in.
    private static let exhaustedBudgetBytes = 1

    /// The metadata key of the fallback reason of a call.
    private static let fallbackReasonKey = "speculativeDecodingFallbackReason"

    /// The metadata key of whether MTP drafted in a call.
    private static let speculativeDecodingKey = "speculativeDecoding"

    /// The fallback reason of a drafter the memory policy did not admit.
    private static let memoryBudgetExceeded = "memoryBudgetExceeded"

    /// A policy with no room for the drafter, which takes `action`.
    private static func exhaustedPolicy(
        _ action: SpeculativeDecodingMemoryAction
    ) -> SpeculativeDecodingMemoryPolicy {
        SpeculativeDecodingMemoryPolicy(limitBytes: exhaustedBudgetBytes, action: action)
    }

    /// The fixture's target with the fixture's drafter under `memoryPolicy`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModel(
        fixture: Qwen35MTPFixture, weights: URL, log: DrafterLoadLog,
        drafterBytes: Int? = drafterByteEstimate, memoryPolicy: SpeculativeDecodingMemoryPolicy
    ) throws -> MLXLanguageModel {
        fixture.makeLanguageModel(
            weights: weights,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding(
                log: log, drafterBytes: drafterBytes, memoryPolicy: memoryPolicy))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the speculative decoding keeps its memory policy and its byte estimate")
    func keepsItsMemoryPolicy() throws {
        let policy = Self.exhaustedPolicy(.fail)

        let speculativeDecoding = try Qwen35MTPFixture.makeSpeculativeDecoding(
            drafterBytes: Self.drafterByteEstimate, memoryPolicy: policy)

        #expect(speculativeDecoding.memoryPolicy == policy)
        #expect(speculativeDecoding.drafterBytes == Self.drafterByteEstimate)
        #expect(try Qwen35MTPFixture.makeSpeculativeDecoding().memoryPolicy == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a drafter the policy does not admit never loads, and the turn decodes plainly")
    func aDrafterTheFallbackPolicyRefusesDecodesPlainly() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let speculative = try makeModel(
            fixture: fixture, weights: weights, log: log,
            memoryPolicy: Self.exhaustedPolicy(.fallbackToDefault))

        let turns = try await MTPExecutorTurns.respond(
            plain: fixture.makeLanguageModel(weights: weights), speculative: speculative,
            to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(log.loadCount == 0)
        #expect(!turns.plain.responseText.isEmpty)
        #expect(turns.speculative.responseText == turns.plain.responseText)
        #expect(!turns.speculative.speculated)
        #expect(turns.speculative.metadata[Self.speculativeDecodingKey] as? Bool == false)
        #expect(
            turns.speculative.metadata[Self.fallbackReasonKey] as? String
                == Self.memoryBudgetExceeded)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a drafter a failing policy does not admit never loads, and the turn throws")
    func aDrafterTheFailingPolicyRefusesThrows() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let session = LanguageModelSession(
            model: try makeModel(
                fixture: fixture, weights: weights, log: log,
                memoryPolicy: Self.exhaustedPolicy(.fail)),
            tools: [], instructions: nil)

        await #expect(throws: SpeculativeDecodingMemoryError.self) {
            _ = try await session.respond(
                to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)
        }
        #expect(log.loadCount == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a drafter the policy admits loads, and the turn speculates")
    func aDrafterThePolicyAdmitsSpeculates() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let session = LanguageModelSession(
            model: try makeModel(
                fixture: fixture, weights: weights, log: log,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(action: .fail)),
            tools: [], instructions: nil)

        let turn = try await MTPExecutorTurns.respond(
            session, to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(log.loadCount == 1)
        #expect(turn.speculated)
        #expect(turn.metadata[Self.speculativeDecodingKey] as? Bool == true)
        #expect(turn.metadata[Self.fallbackReasonKey] == nil)
    }

    /// With no byte estimate the policy has nothing to weigh before the load, so it weighs the
    /// parameters of the loaded drafter, as a `ChatSession` does with a drafter it was handed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("with no byte estimate the policy weighs the loaded drafter")
    func withoutAnEstimateThePolicyWeighsTheLoadedDrafter() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let session = LanguageModelSession(
            model: try makeModel(
                fixture: fixture, weights: weights, log: log, drafterBytes: nil,
                memoryPolicy: Self.exhaustedPolicy(.fallbackToDefault)),
            tools: [], instructions: nil)

        let turn = try await MTPExecutorTurns.respond(
            session, to: Prompt(fixture.text(for: Self.prompt)), options: Self.greedy)

        #expect(log.loadCount == 1)
        #expect(!turn.speculated)
        #expect(turn.metadata[Self.fallbackReasonKey] as? String == Self.memoryBudgetExceeded)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a warm-up under a policy that does not admit the drafter loads no drafter")
    func aWarmUpUnderARefusingPolicyLoadsNoDrafter() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let log = DrafterLoadLog()
        let model = try makeModel(
            fixture: fixture, weights: weights, log: log,
            memoryPolicy: Self.exhaustedPolicy(.fallbackToDefault))

        try await model.warmUp()

        #expect(log.loadCount == 0)
        await model.evict()
    }
}

#endif
