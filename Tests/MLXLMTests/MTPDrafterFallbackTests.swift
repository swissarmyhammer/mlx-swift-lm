// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// A configured MTP drafter that cannot draft for its target decodes plainly and says why: the
/// completion information names the fallback and states the reason, and the reason is logged
/// once for each drafter and target. A registration that chooses by target loads the drafter
/// family of the target it is given, so the target keeps speculating.
@Suite(.serialized)
struct MTPDrafterFallbackTests {

    /// The tokens of the prompt of each decode.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode generates.
    private static let generatedTokenCount = 12

    /// The byte estimate of the drafter of each deferred configuration.
    private static let drafterBytes = 4096

    /// The name of the drafter class that drafts for a vision-language target.
    private static let visionLanguageDrafterName = "Qwen35VLMNextNDraftModel"

    /// The name of the class of the text target of the fixture.
    private static let textTargetName = "Qwen35TextModel"

    /// The speculation of the vision-language drafter, which cannot draft for the text target of
    /// ``Qwen35MTPFixture``.
    private func incompatibleSpeculation() throws -> MTPSpeculation {
        MTPSpeculation(
            drafter: try Qwen35VLMMTPFixture().drafter, blockSize: Qwen35MTPFixture.blockSize,
            minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
    }

    /// The greedy parameters of each decode.
    private var parameters: GenerateParameters {
        GenerateParameters(maxTokens: Self.generatedTokenCount, temperature: 0)
    }

    /// The prompt of each decode as model input.
    private var input: LMInput {
        LMInput(tokens: MLXArray(Self.prompt.map(Int32.init)))
    }

    /// Runs `generateTaskRecordingTokens` over the fixture's target.
    private func recordingRun(
        _ fixture: Qwen35MTPFixture, speculation: MTPSpeculation?
    ) async throws -> (tokens: [Int], info: GenerateCompletionInfo) {
        let (stream, task) = try generateTaskRecordingTokens(
            input: input, parameters: parameters, context: fixture.makeModelContext(),
            speculation: speculation)
        var info: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let completion) = event { info = completion }
        }
        return (await task.value, try #require(info))
    }

    /// Runs the package `generateTokensTask` over the fixture's target.
    private func tokensRun(
        _ fixture: Qwen35MTPFixture, speculation: MTPSpeculation?
    ) async throws -> Qwen35MTPFixture.GreedyDecode {
        let (stream, _) = try generateTokensTask(
            input: input, parameters: parameters, context: fixture.makeModelContext(),
            speculation: speculation)
        return try await Qwen35MTPFixture.collect(stream)
    }

    /// Whether `statedReason` names the vision-language drafter and the text target.
    private func namesTheIncompatiblePair(_ statedReason: String?) -> Bool {
        guard let statedReason else { return false }
        return statedReason.contains(Self.visionLanguageDrafterName)
            && statedReason.contains(Self.textTargetName)
    }

    // MARK: - Package generation

    @Test("a recording generation with an incompatible drafter decodes plainly and says why")
    func recordingGenerationReportsAnIncompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await recordingRun(fixture, speculation: nil)
        let run = try await recordingRun(fixture, speculation: try incompatibleSpeculation())

        #expect(run.tokens == plain.tokens)
        #expect(run.info.proposedDraftTokens == nil)
        #expect(run.info.speculativeDecodingFallbackReason == .incompatibleDrafter)
        #expect(namesTheIncompatiblePair(run.info.passthroughReason))
    }

    @Test("a token generation with an incompatible drafter decodes plainly and says why")
    func tokenGenerationReportsAnIncompatibleDrafter() async throws {
        let fixture = try Qwen35MTPFixture()

        let plain = try await tokensRun(fixture, speculation: nil)
        let run = try await tokensRun(fixture, speculation: try incompatibleSpeculation())

        #expect(run.tokens == plain.tokens)
        #expect(run.info.speculativeDecodingFallbackReason == .incompatibleDrafter)
        #expect(namesTheIncompatiblePair(run.info.passthroughReason))
    }

    @Test("a plain generation reports no drafter fallback")
    func plainGenerationReportsNoFallback() async throws {
        let run = try await recordingRun(try Qwen35MTPFixture(), speculation: nil)

        #expect(run.info.speculativeDecodingFallbackReason == nil)
        #expect(run.info.passthroughReason == nil)
    }

    // MARK: - ChatSession

    @Test("a chat session with an incompatible drafter decodes plainly and states why")
    func chatSessionStatesWhyAnIncompatibleDrafterDecodesPlainly() async throws {
        let fixture = try Qwen35MTPFixture()
        let drafter = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration,
                model: try Qwen35VLMMTPFixture().drafter))
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafter: drafter, blockSize: Qwen35MTPFixture.blockSize),
            generateParameters: parameters)
        let plain = ChatSession(fixture.makeModelContext(), generateParameters: parameters)
        let prompt = fixture.text(for: Self.prompt)

        let result = try await collectChatTurn(session.streamDetails(to: prompt))
        let expected = try await collectChatTurn(plain.streamDetails(to: prompt))

        #expect(result.text == expected.text)
        #expect(result.info.speculativeDecodingFallbackReason == .incompatibleDrafter)
        #expect(namesTheIncompatiblePair(result.info.passthroughReason))
    }

    // MARK: - The log

    @Test("the fallback of a drafter and target is logged once")
    func theFallbackOfAPairIsLoggedOnce() throws {
        let fixture = try Qwen35MTPFixture()
        let visionLanguage = try Qwen35VLMMTPFixture()
        let written = WrittenMessages()
        let log = MTPDrafterFallbackLog { written.append($0) }
        let textTargetFallback = try #require(
            MTPDrafterFallback.incompatibility(
                of: visionLanguage.drafter, with: fixture.target))
        let visionLanguageTargetFallback = try #require(
            MTPDrafterFallback.incompatibility(
                of: fixture.drafter, with: visionLanguage.target))

        log.report(textTargetFallback)
        log.report(textTargetFallback)
        log.report(visionLanguageTargetFallback)

        let messages = written.messages
        #expect(messages.count == 2)
        #expect(messages.first?.contains(textTargetFallback.statedReason) == true)
        #expect(messages.last?.contains(visionLanguageTargetFallback.statedReason) == true)
    }

    @Test("a compatible drafter has no fallback")
    func aCompatibleDrafterHasNoFallback() throws {
        let fixture = try Qwen35MTPFixture()

        let fallback = MTPDrafterFallback.incompatibility(
            of: fixture.drafter, with: fixture.target)

        #expect(fallback == nil)
    }

    // MARK: - Registration

    @Test("a registration that matches the target picks the drafter family of each target")
    func matchingRegistrationPicksTheFamilyOfTheTarget() throws {
        let registration = MTPDrafterRegistration.matchingTarget(.qwen35Text, .qwen35VLM)

        let forText = registration.registration(for: try Qwen35MTPFixture().target)
        let forVisionLanguage = registration.registration(for: try Qwen35VLMMTPFixture().target)

        #expect(forText.factory === MTPDrafterModelFactory.shared)
        #expect(forVisionLanguage.factory === MTPDrafterModelFactory.visionLanguage)
    }

    @Test("each Qwen registration drafts for its own kind of target")
    func eachQwenRegistrationDraftsForItsOwnTarget() throws {
        let textTarget = try Qwen35MTPFixture().target
        let visionLanguageTarget = try Qwen35VLMMTPFixture().target

        #expect(MTPDrafterRegistration.qwen35Text.drafts(for: textTarget))
        #expect(!MTPDrafterRegistration.qwen35Text.drafts(for: visionLanguageTarget))
        #expect(MTPDrafterRegistration.qwen35VLM.drafts(for: visionLanguageTarget))
        #expect(!MTPDrafterRegistration.qwen35VLM.drafts(for: textTarget))
    }

    @Test(
        "the drafter loader of a registration that matches the target loads its target's drafter",
        arguments: [false, true])
    func matchingRegistrationLoaderLoadsTheDrafterOfItsTarget(vision: Bool) async throws {
        try await expectLoadsTheDrafterOfItsTarget(vision: vision) { checkpoint, target in
            let load = MTPDrafterRegistration.matchingTarget(.qwen35Text, .qwen35VLM)
                .drafterLoader(
                    for: target, from: UnusedDownloader(), using: UnusedTokenizerLoader())
            return try await load(checkpoint)
        }
    }

    @Test(
        "a chat session configuration that matches the target loads the drafter of its target",
        arguments: [false, true])
    func chatSessionConfigurationLoadsTheDrafterOfItsTarget(vision: Bool) async throws {
        try await expectLoadsTheDrafterOfItsTarget(vision: vision) { checkpoint, target in
            let configuration = try SpeculativeDecodingConfig(
                mtpDrafterConfiguration: checkpoint,
                mtpDrafterBytes: Self.drafterBytes, from: UnusedDownloader(),
                using: UnusedTokenizerLoader(),
                registration: .matchingTarget(.qwen35Text, .qwen35VLM))
            let source = try #require(configuration.mtpDrafterSource)
            return try await source.loader(for: target)()
        }
    }

    /// Loads a standalone Qwen3.8 MTP checkpoint with `load` for the text or the
    /// vision-language target, and expects the drafter class of that target.
    ///
    /// - Parameters:
    ///   - vision: whether the checkpoint and the target are the vision-language ones.
    ///   - load: loads the drafter of the checkpoint for the target.
    private func expectLoadsTheDrafterOfItsTarget(
        vision: Bool,
        load: (ModelConfiguration, any LanguageModel) async throws -> MTPDrafterContainer
    ) async throws {
        let checkpoint = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: true, family: "qwen3_8")
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let target: any LanguageModel =
            vision ? try Qwen35VLMMTPFixture().target : try Qwen35MTPFixture().target

        let drafter = try await load(ModelConfiguration(directory: checkpoint.directory), target)

        let model = await drafter.perform { SendableBox($0.model) }.consume()
        #expect(
            vision
                ? model is MLXVLM.Qwen35VLMNextNDraftModel : model is MLXLLM.Qwen35MTPDraftModel)
    }
}

/// The messages a fallback log wrote, in order.
// swiftlint:disable:next no_unchecked_sendable  every access to `written` holds `lock`
private final class WrittenMessages: @unchecked Sendable {

    /// Guards ``written``.
    private let lock = NSLock()

    /// The messages so far.
    private var written: [String] = []

    /// Adds one message.
    ///
    /// - Parameter message: The message the log wrote.
    func append(_ message: String) {
        lock.withLock { written.append(message) }
    }

    /// The messages so far, in order.
    var messages: [String] {
        lock.withLock { written }
    }
}
