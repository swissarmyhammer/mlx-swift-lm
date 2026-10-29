// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// A `ChatSession` configured with a deferred MTP drafter loads the drafter on its first
/// speculative turn and not before, shares it with its forks, and asks the memory policy about
/// the drafter's byte estimate before it loads: a policy that does not admit the drafter decodes
/// without speculation, or throws.
@Suite(.serialized)
struct ChatSessionDeferredMTPDrafterTests {

    /// The tokens each turn generates: a few rounds of the fixture block size.
    private static let turnTokenCount = 6

    /// The byte estimate of the drafter that each deferred configuration reports.
    private static let drafterBytes = 4096

    /// The prompts of the turns of a chat, as tokens of the fixture vocabulary.
    private static let promptTokens = [[3, 14, 15], [9, 2, 6], [5, 35]]

    /// A memory budget that no model fits in.
    private static let exhaustedLimitBytes = 0

    /// A greedy session over the fixture target whose MTP drafter `load` loads when a turn
    /// first speculates.
    private func session(
        _ fixture: Qwen35MTPFixture, memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        load: @escaping @Sendable () async throws -> MTPDrafterContainer
    ) throws -> ChatSession {
        ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try SpeculativeDecodingConfig(
                mtpDrafterBytes: Self.drafterBytes, blockSize: Qwen35MTPFixture.blockSize,
                memoryPolicy: memoryPolicy, loadMTPDrafter: load),
            generateParameters: GenerateParameters(
                maxTokens: Self.turnTokenCount, temperature: 0))
    }

    /// The drafter of the fixture in a container, as a loader returns it.
    private static func drafterContainer(_ fixture: Qwen35MTPFixture) -> MTPDrafterContainer {
        MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration, model: fixture.drafter))
    }

    /// The MTP drafter source of `configuration`, or nil when it drafts with a draft model.
    private static func drafterSource(
        of configuration: SpeculativeDecodingConfig
    ) -> SpeculativeDecodingConfig.DrafterSource<MTPDrafterContainer>? {
        if case .mtp(let source, _, _) = configuration.strategy { return source }
        return nil
    }

    /// One turn of `session` over the prompt `tokens`.
    private func turn(
        _ session: ChatSession, _ fixture: Qwen35MTPFixture, tokens: [Int]
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        try await collectChatTurn(session.streamDetails(to: fixture.text(for: tokens)))
    }

    @Test("a deferred MTP drafter loads on the first turn, once across turns and a fork")
    func deferredDrafterLoadsOnceOnFirstTurn() async throws {
        let fixture = try Qwen35MTPFixture()
        let loads = DraftModelLoadCounter()
        let drafter = Self.drafterContainer(fixture)
        let session = try session(fixture) {
            await loads.increment()
            return drafter
        }
        #expect(await loads.value == 0)

        let first = try await turn(session, fixture, tokens: Self.promptTokens[0])
        #expect(await loads.value == 1)
        #expect((first.info.proposedDraftTokens ?? 0) > 0)
        #expect(first.info.speculativeDecodingFallbackReason == nil)

        let forked = try await turn(
            await session.fork(), fixture, tokens: Self.promptTokens[1])
        let second = try await turn(session, fixture, tokens: Self.promptTokens[2])
        #expect(await loads.value == 1)
        #expect((forked.info.proposedDraftTokens ?? 0) > 0)
        #expect((second.info.proposedDraftTokens ?? 0) > 0)
        #expect(second.info.cachedPromptTokenCount > 0)
    }

    @Test("a memory policy that falls back decodes plainly without loading the MTP drafter")
    func memoryPolicyFallbackSkipsTheLoad() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = try session(
            fixture,
            memoryPolicy: SpeculativeDecodingMemoryPolicy(
                limitBytes: Self.exhaustedLimitBytes, action: .fallbackToDefault)
        ) {
            throw UnexpectedDraftModelLoadError()
        }

        for tokens in Self.promptTokens.prefix(2) {
            let result = try await turn(session, fixture, tokens: tokens)
            #expect(result.info.speculativeDecodingFallbackReason == .memoryBudgetExceeded)
            #expect(result.info.speculativeDecodingTelemetry == nil)
            #expect(result.info.proposedDraftTokens == nil)
        }
    }

    @Test("a memory policy that fails throws without loading the MTP drafter")
    func memoryPolicyFailThrowsWithoutLoading() async throws {
        let fixture = try Qwen35MTPFixture()
        let session = try session(
            fixture,
            memoryPolicy: SpeculativeDecodingMemoryPolicy(
                limitBytes: Self.exhaustedLimitBytes, action: .fail)
        ) {
            throw UnexpectedDraftModelLoadError()
        }

        let error = await #expect(throws: SpeculativeDecodingMemoryError.self) {
            _ = try await turn(session, fixture, tokens: Self.promptTokens[0])
        }
        #expect(error?.evaluation.draftModelBytes == Self.drafterBytes)
        #expect(error?.evaluation.isWithinBudget == false)
    }

    @Test("a deferred MTP configuration reports its drafter bytes and no loaded drafter")
    func deferredConfigurationReportsItsBytes() throws {
        let configuration = try SpeculativeDecodingConfig(
            mtpDrafterBytes: Self.drafterBytes, blockSize: Qwen35MTPFixture.blockSize
        ) {
            throw UnexpectedDraftModelLoadError()
        }

        #expect(configuration.estimatedDraftModelBytes == Self.drafterBytes)
        #expect(configuration.mtpDrafter == nil)
        #expect(configuration.draftModel == nil)
        #expect(configuration.numDraftTokens == Qwen35MTPFixture.blockSize - 1)
    }

    @Test(
        "a deferred MTP configuration rejects verification blocks smaller than two",
        arguments: [-1, 0, 1])
    func deferredConfigurationRejectsSmallBlocks(blockSize: Int) {
        #expect(throws: SpeculativeDecodingConfigurationError.invalidMTPBlockSize(blockSize)) {
            try SpeculativeDecodingConfig(mtpDrafterBytes: Self.drafterBytes, blockSize: blockSize)
            {
                throw UnexpectedDraftModelLoadError()
            }
        }
    }

    @Test(
        "the Qwen MTP configurations load their drafter through the registered factory",
        arguments: [false, true])
    func qwenConfigurationsLoadThroughTheFactory(vision: Bool) async throws {
        let checkpoint = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: true, family: "qwen3_8")
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let configuration = ModelConfiguration(directory: checkpoint.directory)
        let speculativeDecoding = try SpeculativeDecodingConfig(
            mtpDrafterConfiguration: configuration, mtpDrafterBytes: Self.drafterBytes,
            from: UnusedDownloader(), using: UnusedTokenizerLoader(),
            registration: vision ? .qwen35VLM : .qwen35Text)
        #expect(speculativeDecoding.estimatedDraftModelBytes == Self.drafterBytes)
        let source = try #require(Self.drafterSource(of: speculativeDecoding))

        let drafter = try await source.load()
        let model = await drafter.perform { SendableBox($0.model) }.consume()
        #expect(
            vision
                ? model is MLXVLM.Qwen35VLMNextNDraftModel : model is MLXLLM.Qwen35MTPDraftModel)
    }
}

/// A downloader that refuses to download: a configuration that names a local directory needs
/// none.
private struct UnusedDownloader: Downloader {

    /// Refuses to download.
    ///
    /// - Parameters:
    ///   - id: The repository of the request.
    ///   - revision: The revision of the request.
    ///   - patterns: The file patterns of the request.
    ///   - useLatest: Whether the request asks for the latest revision.
    ///   - progressHandler: The progress callback of the request.
    /// - Returns: Never.
    /// - Throws: `ModelFactoryError.invalidConfiguration`, always.
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        throw ModelFactoryError.invalidConfiguration("A local MTP checkpoint downloads nothing.")
    }
}
