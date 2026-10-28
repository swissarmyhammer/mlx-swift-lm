// Copyright © 2026 Apple Inc.
//
// Throughput and system prompt cache benchmarks on recently tuned checkpoints,
// measured with `BenchmarkHelpers`. The suite downloads nothing.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/GenerationBenchmarkTests`

import BenchmarkHelpers
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

/// A checkpoint under benchmark, loaded at most once per test process.
private struct BenchmarkCheckpoint: Sendable {
    let id: String
    let load: Task<ModelContainer, Error>?

    init(id: String, factory: any ModelFactory) {
        self.id = id
        self.load =
            hfSnapshotDir(modelId: id) == nil
            ? nil
            : Task {
                try await factory.loadContainer(
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    configuration: ModelConfiguration(id: id)
                ) { _ in }
            }
    }

    func containerOrSkip(testName: String) async throws -> ModelContainer? {
        guard let load else {
            print("Skipping \(testName): no local copy of \(id)")
            return nil
        }
        return try await load.value
    }
}

private let qwen38 = BenchmarkCheckpoint(
    id: "mlx-community/Qwen3.8-27B-mxfp4", factory: LLMModelFactory.shared)

/// The id of Qwen3.8-27B-4bit, which naming loads nothing.
private let qwen38FourBitID = "mlx-community/Qwen3.8-27B-4bit"

private let qwen38FourBit = BenchmarkCheckpoint(
    id: qwen38FourBitID, factory: LLMModelFactory.shared)

private let museGlimmer = BenchmarkCheckpoint(
    id: "mlx-community/Muse-Glimmer-30B-4bit", factory: VLMModelFactory.shared)

/// The 4-bit MTP head of Qwen3.8-27B, the drafter of ``qwen38FourBit``.
private let qwen38FourBitMTPDrafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"

/// Whether the target and the drafter of the MTP benchmarks are local.
private let qwen38FourBitMTPIsLocal =
    qwen38FourBit.load != nil && hfSnapshotDir(modelId: qwen38FourBitMTPDrafterID) != nil

/// Whether the files of the target and the drafter of the MTP benchmarks are local. Unlike
/// ``qwen38FourBitMTPIsLocal`` it starts no load of the shared ``qwen38FourBit`` container,
/// which the generation benchmarks that load their own model context do not use.
private let qwen38FourBitMTPFilesAreLocal =
    hfSnapshotDir(modelId: qwen38FourBitID) != nil
    && hfSnapshotDir(modelId: qwen38FourBitMTPDrafterID) != nil

/// One expectation that the MTP chat benchmark holds each turn to.
private enum MTPChatCheck: String, CaseIterable {
    /// A follow-up turn of the MTP session reuses part of its prompt cache.
    case mtpReusesCache
    /// A follow-up turn of the plain session reuses part of its prompt cache.
    case plainReusesCache
    /// A follow-up turn of the MTP session feeds the drafter only the prompt past the cached prefix.
    case drafterPrefillsOnlyTheSuffix
    /// The MTP session writes the tokens of the plain session, or parts from
    /// them only at a near-tie of plain's two best tokens
    /// (``MTPChatTokenMatch/nearTieLogitTolerance``).
    case matchesPlainTokens
    /// A follow-up turn of the MTP session decodes faster than the plain session.
    case decodesFasterThanPlain

    /// Whether the first turn of a chat is held to the check: it has no cache to reuse.
    var appliesToFirstTurn: Bool { self == .matchesPlainTokens }

    /// Whether the turn pair meets the check.
    func holds(mtp: MTPChatTurn, plain: MTPChatTurn) -> Bool {
        switch self {
        case .mtpReusesCache:
            return mtp.cachedPromptTokenCount > 0
        case .plainReusesCache:
            return plain.cachedPromptTokenCount > 0
        case .drafterPrefillsOnlyTheSuffix:
            return mtp.cachedPromptTokenCount > 0
                && mtp.drafterPrefillTokenCount == mtp.promptTokenCount
        case .matchesPlainTokens:
            return MTPChatTokenMatch(mtp: mtp.tokens, plain: plain.tokens).holds
        case .decodesFasterThanPlain:
            return mtp.decodeTokensPerSecond > plain.decodeTokensPerSecond
        }
    }
}

/// A check that one turn of one chat fails today.
private struct MTPChatGap: Hashable {
    let scenario: String
    let turn: Int
    let check: MTPChatCheck
}

/// The checks the current code fails, each with the task that closes the gap.
/// A fix makes its known issue unrecorded, which fails the benchmark until the
/// row is removed.
private let mtpChatKnownGaps: [MTPChatGap: String] = [:]

@Suite(.serialized, .timeLimit(.minutes(60)))
struct GenerationBenchmarkTests {

    /// Muse Glimmer reasons before it answers and has no switch to turn that off. A
    /// turn with no answer text is not recorded, so give it room to answer.
    private static let museGlimmerMaxTokens = 512
    private static let museGlimmerContext: [String: any Sendable] = ["reasoning_strength": "low"]

    private func generation(_ checkpoint: BenchmarkCheckpoint, testName: String) async throws {
        guard let container = try await checkpoint.containerOrSkip(testName: testName)
        else { return }
        let stats = try await benchmarkLLMGeneration(container: container)
        stats.printSummary(label: checkpoint.id)
    }

    private func systemPromptCache(
        _ checkpoint: BenchmarkCheckpoint, maxTokens: Int = 16,
        kvCache: KVCacheConfiguration? = nil,
        additionalContext: [String: any Sendable]? = nil, testName: String
    ) async throws {
        guard let container = try await checkpoint.containerOrSkip(testName: testName)
        else { return }
        let stats = try await benchmarkSystemPromptCache(
            container: container, maxTokens: maxTokens, kvCache: kvCache,
            additionalContext: additionalContext)
        stats.printSummary(label: checkpoint.id)
        #expect(stats.followUp.allSatisfy { $0.cachedPromptTokenCount > 0 })
        #expect(stats.fork.allSatisfy { $0.cachedPromptTokenCount > 0 })
    }

    @Test
    func qwen38Generation() async throws {
        try await generation(qwen38, testName: "qwen38Generation")
    }

    @Test
    func qwen38SystemPromptCache() async throws {
        try await systemPromptCache(qwen38, testName: "qwen38SystemPromptCache")
    }

    @Test
    func qwen38FourBitGeneration() async throws {
        try await generation(qwen38FourBit, testName: "qwen38FourBitGeneration")
    }

    @Test
    func qwen38FourBitSystemPromptCache() async throws {
        try await systemPromptCache(qwen38FourBit, testName: "qwen38FourBitSystemPromptCache")
    }

    @Test
    func museGlimmerGeneration() async throws {
        try await generation(museGlimmer, testName: "museGlimmerGeneration")
    }

    @Test
    func museGlimmerSystemPromptCache() async throws {
        try await systemPromptCache(
            museGlimmer, maxTokens: Self.museGlimmerMaxTokens,
            additionalContext: Self.museGlimmerContext,
            testName: "museGlimmerSystemPromptCache")
    }

    /// The sliding layers keep 512 rows past the window, so a fork can rewind them.
    @Test
    func museGlimmerSystemPromptCacheWithRewindReserve() async throws {
        try await systemPromptCache(
            museGlimmer, maxTokens: Self.museGlimmerMaxTokens,
            kvCache: KVCacheConfiguration(rewind: try .init(maxTokens: 512)),
            additionalContext: Self.museGlimmerContext,
            testName: "museGlimmerSystemPromptCacheWithRewindReserve")
    }

    /// The temperature of ``qwen38FourBitMTPSampledGeneration()``.
    private static let mtpSampledTemperature: Float = 0.7

    /// The top-p of ``qwen38FourBitMTPSampledGeneration()``.
    private static let mtpSampledTopP: Float = 0.95

    /// The seed of ``qwen38FourBitMTPSampledGeneration()``, so each run samples the same tokens.
    private static let mtpSampledSeed: UInt64 = 38

    /// The prompt whose throughput the sampled MTP benchmark holds above plain decoding.
    private static let mtpSampledCheckedPrompt = "math"

    /// The model context of Qwen3.8-27B-4bit and its 4-bit MTP head, loaded fresh.
    private func loadQwen38FourBitMTP() async throws -> (
        context: ModelContext, drafter: any MTPDrafterModel
    ) {
        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: qwen38FourBitID))
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: qwen38FourBitMTPDrafterID))
        return (context, drafter.model)
    }

    /// Plain against MTP decoding of Qwen3.8-27B-4bit with its 4-bit MTP head.
    @Test(.enabled(if: qwen38FourBitMTPFilesAreLocal))
    func qwen38FourBitMTPGeneration() async throws {
        let (context, drafter) = try await loadQwen38FourBitMTP()
        let results = try await benchmarkMTPGeneration(context: context, drafter: drafter)
        #expect(results.allSatisfy { $0.acceptanceRate > 0 })
    }

    /// Plain against MTP decoding of Qwen3.8-27B-4bit at temperature 0.7 and top-p 0.95, in
    /// each ``MTPSampledVerification`` mode, the default mode first.
    ///
    /// The default mode decodes the math prompt faster than plain decoding.
    @Test(.enabled(if: qwen38FourBitMTPFilesAreLocal))
    func qwen38FourBitMTPSampledGeneration() async throws {
        let (context, drafter) = try await loadQwen38FourBitMTP()
        let defaultVerification = GenerateParameters().mtpSampledVerification
        let modes =
            [defaultVerification]
            + MTPSampledVerification.allCases.filter { $0 != defaultVerification }
        for verification in modes {
            var sampling = GenerateParameters(
                temperature: Self.mtpSampledTemperature, topP: Self.mtpSampledTopP,
                seed: Self.mtpSampledSeed)
            sampling.mtpSampledVerification = verification
            let results = try await benchmarkMTPGeneration(
                context: context, drafter: drafter, sampling: sampling)
            #expect(results.allSatisfy { $0.acceptanceRate > 0 }, "\(verification)")
            let checked = try #require(
                results.first { $0.prompt == Self.mtpSampledCheckedPrompt })
            if verification == defaultVerification {
                #expect(
                    checked.mtpTokensPerSecond > checked.baselineTokensPerSecond,
                    """
                    \(verification) \(checked.prompt): MTP \(checked.mtpTokensPerSecond) tok/s, \
                    plain \(checked.baselineTokensPerSecond) tok/s
                    """)
            }
        }
    }

    /// MTP against plain decoding of Qwen3.8-27B-4bit in multi-turn chat sessions
    /// with a system prompt: follow-ups, follow-ups with thinking on, and an edit.
    ///
    /// Each follow-up turn reuses the prompt cache, feeds the drafter only the new
    /// suffix, writes the tokens of the plain session at temperature 0 up to a
    /// near-tie of plain's two best tokens, and decodes faster than the plain
    /// session. A check the current code fails stands in
    /// ``mtpChatKnownGaps`` with the task that closes it.
    @Test(.enabled(if: qwen38FourBitMTPIsLocal))
    func qwen38FourBitMTPSystemPromptCache() async throws {
        let container = try await #require(qwen38FourBit.load).value
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: qwen38FourBitMTPDrafterID))
        let results = try await benchmarkMTPSystemPromptCache(
            container: container, drafter: drafter)
        #expect(results.count == MTPChatScenario.defaults.count)
        for result in results {
            expectMTPChat(result)
        }
    }

    /// Holds each turn pair of `result` to each ``MTPChatCheck`` that applies to it.
    private func expectMTPChat(_ result: MTPChatScenarioResult) {
        #expect(result.mtp.count == result.plain.count)
        for (mtp, plain) in zip(result.mtp, result.plain) {
            let checks = MTPChatCheck.allCases.filter { mtp.index > 0 || $0.appliesToFirstTurn }
            for check in checks {
                expect(check, holdsFor: mtp, beside: plain, in: result.scenario.name)
            }
        }
    }

    /// Expects `check` to hold for the turn pair, as a known issue when
    /// ``mtpChatKnownGaps`` names it.
    private func expect(
        _ check: MTPChatCheck, holdsFor mtp: MTPChatTurn, beside plain: MTPChatTurn,
        in scenario: String
    ) {
        let holds = check.holds(mtp: mtp, plain: plain)
        let comment: Comment = """
            [\(scenario)] turn \(mtp.index) \(check.rawValue): \
            MTP cached \(mtp.cachedPromptTokenCount) prefilled \(mtp.promptTokenCount) \
            drafter \(mtp.drafterPrefillTokenCount.map(String.init) ?? "-") \
            speculation \(mtp.speculationDescription) \
            tokens \(MTPChatTokenMatch(mtp: mtp.tokens, plain: plain.tokens).summary) \
            \(String(format: "%.1f", mtp.decodeTokensPerSecond)) tok/s; \
            plain cached \(plain.cachedPromptTokenCount) \
            \(String(format: "%.1f", plain.decodeTokensPerSecond)) tok/s
            """
        let gap = MTPChatGap(scenario: scenario, turn: mtp.index, check: check)
        if let task = mtpChatKnownGaps[gap] {
            withKnownIssue("\(task)") {
                #expect(holds, comment)
            }
        } else {
            #expect(holds, comment)
        }
    }
}
