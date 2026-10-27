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

private let qwen38FourBit = BenchmarkCheckpoint(
    id: "mlx-community/Qwen3.8-27B-4bit", factory: LLMModelFactory.shared)

private let museGlimmer = BenchmarkCheckpoint(
    id: "mlx-community/Muse-Glimmer-30B-4bit", factory: VLMModelFactory.shared)

/// The 4-bit MTP head of Qwen3.8-27B, the drafter of ``qwen38FourBit``.
private let qwen38FourBitMTPDrafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"

/// Whether the target and the drafter of the MTP benchmarks are local.
private let qwen38FourBitMTPIsLocal =
    qwen38FourBit.load != nil && hfSnapshotDir(modelId: qwen38FourBitMTPDrafterID) != nil

/// One expectation that the MTP chat benchmark holds each turn to.
private enum MTPChatCheck: String, CaseIterable {
    /// A follow-up turn of the MTP session reuses part of its prompt cache.
    case mtpReusesCache
    /// A follow-up turn of the plain session reuses part of its prompt cache.
    case plainReusesCache
    /// A follow-up turn of the MTP session feeds the drafter only the prompt past the cached prefix.
    case drafterPrefillsOnlyTheSuffix
    /// The MTP session writes the text of the plain session.
    case matchesPlainText
    /// A follow-up turn of the MTP session decodes faster than the plain session.
    case decodesFasterThanPlain

    /// Whether the first turn of a chat is held to the check: it has no cache to reuse.
    var appliesToFirstTurn: Bool { self == .matchesPlainText }

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
        case .matchesPlainText:
            return mtp.text == plain.text
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

/// An edit with MTP on rebuilds the whole chat: the session keeps no hybrid
/// checkpoints when speculation is on, and Qwen3.8's GDN layers cannot trim.
private let mtpCheckpointGap =
    "^achvxy1 keeps no checkpoints with MTP on, then ^ahnb0j9 restores the drafter with them"

/// MTP greedy text leaves plain text on warm follow-up turns.
private let mtpWarmTextGap = "^chgkh4c MTP text diverges from plain on warm follow-ups"

/// The checks the current code fails, each with the task that closes the gap.
/// A fix makes its known issue unrecorded, which fails the benchmark until the
/// row is removed.
private let mtpChatKnownGaps: [MTPChatGap: String] = {
    let followUps = MTPChatScenario.followUps.name
    let thinking = MTPChatScenario.thinkingFollowUps.name
    let edit = MTPChatScenario.editAndRegenerate
    let editTurn = edit.prompts.count
    return [
        MTPChatGap(scenario: followUps, turn: 2, check: .matchesPlainText): mtpWarmTextGap,
        MTPChatGap(scenario: thinking, turn: 1, check: .matchesPlainText): mtpWarmTextGap,
        MTPChatGap(scenario: thinking, turn: 2, check: .matchesPlainText): mtpWarmTextGap,
        MTPChatGap(scenario: edit.name, turn: editTurn, check: .mtpReusesCache): mtpCheckpointGap,
        MTPChatGap(scenario: edit.name, turn: editTurn, check: .drafterPrefillsOnlyTheSuffix):
            mtpCheckpointGap,
    ]
}()

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

    /// Plain against MTP decoding of Qwen3.8-27B-4bit with its 4-bit MTP head.
    @Test
    func qwen38FourBitMTPGeneration() async throws {
        let targetID = "mlx-community/Qwen3.8-27B-4bit"
        let drafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"
        guard hfSnapshotDir(modelId: targetID) != nil, hfSnapshotDir(modelId: drafterID) != nil
        else {
            print(
                "Skipping qwen38FourBitMTPGeneration: no local copy of \(targetID) or \(drafterID)")
            return
        }
        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: targetID))
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: drafterID))
        let results = try await benchmarkMTPGeneration(context: context, drafter: drafter.model)
        #expect(results.allSatisfy { $0.acceptanceRate > 0 })
    }

    /// MTP against plain decoding of Qwen3.8-27B-4bit in multi-turn chat sessions
    /// with a system prompt: follow-ups, follow-ups with thinking on, and an edit.
    ///
    /// Each follow-up turn reuses the prompt cache, feeds the drafter only the new
    /// suffix, writes the text of the plain session at temperature 0, and decodes
    /// faster than the plain session. A check the current code fails stands in
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
