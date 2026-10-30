// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import MLXLMCommon
import MLXLMTestSupport
import Testing

/// The MTP chat benchmark runs its MTP sessions under the adaptive guard it is given, and reports
/// what the guard did in each turn.
@Suite(.serialized)
struct MTPChatBenchmarkAdaptivePolicyTests {

    /// The tokens each turn generates: enough for the guard of
    /// ``MTPAdaptivePolicy/losingEveryRound`` to pause speculation.
    private static let turnTokenBudget = 40

    /// The system prompt of both sessions: any text, which the fixture's tokenizer maps to tokens.
    private static let systemPrompt = "system"

    /// A two-turn chat, so the second turn speculates from a reused cache.
    private static let scenario = MTPChatScenario(
        name: "fixture", thinking: false, prompts: ["first", "second"],
        maxTokens: turnTokenBudget)

    /// The MTP turns of the benchmark on the fixture under `policy`.
    private func mtpTurns(policy: MTPAdaptivePolicy) async throws -> [MTPChatTurn] {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let drafter = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: fixture.modelConfiguration, model: fixture.drafter))
        let results = try await Benchmark.mtpSystemPromptCache(
            container: ModelContainer(context: fixture.makeModelContext()), drafter: drafter,
            scenarios: [Self.scenario], systemPrompt: Self.systemPrompt,
            blockSize: Qwen35MTPFixture.blockSize, adaptivePolicy: policy)
        return try #require(results.first).mtp
    }

    @Test("a guard that loses every round pauses speculation in each MTP turn")
    func losingGuardPausesEachTurn() async throws {
        let turns = try await mtpTurns(policy: .losingEveryRound)

        #expect(turns.count == Self.scenario.prompts.count)
        for turn in turns {
            let telemetry = try #require(turn.speculativeDecodingTelemetry, "turn \(turn.index)")
            #expect(telemetry.adaptiveBlockSize != nil, "turn \(turn.index)")
            #expect(telemetry.adaptivePauseCount > 0, "turn \(turn.index)")
        }
    }

    @Test("a disabled guard runs the MTP turns without a guard")
    func disabledGuardRunsWithoutAGuard() async throws {
        let turns = try await mtpTurns(policy: .disabled)

        for turn in turns {
            let telemetry = try #require(turn.speculativeDecodingTelemetry, "turn \(turn.index)")
            #expect(telemetry.adaptiveBlockSize == nil, "turn \(turn.index)")
            #expect(telemetry.adaptivePauseCount == 0, "turn \(turn.index)")
        }
    }
}
