// Copyright © 2026 Apple Inc.

import MLXLMCommon
import MLXLMTestSupport
import Testing

/// Smoke tests of the shared Qwen 3.5 + MTP fixture: it builds, it decodes the
/// same greedy tokens with and without MTP, and its two drafter modes give the
/// acceptance the MTP suites rely on.
@Suite(.serialized)
struct Qwen35MTPFixtureSmokeTests {

    /// The tokens of the prompt that each decode of these tests starts from.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode of these tests generates.
    private static let generatedTokenCount = 48

    /// The least share of drafted tokens the target accepts from the copying
    /// drafter.
    private static let copyingAcceptanceFloor = 0.5

    /// The most share of drafted tokens the target accepts from the
    /// independent drafter.
    private static let independentAcceptanceCeiling = 0.25

    @Test(arguments: [Qwen35MTPFixture.DrafterMode.copiesTarget, .independent])
    func greedyMTPStreamEqualsPlainDecode(mode: Qwen35MTPFixture.DrafterMode) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)

        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true)

        #expect(plain.tokens.count == Self.generatedTokenCount)
        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculative.info.passthroughReason == nil)
    }

    @Test
    func copyingDrafterIsMostlyAccepted() async throws {
        let rate = try await acceptanceRate(of: Qwen35MTPFixture(drafter: .copiesTarget))
        #expect(rate >= Self.copyingAcceptanceFloor)
    }

    @Test
    func independentDrafterIsMostlyRejected() async throws {
        let rate = try await acceptanceRate(of: Qwen35MTPFixture(drafter: .independent))
        #expect(rate <= Self.independentAcceptanceCeiling)
    }

    @Test
    func theSameSeedBuildsTheSameWeights() async throws {
        let first = try await plainDecode(of: Qwen35MTPFixture())
        let second = try await plainDecode(of: Qwen35MTPFixture())
        let reseeded = try await plainDecode(
            of: Qwen35MTPFixture(seed: Qwen35MTPFixture.defaultSeed + 1))

        #expect(first.tokens == second.tokens)
        #expect(reseeded.tokens != first.tokens)
    }

    @Test
    func targetPredictionsDependOnTheContext() async throws {
        let fixture = try Qwen35MTPFixture()
        let lastToken = try #require(Self.prompt.last)
        let otherContext = Self.prompt.dropLast().reversed() + [lastToken]

        let original = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let reordered = try await fixture.decodeGreedily(
            prompt: Array(otherContext), maxTokens: Self.generatedTokenCount,
            speculative: false)

        #expect(original.tokens != reordered.tokens)
    }

    @Test
    func multiTurnChatSessionWithMTPEqualsPlainSession() async throws {
        let fixture = try Qwen35MTPFixture()
        let speculative = try fixture.makeChatSession(
            speculative: true, maxTokens: Self.generatedTokenCount)
        let plain = try fixture.makeChatSession(
            speculative: false, maxTokens: Self.generatedTokenCount)
        let turns = [
            fixture.text(for: Self.prompt), fixture.text(for: Array(Self.prompt.prefix(2))),
        ]

        for turn in turns {
            let speculativeReply = try await speculative.respond(to: turn)
            let plainReply = try await plain.respond(to: turn)
            #expect(speculativeReply == plainReply)
        }
    }

    /// One greedy decode of ``prompt`` by the target of `fixture`, without MTP.
    private func plainDecode(of fixture: Qwen35MTPFixture) async throws
        -> Qwen35MTPFixture.GreedyDecode
    {
        try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
    }

    /// The share of drafted tokens the target of `fixture` accepts in one
    /// greedy MTP decode of ``prompt``.
    private func acceptanceRate(of fixture: Qwen35MTPFixture) async throws -> Double {
        let info = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true
        ).info
        let proposed = try #require(info.proposedDraftTokens)
        let accepted = try #require(info.acceptedDraftTokens)
        try #require(proposed > 0)
        return Double(accepted) / Double(proposed)
    }
}
