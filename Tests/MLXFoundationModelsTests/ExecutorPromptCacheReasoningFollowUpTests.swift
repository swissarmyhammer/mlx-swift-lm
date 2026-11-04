// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// A follow-up of a reasoning turn of the Qwen protocol reuses the whole prompt cache of that
/// turn, with the MTP drafter and without it, and resumes the drafter state of that turn.
///
/// The history render of a model whose protocol replays reasoning writes the reasoning of the
/// turn inside its think block, as `ChatSession` records it. A turn that stopped on its token
/// budget inside its reasoning leaves an empty response entry beside its reasoning entry, and
/// its follow-up extends the tokens the caches hold. A turn that closed its reasoning and then
/// answered gets its follow-up spliced at the commit that closes the turn.
@Suite(.serialized)
struct ExecutorPromptCacheReasoningFollowUpTests {

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The conversation of `firstTurn`, with the fixture's drafter or without it.
    ///
    /// - Parameters:
    ///   - firstTurn: How the first turn ends.
    ///   - speculates: Whether the model drafts with the fixture's drafter.
    /// - Returns: What the passes left.
    /// - Throws: The error of the setup or of a pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func converse(
        _ firstTurn: ExecutorReasoningConversation.FirstTurn, speculates: Bool
    ) async throws -> ExecutorReasoningConversation.Passes {
        try await ExecutorReasoningConversation.converse(
            ExecutorReasoningConversation.makeSetup(
                firstTurn, generatedTokenCount: Self.generatedTokenCount),
            speculates: speculates)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [false, true])
    func aFollowUpOfATurnThatStoppedInsideItsReasoningExtendsItsLedger(speculates: Bool)
        async throws
    {
        let conversation = try await converse(.stopsInsideReasoning, speculates: speculates)

        try #require(conversation.first.responseText.isEmpty)
        try #require(!conversation.first.reasoningText.isEmpty)
        let usage = try #require(conversation.second.usage.first)
        #expect(usage.cachedPromptTokenCount == conversation.firstLedger.count)
        #expect(conversation.second.speculated == speculates)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpOfATurnThatStoppedInsideItsReasoningResumesTheDrafterState() async throws {
        let conversation = try await converse(.stopsInsideReasoning, speculates: true)

        // A drafter state that the follow-up resumed still begins where the first turn
        // started it.
        #expect(try conversation.drafterState().state.cacheStartPosition == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [false, true])
    func aFollowUpOfATurnThatAnsweredReusesItsWholeLedger(speculates: Bool) async throws {
        let conversation = try await converse(.answersAfterReasoning, speculates: speculates)

        try #require(!conversation.first.responseText.isEmpty)
        try #require(!conversation.first.reasoningText.isEmpty)
        let usage = try #require(conversation.second.usage.first)
        #expect(usage.cachedPromptTokenCount >= conversation.firstLedger.count)
        #expect(conversation.second.speculated == speculates)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpOfATurnThatAnsweredResumesTheDrafterState() async throws {
        let conversation = try await converse(.answersAfterReasoning, speculates: true)

        // The splice feeds the follow-up onto the caches where the first turn left them, thus
        // the drafter state that stands there still begins where the first turn started it.
        #expect(try conversation.drafterState().state.cacheStartPosition == 0)
    }
}

#endif
