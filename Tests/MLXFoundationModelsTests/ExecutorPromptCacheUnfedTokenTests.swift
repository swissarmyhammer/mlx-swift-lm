// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// The follow-up of an MTP turn that stopped on its token budget reads every token that turn
/// generated, and recomputes what the same turn recomputes without MTP (card `^g2t7caa`).
///
/// An MTP round returns its final verified token before any forward pass feeds it, thus the
/// caches of a turn that stops on its budget there miss that token. The entry keeps it as an
/// uncommitted token, and the splice of the follow-up feeds it ahead of the commit that closes
/// the turn. The usage of the follow-up does not count it: it is the end of the answer, not a
/// token of the prompt.
@Suite(.serialized)
struct ExecutorPromptCacheUnfedTokenTests {

    /// The tokens each turn generates. The MTP round that returns the last of them leaves it
    /// unfed on the fixture.
    private static let generatedTokenCount = 16

    /// The setup of a first turn that closes its reasoning and then runs out of budget in its
    /// answer.
    ///
    /// - Returns: The setup.
    /// - Throws: The error of the fixture.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeSetup() async throws -> ExecutorReasoningConversation.Setup {
        try await ExecutorReasoningConversation.makeSetup(
            .answersAfterReasoning, generatedTokenCount: Self.generatedTokenCount)
    }

    /// The MTP conversation of `setup`, whose first ledger misses the last token the turn
    /// generated.
    ///
    /// - Parameter setup: The setup.
    /// - Returns: What the passes left.
    /// - Throws: The error of a pass, or an issue when the first ledger holds another prefix of
    ///   the plain greedy decode than all of it but its last token.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func converseWithAnUnfedLastToken(_ setup: ExecutorReasoningConversation.Setup)
        async throws -> ExecutorReasoningConversation.Passes
    {
        let conversation = try await ExecutorReasoningConversation.converse(
            setup, speculates: true)
        let firstEntry = conversation.firstEntry
        try #require(conversation.first.speculated)
        try #require(
            firstEntry.tokens
                == firstEntry.renderTokens + setup.firstDecode.prefix(Self.generatedTokenCount - 1),
            "The first MTP pass must write the plain decode and leave its last token unfed.")
        return conversation
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theEntryOfTheTurnKeepsTheTokenItsCachesMiss() async throws {
        let setup = try await makeSetup()
        let conversation = try await converseWithAnUnfedLastToken(setup)

        #expect(conversation.firstEntry.uncommittedTokens == [setup.firstDecode.last])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpilledEntryKeepsTheTokenItsCachesMiss() async throws {
        let setup = try await makeSetup()
        let firstEntry = try await converseWithAnUnfedLastToken(setup).firstEntry
        // The follow-up extended the caches of the first entry, thus the spilled entry gets
        // caches of its own that hold the first ledger.
        let caches = try setup.fixture.target.newCache(parameters: nil)
        _ = try PromptCacheCheckpoint.prefill(
            firstEntry.tokens, into: caches, state: nil, model: setup.fixture.target,
            prefill: GenerateParameters().prefill)
        let entry = ExecutorPromptCacheEntry(
            caches: caches, tokens: firstEntry.tokens, renderTokens: firstEntry.renderTokens,
            uncommittedTokens: firstEntry.uncommittedTokens)
        try #require(!entry.uncommittedTokens.isEmpty)
        let key = ExecutorPromptCacheKey(
            modelID: Self.spillModelID, sessionID: ExecutorReasoningConversation.sessionID)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: Self.spillGeneration))
        defer { try? FileManager.default.removeItem(at: url) }

        try ExecutorPromptCacheFile.write(ExecutorPromptCacheFile.prepare(entry, key: key), to: url)
        let restored = try ExecutorPromptCacheFile.read(
            from: url, key: key, templates: setup.fixture.target.newCache(parameters: nil))

        #expect(restored.tokens == entry.tokens)
        #expect(restored.uncommittedTokens == entry.uncommittedTokens)
    }

    /// The model the key of the spilled entry names.
    private static let spillModelID = "executor-prompt-cache-unfed-token"

    /// The generation the file name of the spilled entry holds.
    private static let spillGeneration: UInt64 = 1

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpReadsEveryTokenTheTurnGeneratedBeforeTheCommit() async throws {
        let setup = try await makeSetup()
        let conversation = try await converseWithAnUnfedLastToken(setup)

        let entry = try #require(conversation.entry, "The follow-up pass must check in.")
        let firstTurn =
            conversation.firstEntry.renderTokens + setup.firstDecode + [setup.tokenizer.endOfTurn]
        #expect(entry.tokens.starts(with: firstTurn))
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpRecomputesWhatAPlainFollowUpRecomputes() async throws {
        let setup = try await makeSetup()
        let speculative = try await converseWithAnUnfedLastToken(setup)
        let plain = try await ExecutorReasoningConversation.converse(setup, speculates: false)

        let speculativeUsage = try #require(speculative.second.usage.first)
        let plainUsage = try #require(plain.second.usage.first)
        #expect(speculativeUsage.promptTokenCount == plainUsage.promptTokenCount)
        #expect(speculativeUsage.cachedPromptTokenCount == plainUsage.cachedPromptTokenCount)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpResumesTheDrafterStateOfTheTurn() async throws {
        let conversation = try await converseWithAnUnfedLastToken(makeSetup())

        // The splice feeds the unfed token and the follow-up onto the caches where the first
        // turn left them, thus the drafter state there still begins where that turn started it.
        #expect(try conversation.drafterState().state.cacheStartPosition == 0)
        #expect(conversation.second.speculated)
    }
}

#endif
