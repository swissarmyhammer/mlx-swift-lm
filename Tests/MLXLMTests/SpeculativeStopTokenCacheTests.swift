// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXScriptedLM
import Testing

@testable import MLXLMCommon

/// A draft-model stream reports whether the main cache holds the stop token.
///
/// A round of ``SpeculativeTokenIterator`` returns the accepted drafts, which
/// the verify pass fed to the main cache, and then the final token of the
/// round, which no forward pass fed yet. The generation loop names a stop
/// token in ``GenerateCompletionInfo/stopTokenFedToCache`` only when the main
/// cache holds it, thus a caller that keeps a token ledger beside the cache
/// (the FoundationModels executor, `MLXLanguageModel.generatedTokenCount(of:)`)
/// stays equal to the cache.
@Suite("A draft-model stream and the cache state of its stop token")
struct SpeculativeStopTokenCacheTests {

    /// The vocabulary of the transition models.
    private static let vocabularySize = 100

    /// The prompt of the stream.
    private static let prompt = [7]

    /// The drafts each round proposes.
    private static let numDraftTokens = 3

    /// The tokens the first round returns from ``prompt``: the three accepted
    /// drafts, then the final token of the round. The draft model and the main
    /// model are the same transition, thus the verify pass accepts each draft.
    private static let firstRoundTokens = [24, 51, 88, 35]

    /// The position in ``firstRoundTokens`` of the stop token, and whether the
    /// main cache holds that token when the iterator returns it.
    enum StopSlot: CaseIterable, Sendable {
        /// The stop token is an accepted draft, which the verify pass fed.
        case acceptedDraft
        /// The stop token is the final token of the round, which no pass fed.
        case roundFinal

        /// The index of the stop token in ``firstRoundTokens``.
        var index: Int {
            switch self {
            case .acceptedDraft: 1
            case .roundFinal: SpeculativeStopTokenCacheTests.numDraftTokens
            }
        }

        /// Whether the main cache holds the stop token when the loop reads it.
        var isFed: Bool {
            switch self {
            case .acceptedDraft: true
            case .roundFinal: false
            }
        }
    }

    /// What a draft-model stream reported, and the main cache after it.
    private struct StoppedStream {
        /// The completion info of the stream.
        let info: GenerateCompletionInfo
        /// The main cache after the loop finalized the generation.
        let mainCache: [KVCache]
    }

    /// Runs a draft-model stream from ``prompt`` that stops on `stopToken`.
    private func stream(stoppingOn stopToken: Int) async throws -> StoppedStream {
        let processor = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(vocabularySize: Self.vocabularySize),
            configuration: ModelConfiguration(id: "test/stop", eosTokenIds: [stopToken]),
            messageGenerator: DefaultMessageGenerator())
        let context = ModelContext(
            configuration: processor.configuration,
            model: CacheTrackingTransitionModel(vocabularySize: Self.vocabularySize),
            processor: processor, tokenizer: processor.tokenizer)
        let mainCache = [KVCacheSimple()]
        var info: GenerateCompletionInfo?
        for await generation in try generateTokens(
            input: LMInput(tokens: MLXArray(Self.prompt)), cache: mainCache,
            parameters: GenerateParameters(maxTokens: 16, temperature: 0), context: context,
            draftModel: CacheTrackingTransitionModel(vocabularySize: Self.vocabularySize),
            draftCache: [KVCacheSimple()], numDraftTokens: Self.numDraftTokens)
        {
            if let completion = generation.info {
                info = completion
            }
        }
        return StoppedStream(info: try #require(info), mainCache: mainCache)
    }

    @Test(
        "names the stop token only when the main cache holds it",
        arguments: StopSlot.allCases)
    func namesTheStopTokenOnlyWhenTheMainCacheHoldsIt(slot: StopSlot) async throws {
        let stopToken = Self.firstRoundTokens[slot.index]

        let stopped = try await stream(stoppingOn: stopToken)

        #expect(stopped.info.stopReason == .stop)
        #expect(stopped.info.discardedStopToken == stopToken)
        #expect(stopped.info.stopTokenFedToCache == (slot.isFed ? stopToken : nil))
        #expect(stopped.info.generationTokenCount == slot.index)
    }

    @Test(
        "leaves the main cache at the prompt, the generated tokens and a fed stop token",
        arguments: StopSlot.allCases)
    func leavesTheMainCacheAtTheLedger(slot: StopSlot) async throws {
        let stopped = try await stream(stoppingOn: Self.firstRoundTokens[slot.index])

        let ledger =
            Self.prompt.count + stopped.info.generationTokenCount
            + (stopped.info.stopTokenFedToCache == nil ? 0 : 1)
        #expect(stopped.mainCache.first?.offset == ledger)
    }
}
