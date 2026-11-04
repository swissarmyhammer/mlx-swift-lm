// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The output count a pass reports must name every token the model wrote,
/// the stop token included.
///
/// `GenerateCompletionInfo.generationTokenCount` leaves the stop token out
/// when the loop suppresses it, and names it in `discardedStopToken`. Plain
/// decode feeds that token to the caches before it returns it, and names it in
/// `stopTokenFedToCache` too; MTP decode can return it as the last token of a
/// round, which no forward pass fed. The model wrote the token either way, thus
/// a turn reports the same output count with MTP and without it.
@Suite("The output count of a pass")
struct ExecutorUsageOutputTests {

    /// Tokens the fixture generated before the stop token.
    private static let generatedTokenCount = 27

    /// The stop token that ended the fixture's generation.
    private static let stopToken = 248_046

    /// A completion report with `generatedTokenCount` tokens, the stop token
    /// `discarded` the loop left out of the output, and the stop token `fed`
    /// the caches hold.
    private func info(discarded: Int?, fed: Int?) -> GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: 1, generationTokenCount: Self.generatedTokenCount,
            promptTime: 0, generationTime: 0, stopTokenFedToCache: fed,
            discardedStopToken: discarded)
    }

    @Test("counts the stop token the caches hold")
    func countsTheStopTokenTheCachesHold() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.generatedTokenCount(
                of: info(discarded: Self.stopToken, fed: Self.stopToken))
                == Self.generatedTokenCount + 1)
    }

    @Test("counts the stop token no forward pass fed")
    func countsTheStopTokenNoForwardPassFed() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.generatedTokenCount(
                of: info(discarded: Self.stopToken, fed: nil))
                == Self.generatedTokenCount + 1)
    }

    @Test("counts no stop token when the pass ended without one")
    func countsNoStopTokenWhenThePassEndedWithoutOne() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.generatedTokenCount(of: info(discarded: nil, fed: nil))
                == Self.generatedTokenCount)
    }
}

#endif  // FoundationModelsIntegration
