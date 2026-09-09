// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The output count a pass reports must name every token the model wrote
/// and the caches hold, the stop token included.
///
/// `GenerateCompletionInfo.generationTokenCount` leaves the stop token out
/// when the loop suppresses it, and names it in `stopTokenFedToCache`. The
/// ledger of the prompt cache holds that token, thus a consumer that adds the
/// prompt count and the output count of a turn must reach the tokens the next
/// turn can reuse. `FoundationModelsRouter` reads the two counts that way.
@Suite("The output count of a pass")
struct ExecutorUsageOutputTests {

    /// Tokens the fixture generated before the stop token.
    private static let generatedTokenCount = 27

    /// The stop token the fixture fed into the caches.
    private static let stopToken = 248_046

    /// A completion report with `generatedTokenCount` tokens and the stop
    /// token `stopToken`, or no stop token when it is nil.
    private func info(stopToken: Int?) -> GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: 1, generationTokenCount: Self.generatedTokenCount,
            promptTime: 0, generationTime: 0, stopTokenFedToCache: stopToken)
    }

    @Test("counts the stop token the caches hold")
    func countsTheStopTokenTheCachesHold() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.generatedTokenCount(of: info(stopToken: Self.stopToken))
                == Self.generatedTokenCount + 1)
    }

    @Test("counts no stop token when the pass fed none")
    func countsNoStopTokenWhenThePassFedNone() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            MLXLanguageModel.Executor.generatedTokenCount(of: info(stopToken: nil))
                == Self.generatedTokenCount)
    }
}

#endif  // FoundationModelsIntegration
