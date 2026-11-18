// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Tests for ``ExecutorPrefillRecorder``, which names whether the prefill of a pass ended and,
/// when it ended, the model state it left.
///
/// No weights are needed. The slot of each recorder holds no plan, thus the end of the prefill
/// takes no checkpoint.
@Suite("A prefill recorder names whether the prefill ended")
struct ExecutorPrefillRecorderTests {

    /// A recorder over a slot that holds nothing.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeRecorder() -> ExecutorPrefillRecorder {
        ExecutorPrefillRecorder(promptCache: ExecutorPromptCacheSlot(nil, report: { _ in }))
    }

    @Test("a recorder whose prefill did not end holds a pending outcome")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRecorderWhosePrefillDidNotEndIsPending() {
        let recorder = makeRecorder()

        let isPending = if case .pending = recorder.outcome { true } else { false }

        #expect(isPending)
    }

    @Test("a recorder whose prefill ended with no model state holds that end")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRecorderWhosePrefillEndedWithNoStateHoldsThatEnd() {
        let recorder = makeRecorder()
        recorder.record(state: nil, drafterState: { nil })

        let endedWithNoState = if case .ended(.none) = recorder.outcome { true } else { false }

        #expect(endedWithNoState)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
