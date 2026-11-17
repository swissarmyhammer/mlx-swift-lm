// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels

/// A receiver holds the one value a generation hands over when it ends, and the pass takes it
/// once, when it commits its cache.
@Suite("A hand-off receiver gives up the value it received")
struct ExecutorHandOffReceiverTests {

    /// The value each test hands over.
    private static let handedOver = 7

    @Test
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aTakeGivesTheValueTheGenerationHandedOver() {
        let receiver = ExecutorHandOffReceiver<Int>()
        receiver.receive(Self.handedOver)

        #expect(receiver.take() == Self.handedOver)
    }

    @Test
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aSecondTakeGivesNothing() {
        let receiver = ExecutorHandOffReceiver<Int>()
        receiver.receive(Self.handedOver)
        _ = receiver.take()

        #expect(receiver.take() == nil)
    }

    @Test
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aTakeBeforeAHandOffGivesNothing() {
        #expect(ExecutorHandOffReceiver<Int>().take() == nil)
    }
}

#endif
