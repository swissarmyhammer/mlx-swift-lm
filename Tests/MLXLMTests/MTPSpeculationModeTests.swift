// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

/// The reason of a stream exists only in a mode that has one.
struct MTPSpeculationModeTests {

    @Test func aSpeculatingStreamHasNoReason() {
        #expect(MTPSpeculationMode.speculating.reason == nil)
    }

    @Test func passthroughKeepsItsReason() {
        #expect(MTPSpeculationMode.passthrough(reason: "stood down").reason == "stood down")
    }

    @Test func aDiscardedContinuationKeepsItsReason() {
        #expect(MTPSpeculationMode.continuationDiscarded(reason: "lost").reason == "lost")
    }

    @Test func onlyPassthroughDecodesPlainly() {
        #expect(MTPSpeculationMode.passthrough(reason: "stood down").decodesPlainly)
    }

    @Test func aDiscardedContinuationDoesNotDecodePlainly() {
        #expect(!MTPSpeculationMode.continuationDiscarded(reason: "lost").decodesPlainly)
    }
}

/// How a handler's disposition ends the generation loop.
struct TokenLoopDispositionStopReasonTests {

    @Test func moreRecordsNoReason() {
        #expect(TokenLoopDisposition.more.stopReason == nil)
    }

    @Test func stopRecordsStop() {
        #expect(TokenLoopDisposition.stop.stopReason == .stop)
    }

    @Test func cancelledRecordsCancelled() {
        #expect(TokenLoopDisposition.cancelled.stopReason == .cancelled)
    }
}

/// A reported count stays inside its total.
struct ClampedCountTests {

    @Test func aNegativeCountBecomesZero() {
        #expect(clampedCount(-3, to: 5) == 0)
    }

    @Test func aCountPastTheLimitBecomesTheLimit() {
        #expect(clampedCount(9, to: 5) == 5)
    }
}
