// Copyright © 2026 Apple Inc.

import MLXGuidedGeneration
import Testing

@Suite
struct RepetitionCycleTrackerTests {

    @Test("A period-1 run is detected once it reaches the minimum run length")
    func periodOneRunDetectedAtMinimumRunLength() {
        var tracker = RepetitionCycleTracker()

        for _ in 0 ..< 15 {
            tracker.record(tokenID: 7)
        }
        #expect(!tracker.isActive, "15 repeats are below the 16-token minimum run")

        tracker.record(tokenID: 7)
        #expect(tracker.isActive)
        #expect(tracker.suppressedTokenIDs == [7])
    }

    @Test("A period-2 alternation is detected and suppresses both cycle tokens")
    func periodTwoAlternationDetected() {
        var tracker = RepetitionCycleTracker()

        // The `}7}7}7...` live signature: alternate two ids for 16 tokens.
        for _ in 0 ..< 8 {
            tracker.record(tokenID: 125)
            tracker.record(tokenID: 55)
        }
        #expect(tracker.isActive)
        #expect(tracker.suppressedTokenIDs == [125, 55])
    }

    @Test("A period-10 digit-group cycle is detected after three full cycles")
    func periodTenCycleDetectedAfterThreeCycles() {
        var tracker = RepetitionCycleTracker()

        // The `1234567890...` live signature in token space: BPE digit
        // groups repeating with period 10. Three cycles = 30 tokens.
        let cycle = Array(100 ..< 110)
        for _ in 0 ..< 2 {
            for id in cycle {
                tracker.record(tokenID: id)
            }
        }
        #expect(!tracker.isActive, "two cycles are not yet three")

        for id in cycle {
            tracker.record(tokenID: id)
        }
        #expect(tracker.isActive)
        #expect(tracker.suppressedTokenIDs == Set(cycle))
    }

    @Test("Diverse output never activates suppression")
    func diverseStreamNeverActivates() {
        var tracker = RepetitionCycleTracker()

        for id in 0 ..< 200 {
            tracker.record(tokenID: id)
        }
        #expect(!tracker.isActive)
        #expect(tracker.suppressedTokenIDs.isEmpty)
    }

    @Test("Suppression latches: later diverse tokens never clear it")
    func suppressionLatches() {
        var tracker = RepetitionCycleTracker()

        for _ in 0 ..< 16 {
            tracker.record(tokenID: 7)
        }
        for id in 1000 ..< 1100 {
            tracker.record(tokenID: id)
        }
        #expect(tracker.isActive)
        #expect(tracker.suppressedTokenIDs == [7])
    }

    @Test("Distinct cycles accumulate into one suppression set")
    func distinctCyclesAccumulate() {
        var tracker = RepetitionCycleTracker()

        for _ in 0 ..< 16 {
            tracker.record(tokenID: 7)
        }
        for _ in 0 ..< 8 {
            tracker.record(tokenID: 8)
            tracker.record(tokenID: 9)
        }
        #expect(tracker.suppressedTokenIDs == [7, 8, 9])
    }

    @Test("interrupt() clears the run window so split runs don't join")
    func interruptClearsWindow() {
        var tracker = RepetitionCycleTracker()

        for _ in 0 ..< 8 {
            tracker.record(tokenID: 7)
        }
        tracker.interrupt()
        for _ in 0 ..< 8 {
            tracker.record(tokenID: 7)
        }
        #expect(!tracker.isActive, "8 + 8 across an interrupt is never a 16-token run")
    }

    @Test("record reports true exactly when a new cycle is detected")
    func recordReportsDetection() {
        var tracker = RepetitionCycleTracker()

        for _ in 0 ..< 15 {
            let detected = tracker.record(tokenID: 7)
            #expect(!detected)
        }
        let detected = tracker.record(tokenID: 7)
        #expect(detected)
    }
}
