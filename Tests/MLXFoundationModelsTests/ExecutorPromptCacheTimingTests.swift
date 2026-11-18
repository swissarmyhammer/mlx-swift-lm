// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels

@Suite("ExecutorPromptCacheTiming")
struct ExecutorPromptCacheTimingTests {

    /// The value the measured work returns.
    private static let measuredValue = 42

    @Test("measure returns the value of the work it timed")
    func measureReturnsTheValueOfTheWork() {
        let measured = ExecutorPromptCacheTiming.measure { Self.measuredValue }

        #expect(measured.value == Self.measuredValue)
    }

    @Test("measure rethrows the error of the work it timed")
    func measureRethrowsTheErrorOfTheWork() {
        #expect(throws: CocoaError(.fileReadUnknown)) {
            try ExecutorPromptCacheTiming.measure { () throws -> Int in
                throw CocoaError(.fileReadUnknown)
            }
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
