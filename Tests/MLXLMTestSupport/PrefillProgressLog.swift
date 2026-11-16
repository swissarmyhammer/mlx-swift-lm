// Copyright © 2026 Apple Inc.

import os

/// Records each progress report of a prefill. The lock serializes every read
/// and write of the counts, so the `@Sendable` progress closure of
/// `PrefillParameters` can record from any thread.
public struct PrefillProgressLog: Sendable {
    private let reports = OSAllocatedUnfairLock<[Int]>(initialState: [])

    /// Makes an empty log.
    public init() {}

    /// The `processed` count of each report, in order.
    public var processed: [Int] { reports.withLock { $0 } }

    /// Records the `processed` count of one report.
    ///
    /// - Parameter processed: The positions the prefill has processed so far.
    public func record(_ processed: Int) {
        reports.withLock { $0.append(processed) }
    }
}
