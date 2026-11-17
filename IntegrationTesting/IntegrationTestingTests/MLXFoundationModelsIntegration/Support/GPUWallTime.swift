// Copyright © 2026 Apple Inc.
//
// The wall time of GPU work, as the MTP cost benchmarks measure it: a GPU sync before and after
// the work, and the median of a few runs.

import Foundation
import MLX

/// The wall time of GPU work, in milliseconds.
enum GPUWallTime {

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1_000.0

    /// The wall milliseconds of `body`, with a GPU sync before and after it.
    ///
    /// - Parameter body: The work; it evaluates what it times.
    /// - Returns: The milliseconds from the first sync to the end of the second.
    static func milliseconds<Result>(_ body: () -> Result) -> Double {
        measure(body).milliseconds
    }

    /// The value of `body` and its wall milliseconds, with a GPU sync before and after it.
    ///
    /// - Parameter body: The work; it evaluates what it times.
    /// - Returns: The value of `body`, and the milliseconds from the first sync to the end of
    ///   the second.
    static func measure<Result>(
        _ body: () throws -> Result
    ) rethrows -> (value: Result, milliseconds: Double) {
        Stream.gpu.synchronize()
        let start = Date()
        let value = try body()
        Stream.gpu.synchronize()
        return (value, Date().timeIntervalSince(start) * millisecondsPerSecond)
    }

    /// The rate per second of `count` items done in `milliseconds`.
    ///
    /// - Parameters:
    ///   - count: The items done, for example tokens.
    ///   - milliseconds: The wall milliseconds the items took, more than 0.
    /// - Returns: The items per second.
    static func ratePerSecond(_ count: Int, milliseconds: Double) -> Double {
        Double(count) * millisecondsPerSecond / milliseconds
    }

    /// The median of `repeats` runs of `sample`.
    ///
    /// - Parameters:
    ///   - repeats: The runs, 1 or more.
    ///   - sample: One run, which gives its milliseconds.
    /// - Returns: The middle run of the sorted runs.
    static func median(repeats: Int, _ sample: () -> Double) -> Double {
        SampleMedian.of((0 ..< repeats).map { _ in sample() })
    }
}
