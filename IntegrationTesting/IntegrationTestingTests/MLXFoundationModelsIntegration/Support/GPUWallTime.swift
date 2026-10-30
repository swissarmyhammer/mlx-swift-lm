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

    /// The divisor that picks the middle run of the sorted runs.
    private static let medianDivisor = 2

    /// The wall milliseconds of `body`, with a GPU sync before and after it.
    ///
    /// - Parameter body: The work; it evaluates what it times.
    /// - Returns: The milliseconds from the first sync to the end of the second.
    static func milliseconds<Result>(_ body: () -> Result) -> Double {
        Stream.gpu.synchronize()
        let start = Date()
        _ = body()
        Stream.gpu.synchronize()
        return Date().timeIntervalSince(start) * millisecondsPerSecond
    }

    /// The median of `repeats` runs of `sample`.
    ///
    /// - Parameters:
    ///   - repeats: The runs, 1 or more.
    ///   - sample: One run, which gives its milliseconds.
    /// - Returns: The middle run of the sorted runs.
    static func median(repeats: Int, _ sample: () -> Double) -> Double {
        let samples = (0 ..< repeats).map { _ in sample() }.sorted()
        return samples[samples.count / medianDivisor]
    }
}
