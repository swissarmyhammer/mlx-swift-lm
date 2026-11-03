// Copyright © 2026 Apple Inc.
//
// The median of the timed samples of a benchmark, shared by every benchmark of this target so
// each reports its middle sample the same way.

/// The median of the samples of a benchmark.
enum SampleMedian {

    /// The divisor that picks the middle position of the sorted samples.
    private static let middleDivisor = 2

    /// The middle sample of the sorted `samples`: for an even count, the later of the two middle
    /// samples, as ``BenchmarkStats`` takes it.
    ///
    /// - Parameter samples: The samples, 1 or more.
    /// - Returns: The middle sample.
    static func of(_ samples: [Double]) -> Double {
        precondition(!samples.isEmpty, "a median needs at least one sample")
        let sorted = samples.sorted()
        return sorted[sorted.count / middleDivisor]
    }
}
