// Copyright © 2026 Apple Inc.

import MLXLMCommon

extension MTPAdaptivePolicy {

    /// A policy under which every judged round loses to plain decode, whatever the clock reads:
    /// each round is judged, a probe is a warm-up step and one timed step, and pauses last three
    /// to six tokens. A stream under it lowers its block size to the smallest one, pauses, and
    /// resumes speculation after each pause, at the same points on every run.
    public static let losingEveryRound = MTPAdaptivePolicy(
        minimumRoundSamples: 1, probeLength: 2, probeInterval: 64, maximumProbeInterval: 64,
        stalePlainEstimateRounds: 64, stalePlainContextGrowth: .greatestFiniteMagnitude,
        requiredSpeedup: .greatestFiniteMagnitude,
        growthSpeedup: .greatestFiniteMagnitude, growthHoldRounds: 0, pauseLength: 3,
        maximumPauseLength: 6, smoothing: 1)
}
