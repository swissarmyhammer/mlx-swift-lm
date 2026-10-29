// Copyright © 2026 Apple Inc.

import MLXLMCommon

extension MTPAdaptivePolicy {

    /// The plain steps of each probe of ``losingEveryRound``: the warm-up step and one timed step.
    private static let losingProbeLength = 2

    /// The rounds between two probes of ``losingEveryRound``, beyond any test decode's rounds, so
    /// the one probe a decode takes is the first.
    private static let losingProbeInterval = 64

    /// The plain tokens of the first pause of ``losingEveryRound``.
    private static let losingPauseLength = 3

    /// The plain tokens of the longest pause of ``losingEveryRound``.
    private static let losingMaximumPauseLength = 6

    /// A policy under which every judged round loses to plain decode, whatever the clock reads:
    /// each round is judged, a probe is a warm-up step and one timed step, and pauses last three
    /// to six tokens. A stream under it lowers its block size to the smallest one, pauses, and
    /// resumes speculation after each pause, at the same points on every run.
    public static let losingEveryRound = MTPAdaptivePolicy(
        minimumRoundSamples: 1, probeLength: losingProbeLength,
        probeInterval: losingProbeInterval, maximumProbeInterval: losingProbeInterval,
        stalePlainEstimateRounds: losingProbeInterval,
        stalePlainContextGrowth: .greatestFiniteMagnitude,
        requiredSpeedup: .greatestFiniteMagnitude, growthSpeedup: .greatestFiniteMagnitude,
        growthHoldRounds: 0, pauseLength: losingPauseLength,
        maximumPauseLength: losingMaximumPauseLength, smoothing: 1)
}
