// Copyright © 2026 Apple Inc.

import Foundation

/// How MTP speculation guards against decoding slower than plain decode.
///
/// A speculative round costs a drafter chain, a verify forward of several tokens, and host
/// synchronizations that a loaded CPU stretches; at long context the drafter's attention and the
/// multi-row verify grow too. When those costs outgrow what the round accepts, speculation
/// decodes slower than the target alone. The guard measures the tokens per second of its rounds
/// against short plain-decode probes taken at the current position. When rounds lose, it first
/// lowers the block size one step at a time, which shortens the drafter chain and the verify;
/// when the smallest block still loses, it pauses speculation and decodes plainly, then tries
/// speculation again. Output does not depend on the guard: greedy output is the output of the
/// iterator without it, up to the near-ties a multi-token verify can round apart, and sampled
/// output keeps the target distribution.
public struct MTPAdaptivePolicy: Sendable, Hashable {

    /// Whether the guard runs at all. A disabled policy speculates at the configured block size
    /// for the whole stream.
    public var isEnabled: Bool

    /// The rounds at one block size the guard measures before it judges that block size.
    public var minimumRoundSamples: Int

    /// The plain steps of one probe. The first step of each plain run refills the decode
    /// pipeline, so the estimate leaves it out; a probe takes at least two steps.
    public var probeLength: Int

    /// The rounds between two probes, at first. The interval doubles each time a probe finds
    /// speculation ahead by ``growthSpeedup``, up to ``maximumProbeInterval``.
    public var probeInterval: Int

    /// The most rounds between two probes.
    public var maximumProbeInterval: Int

    /// The rounds after which a plain estimate is too old to condemn a block size: rounds that
    /// lose against an older estimate probe plain decode again first.
    public var stalePlainEstimateRounds: Int

    /// The growth of the context, as a fraction of the context the plain estimate was measured
    /// at, after which the estimate no longer describes plain decode: plain decode slows as the
    /// context grows, so the guard probes it again at the current position.
    public var stalePlainContextGrowth: Double

    /// The speculative tokens per second, as a multiple of plain decode's, below which the guard
    /// lowers the block size, or pauses speculation at the smallest block size.
    public var requiredSpeedup: Double

    /// The speculative tokens per second, as a multiple of plain decode's, above which the guard
    /// grows a lowered block size back by one.
    public var growthSpeedup: Double

    /// The rounds after a lowering during which the guard does not grow the block size.
    public var growthHoldRounds: Int

    /// The plain tokens of the first pause. Each pause whose resume still loses doubles the next
    /// one, up to ``maximumPauseLength``; a resume that wins restores this length.
    public var pauseLength: Int

    /// The most plain tokens of one pause.
    public var maximumPauseLength: Int

    /// The weight of the newest sample in each moving estimate, above 0 and at most 1.
    public var smoothing: Double

    /// The guard with its tuned settings.
    public static let enabled = MTPAdaptivePolicy()

    /// A policy that speculates at the configured block size for the whole stream.
    public static let disabled = MTPAdaptivePolicy(isEnabled: false)

    /// The default policy: ``disabled``. Real-model runs showed the guard backing off while
    /// unguarded rounds still beat plain decode, so a stream opts in with ``enabled``.
    public static let `default` = disabled

    /// Creates a policy.
    ///
    /// - Parameters:
    ///   - isEnabled: Whether the guard runs.
    ///   - minimumRoundSamples: The rounds that judge one block size.
    ///   - probeLength: The plain steps of one probe, its warm-up step included.
    ///   - probeInterval: The rounds between two probes, at first.
    ///   - maximumProbeInterval: The most rounds between two probes.
    ///   - stalePlainEstimateRounds: The rounds after which a plain estimate is probed again
    ///     before it condemns a block size.
    ///   - stalePlainContextGrowth: The context growth, as a fraction, after which a plain
    ///     estimate is probed again.
    ///   - requiredSpeedup: The speedup over plain decode below which speculation backs off.
    ///   - growthSpeedup: The speedup over plain decode above which a lowered block size grows.
    ///   - growthHoldRounds: The rounds after a lowering without growth.
    ///   - pauseLength: The plain tokens of the first pause.
    ///   - maximumPauseLength: The most plain tokens of one pause.
    ///   - smoothing: The weight of the newest sample in each moving estimate.
    public init(
        isEnabled: Bool = true,
        minimumRoundSamples: Int = 16,
        probeLength: Int = 4,
        probeInterval: Int = 64,
        maximumProbeInterval: Int = 512,
        stalePlainEstimateRounds: Int = 32,
        stalePlainContextGrowth: Double = 0.1,
        requiredSpeedup: Double = 1,
        growthSpeedup: Double = 1.2,
        growthHoldRounds: Int = 128,
        pauseLength: Int = 64,
        maximumPauseLength: Int = 1024,
        smoothing: Double = 0.25
    ) {
        self.isEnabled = isEnabled
        self.minimumRoundSamples = minimumRoundSamples
        self.probeLength = probeLength
        self.probeInterval = probeInterval
        self.maximumProbeInterval = maximumProbeInterval
        self.stalePlainEstimateRounds = stalePlainEstimateRounds
        self.stalePlainContextGrowth = stalePlainContextGrowth
        self.requiredSpeedup = requiredSpeedup
        self.growthSpeedup = growthSpeedup
        self.growthHoldRounds = growthHoldRounds
        self.pauseLength = pauseLength
        self.maximumPauseLength = maximumPauseLength
        self.smoothing = smoothing
    }

    /// The fewest plain steps of a probe: the warm-up step and one timed step.
    private static let minimumProbeLength = 2

    /// Each requirement of a valid policy, with the words its refusal states.
    private var requirements: [(holds: Bool, statement: String)] {
        [
            (minimumRoundSamples >= 1, "minimumRoundSamples must be at least 1"),
            (
                probeLength >= Self.minimumProbeLength,
                "probeLength must be at least \(Self.minimumProbeLength)"
            ),
            (probeInterval >= 1, "probeInterval must be at least 1"),
            (
                maximumProbeInterval >= probeInterval,
                "maximumProbeInterval must be at least probeInterval"
            ),
            (stalePlainEstimateRounds >= 1, "stalePlainEstimateRounds must be at least 1"),
            (stalePlainContextGrowth >= 0, "stalePlainContextGrowth must not be negative"),
            (requiredSpeedup > 0, "requiredSpeedup must be above 0"),
            (growthSpeedup >= requiredSpeedup, "growthSpeedup must be at least requiredSpeedup"),
            (growthHoldRounds >= 0, "growthHoldRounds must not be negative"),
            (pauseLength >= 1, "pauseLength must be at least 1"),
            (maximumPauseLength >= pauseLength, "maximumPauseLength must be at least pauseLength"),
            (smoothing > 0 && smoothing <= 1, "smoothing must be above 0 and at most 1"),
        ]
    }

    /// Checks every setting against its range.
    ///
    /// - Throws: ``SpeculativeDecodingConfigurationError/invalidMTPAdaptivePolicy(_:)`` naming
    ///   the first setting outside its range.
    public func validate() throws {
        if let broken = requirements.first(where: { !$0.holds }) {
            throw SpeculativeDecodingConfigurationError.invalidMTPAdaptivePolicy(broken.statement)
        }
    }
}

/// The wall clock an adaptive guard times its units with.
package struct MTPAdaptiveClock: Sendable {

    /// The current time in seconds, from any fixed origin.
    package let now: @Sendable () -> TimeInterval

    /// The nanoseconds of one second.
    private static let nanosecondsPerSecond = 1e9

    /// - Parameter now: Reads the current time in seconds.
    package init(now: @escaping @Sendable () -> TimeInterval) {
        self.now = now
    }

    /// The monotonic system clock.
    package static let system = MTPAdaptiveClock {
        Double(DispatchTime.now().uptimeNanoseconds) / nanosecondsPerSecond
    }
}

/// A moving estimate of tokens per second: an exponentially weighted mean of the tokens and of
/// the seconds of each sample.
struct MTPThroughputEstimate: Sendable {

    /// The samples the estimate holds.
    private(set) var sampleCount = 0

    /// The weighted mean tokens of a sample.
    private var tokens = 0.0

    /// The weighted mean seconds of a sample.
    private var seconds = 0.0

    /// The tokens per second of the estimate, or 0 before its first sample.
    var tokensPerSecond: Double {
        seconds > 0 ? tokens / seconds : 0
    }

    /// Adds a sample. A sample without tokens or without measurable time is left out.
    ///
    /// - Parameters:
    ///   - sampleTokens: The tokens of the sample.
    ///   - sampleSeconds: The seconds of the sample.
    ///   - smoothing: The weight of the sample against the estimate.
    mutating func add(tokens sampleTokens: Int, seconds sampleSeconds: Double, smoothing: Double) {
        guard sampleTokens > 0, sampleSeconds > 0, sampleSeconds.isFinite else { return }
        let weight = sampleCount == 0 ? 1 : smoothing
        tokens += weight * (Double(sampleTokens) - tokens)
        seconds += weight * (sampleSeconds - seconds)
        sampleCount += 1
    }
}

/// The decisions of the adaptive guard, from the costs of the rounds and plain steps it measured.
///
/// The controller holds no clock: each round and each plain step reports its own tokens and
/// seconds, and the context it ran at, so the decisions follow from the reported costs alone.
package struct MTPAdaptiveController: Sendable {

    /// What the stream runs next.
    package enum Step: Equatable, Sendable {

        /// A speculative round of `blockSize` tokens: up to `blockSize - 1` drafts and the bonus.
        case speculate(blockSize: Int)

        /// One plain decode step.
        case decodePlainly
    }

    /// What the controller is doing.
    private enum Phase: Sendable {

        /// It speculates at ``MTPAdaptiveController/blockSize``.
        case speculating

        /// It decodes plainly to measure plain decode, for `remaining` more steps.
        case probing(remaining: Int)

        /// It decodes plainly because speculation lost, for `remaining` more steps.
        case paused(remaining: Int)
    }

    /// The smallest block size: one draft and the bonus token.
    package static let minimumBlockSize = MTPDraftBlockOutput.minimumChainedMTPBlockSize

    /// The fewest tokens of context growth that make a plain estimate stale, so a short context
    /// does not probe plain decode every few tokens.
    package static let minimumStaleContextGrowth = 4096

    /// The factor by which a winning probe stretches the probe interval, and a losing resume
    /// stretches the next pause.
    private static let backoffFactor = 2

    /// The policy of the controller.
    package let policy: MTPAdaptivePolicy

    /// The configured block size, the largest the controller speculates at.
    package let maximumBlockSize: Int

    /// The block size of the next round.
    package private(set) var blockSize: Int

    /// The pauses of speculation so far.
    package private(set) var pauseCount = 0

    /// The probes of plain decode so far.
    package private(set) var probeCount = 0

    /// The changes of the block size so far.
    package private(set) var blockSizeChangeCount = 0

    private var phase = Phase.speculating
    private var speculative = MTPThroughputEstimate()
    private var plain = MTPThroughputEstimate()

    /// The rounds measured at the current block size.
    private var roundsAtBlockSize = 0

    /// The rounds since the last plain step.
    private var roundsSincePlainSample = 0

    /// The context of the last round, or 0 when the stream does not report it.
    private var contextLength = 0

    /// The context of the last plain step, or 0 when the stream does not report it.
    private var plainContextLength = 0

    /// The rounds between two probes now.
    private var probeInterval: Int

    /// The plain tokens of the next pause.
    private var nextPauseLength: Int

    /// The rounds left before the block size may grow.
    private var growthHeldRounds = 0

    /// Whether the next plain step is the warm-up step of a plain run.
    private var awaitsWarmupStep = false

    /// Whether the rounds since the last pause have not yet been judged.
    private var resumedFromPause = false

    /// Whether the rounds since the last probe have not yet been judged.
    private var awaitsProbeVerdict = false

    /// - Parameters:
    ///   - policy: The policy of the controller.
    ///   - blockSize: The configured block size, at which the stream starts.
    package init(policy: MTPAdaptivePolicy, blockSize: Int) {
        self.policy = policy
        self.maximumBlockSize = blockSize
        self.blockSize = blockSize
        self.probeInterval = policy.probeInterval
        self.nextPauseLength = policy.pauseLength
    }

    /// What the stream runs next.
    package var nextStep: Step {
        switch phase {
        case .speculating: .speculate(blockSize: blockSize)
        case .probing, .paused: .decodePlainly
        }
    }

    /// The plain steps left in the current probe or pause, or 0 while the stream speculates.
    package var plainStepsRemaining: Int {
        switch phase {
        case .speculating: 0
        case .probing(let remaining), .paused(let remaining): remaining
        }
    }

    /// Records one speculative round and judges the block size.
    ///
    /// - Parameters:
    ///   - tokens: The tokens the round emitted.
    ///   - seconds: The wall time of the round, until the next unit began.
    ///   - contextLength: The tokens the target cache held when the round began, or 0 when
    ///     the stream does not report them.
    package mutating func recordRound(tokens: Int, seconds: Double, contextLength: Int = 0) {
        speculative.add(tokens: tokens, seconds: seconds, smoothing: policy.smoothing)
        roundsAtBlockSize += 1
        roundsSincePlainSample += 1
        growthHeldRounds = Swift.max(0, growthHeldRounds - 1)
        self.contextLength = contextLength
        judgeBlockSize()
    }

    /// Records one plain step of a probe or a pause.
    ///
    /// - Parameters:
    ///   - tokens: The tokens the step emitted.
    ///   - seconds: The wall time of the step, until the next unit began.
    ///   - contextLength: The tokens the target cache held when the step began, or 0 when the
    ///     stream does not report them.
    package mutating func recordPlainStep(tokens: Int, seconds: Double, contextLength: Int = 0) {
        if awaitsWarmupStep {
            awaitsWarmupStep = false
        } else {
            plain.add(tokens: tokens, seconds: seconds, smoothing: policy.smoothing)
        }
        plainContextLength = contextLength
        switch phase {
        case .speculating:
            return
        case .probing(let remaining) where remaining > 1:
            phase = .probing(remaining: remaining - 1)
        case .paused(let remaining) where remaining > 1:
            phase = .paused(remaining: remaining - 1)
        case .probing, .paused:
            finishPlainRun()
        }
    }

    /// Whether the plain estimate describes plain decode now: it holds a probe's samples, its
    /// probe is not yet due, and the context has not outgrown the context it was measured at.
    private var plainEstimateIsCurrent: Bool {
        let growth = contextLength - plainContextLength
        let contextOutgrewEstimate =
            plainContextLength > 0 && growth >= Self.minimumStaleContextGrowth
            && Double(growth) > Double(plainContextLength) * policy.stalePlainContextGrowth
        return plain.sampleCount >= policy.probeLength - 1
            && roundsSincePlainSample < probeInterval && !contextOutgrewEstimate
    }

    /// Judges the block size once it has enough rounds: keeps it, lowers it, grows it, pauses
    /// speculation, or first probes plain decode when the plain estimate is missing or due.
    private mutating func judgeBlockSize() {
        guard roundsAtBlockSize >= policy.minimumRoundSamples else { return }
        guard plainEstimateIsCurrent else { return startProbe() }
        let speedup = speculative.tokensPerSecond / plain.tokensPerSecond
        if speedup < policy.requiredSpeedup {
            backOff()
        } else {
            keepSpeculating(speedup: speedup)
        }
    }

    /// Lowers the block size of rounds that lose to plain decode, or pauses speculation at the
    /// smallest block size; an old plain estimate is probed again first.
    private mutating func backOff() {
        guard roundsSincePlainSample <= policy.stalePlainEstimateRounds else {
            return startProbe()
        }
        guard blockSize > Self.minimumBlockSize else { return startPause() }
        changeBlockSize(by: -1)
        growthHeldRounds = policy.growthHoldRounds
    }

    /// Keeps speculating after rounds that win: restores the first pause's length after a
    /// resume, sets the probe interval after a probe, and grows a lowered block size.
    private mutating func keepSpeculating(speedup: Double) {
        if resumedFromPause {
            resumedFromPause = false
            nextPauseLength = policy.pauseLength
        }
        let leadsByMargin = speedup >= policy.growthSpeedup
        if awaitsProbeVerdict {
            awaitsProbeVerdict = false
            probeInterval =
                leadsByMargin
                ? Swift.min(
                    Self.backoffFactor * probeInterval, policy.maximumProbeInterval)
                : policy.probeInterval
        }
        guard leadsByMargin, blockSize < maximumBlockSize, growthHeldRounds == 0 else { return }
        changeBlockSize(by: 1)
    }

    /// Moves the block size by `delta` and starts measuring the new one afresh.
    private mutating func changeBlockSize(by delta: Int) {
        blockSize += delta
        blockSizeChangeCount += 1
        restartRoundMeasurement()
    }

    /// Forgets the measured rounds.
    private mutating func restartRoundMeasurement() {
        speculative = MTPThroughputEstimate()
        roundsAtBlockSize = 0
    }

    /// Starts a probe of plain decode.
    private mutating func startProbe() {
        phase = .probing(remaining: policy.probeLength)
        probeCount += 1
        awaitsWarmupStep = true
        awaitsProbeVerdict = true
    }

    /// Pauses speculation for the next pause's length, and doubles the pause after it.
    private mutating func startPause() {
        phase = .paused(remaining: nextPauseLength)
        pauseCount += 1
        nextPauseLength = Swift.min(
            Self.backoffFactor * nextPauseLength, policy.maximumPauseLength)
        awaitsWarmupStep = true
        resumedFromPause = true
    }

    /// Returns to speculation after a probe or a pause. Rounds after a pause are measured
    /// afresh, since the load that paused speculation may have changed.
    private mutating func finishPlainRun() {
        if case .paused = phase {
            restartRoundMeasurement()
        }
        phase = .speculating
        roundsSincePlainSample = 0
    }
}

/// An adaptive controller with the clock that times each unit of the stream: each speculative
/// round and each plain step lasts from its start to the start of the next unit, so a unit's
/// time includes the consumer's handling of the tokens it emitted.
package struct MTPAdaptiveGuard: Sendable {

    /// A unit that has started and not yet ended.
    private struct OpenUnit: Sendable {

        /// What the unit runs.
        let step: MTPAdaptiveController.Step

        /// The time the unit started.
        let start: TimeInterval

        /// The tokens the target cache held when the unit started, or 0 when unknown.
        let contextLength: Int

        /// The tokens the unit emitted so far.
        var tokens = 0
    }

    /// The decisions of the guard.
    package private(set) var controller: MTPAdaptiveController

    /// The clock of the guard.
    private let clock: MTPAdaptiveClock

    /// The unit that runs now, or `nil` before the first unit.
    private var openUnit: OpenUnit?

    /// Makes a guard, or `nil` when `policy` is disabled.
    ///
    /// - Parameters:
    ///   - policy: The policy of the guard.
    ///   - blockSize: The configured block size.
    ///   - clock: The clock that times each unit.
    package init?(
        policy: MTPAdaptivePolicy, blockSize: Int, clock: MTPAdaptiveClock = .system
    ) {
        guard policy.isEnabled else { return nil }
        self.controller = MTPAdaptiveController(policy: policy, blockSize: blockSize)
        self.clock = clock
    }

    /// Ends the unit that runs now, reports it to the controller, and starts the next one.
    ///
    /// - Parameter contextLength: The tokens the target cache holds now, or 0 when unknown.
    /// - Returns: What the next unit runs.
    package mutating func beginUnit(contextLength: Int = 0) -> MTPAdaptiveController.Step {
        let now = clock.now()
        if let unit = openUnit {
            record(unit, seconds: now - unit.start)
        }
        let step = controller.nextStep
        openUnit = OpenUnit(step: step, start: now, contextLength: contextLength)
        return step
    }

    /// Adds `tokens` emitted tokens to the unit that runs now.
    package mutating func recordEmitted(_ tokens: Int) {
        openUnit?.tokens += tokens
    }

    /// Reports an ended unit to the controller.
    private mutating func record(_ unit: OpenUnit, seconds: Double) {
        switch unit.step {
        case .speculate:
            controller.recordRound(
                tokens: unit.tokens, seconds: seconds, contextLength: unit.contextLength)
        case .decodePlainly:
            controller.recordPlainStep(
                tokens: unit.tokens, seconds: seconds, contextLength: unit.contextLength)
        }
    }
}
