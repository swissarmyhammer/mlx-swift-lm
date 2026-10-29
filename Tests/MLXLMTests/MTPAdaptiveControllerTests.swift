// Copyright © 2026 Apple Inc.

import Foundation
import Testing
import os

@testable import MLXLMCommon

/// The adaptive guard of MTP speculation, driven by measured round and plain-step costs: it
/// keeps speculating while rounds out-decode plain decode, lowers the block size when they do
/// not, pauses speculation when even the smallest block loses, and comes back after the pause.
@Suite
struct MTPAdaptiveControllerTests {

    /// The configured block size of each controller: the seed, three chained drafts and the bonus.
    private static let blockSize = 5

    /// A policy small enough that each decision takes a few samples: two rounds judge a block
    /// size, a probe is a warm-up step and one timed step, and the newest sample is the estimate.
    private static let policy = MTPAdaptivePolicy(
        minimumRoundSamples: 2, probeLength: 2, probeInterval: 16, maximumProbeInterval: 64,
        stalePlainEstimateRounds: 8, requiredSpeedup: 1, growthSpeedup: 1.2,
        growthHoldRounds: 4, pauseLength: 3, maximumPauseLength: 6, smoothing: 1)

    /// The seconds of one timed unit.
    private static let unitSeconds = 0.1

    /// Plain decode's tokens per unit: ten tokens a second.
    private static let plainTokens = 1

    /// The tokens of a round that out-decodes plain decode by more than the growth margin.
    private static let fastRoundTokens = 3

    /// The tokens of a round that out-decodes plain decode by less than the growth margin.
    private static let winningRoundTokens = 2

    /// The seconds of a round that emits one token at half plain decode's rate.
    private static let slowRoundSeconds = 0.2

    /// Records `count` rounds of `tokens` tokens in `seconds` each.
    private func recordRounds(
        _ count: Int, tokens: Int, seconds: Double = unitSeconds,
        into controller: inout MTPAdaptiveController
    ) {
        for _ in 0 ..< count {
            controller.recordRound(tokens: tokens, seconds: seconds)
        }
    }

    /// Records one round that emits one token at half plain decode's rate.
    private func recordSlowRound(into controller: inout MTPAdaptiveController) {
        controller.recordRound(tokens: Self.plainTokens, seconds: Self.slowRoundSeconds)
    }

    /// Records the plain steps the controller asks for, at plain decode's rate.
    private func recordPlainRun(into controller: inout MTPAdaptiveController) {
        while controller.nextStep == .decodePlainly {
            controller.recordPlainStep(tokens: Self.plainTokens, seconds: Self.unitSeconds)
        }
    }

    /// Records slow rounds, and the plain steps of each probe they bring, until the controller
    /// pauses speculation.
    private func recordSlowRoundsUntilPause(into controller: inout MTPAdaptiveController) {
        let pauses = controller.pauseCount
        while controller.pauseCount == pauses {
            if controller.nextStep == .decodePlainly {
                recordPlainRun(into: &controller)
            } else {
                recordSlowRound(into: &controller)
            }
        }
    }

    /// A controller that has judged its first block size against one probe of plain decode.
    private func probedController() -> MTPAdaptiveController {
        var controller = MTPAdaptiveController(policy: Self.policy, blockSize: Self.blockSize)
        recordRounds(
            Self.policy.minimumRoundSamples, tokens: Self.fastRoundTokens, into: &controller)
        recordPlainRun(into: &controller)
        return controller
    }

    @Test("a new controller speculates at its block size")
    func newControllerSpeculates() {
        let controller = MTPAdaptiveController(policy: Self.policy, blockSize: Self.blockSize)

        #expect(controller.nextStep == .speculate(blockSize: Self.blockSize))
    }

    @Test("the first judged rounds probe plain decode for the probe's length")
    func firstJudgedRoundsProbePlainDecode() {
        var controller = MTPAdaptiveController(policy: Self.policy, blockSize: Self.blockSize)

        recordRounds(
            Self.policy.minimumRoundSamples, tokens: Self.fastRoundTokens, into: &controller)

        #expect(controller.nextStep == .decodePlainly)
        #expect(controller.plainStepsRemaining == Self.policy.probeLength)
        #expect(controller.probeCount == 1)
    }

    @Test("rounds faster than plain decode keep the block size")
    func fastRoundsKeepTheBlockSize() {
        var controller = probedController()

        recordRounds(
            Self.policy.minimumRoundSamples, tokens: Self.fastRoundTokens, into: &controller)

        #expect(controller.nextStep == .speculate(blockSize: Self.blockSize))
        #expect(controller.blockSizeChangeCount == 0)
    }

    @Test("a round slower than plain decode lowers the block size by one")
    func slowRoundLowersTheBlockSize() {
        var controller = probedController()

        recordSlowRound(into: &controller)

        #expect(controller.nextStep == .speculate(blockSize: Self.blockSize - 1))
        #expect(controller.blockSizeChangeCount == 1)
    }

    @Test("slow rounds lower the block size to the smallest one before they pause speculation")
    func slowRoundsPauseAtTheSmallestBlock() {
        var controller = probedController()

        recordSlowRoundsUntilPause(into: &controller)

        #expect(controller.blockSize == MTPAdaptiveController.minimumBlockSize)
        #expect(controller.blockSizeChangeCount == Self.blockSize - 2)
        #expect(controller.nextStep == .decodePlainly)
        #expect(controller.plainStepsRemaining == Self.policy.pauseLength)
        #expect(controller.pauseCount == 1)
    }

    @Test("a pause resumes speculation at the block size it paused at")
    func pauseResumesSpeculation() {
        var controller = probedController()
        recordSlowRoundsUntilPause(into: &controller)

        recordPlainRun(into: &controller)

        #expect(
            controller.nextStep
                == .speculate(blockSize: MTPAdaptiveController.minimumBlockSize))
    }

    @Test("a resume that still loses doubles the next pause, up to the longest pause")
    func losingResumeDoublesThePause() {
        var controller = probedController()
        var pauses: [Int] = []

        for _ in 0 ..< 3 {
            recordSlowRoundsUntilPause(into: &controller)
            pauses.append(controller.plainStepsRemaining)
            recordPlainRun(into: &controller)
        }

        #expect(
            pauses == [
                Self.policy.pauseLength, Self.policy.maximumPauseLength,
                Self.policy.maximumPauseLength,
            ])
        #expect(controller.pauseCount == 3)
    }

    @Test("a resume that wins restores the first pause's length")
    func winningResumeRestoresThePause() {
        var controller = probedController()
        recordSlowRoundsUntilPause(into: &controller)
        recordPlainRun(into: &controller)
        recordRounds(
            Self.policy.minimumRoundSamples, tokens: Self.winningRoundTokens, into: &controller)

        recordSlowRoundsUntilPause(into: &controller)

        #expect(controller.plainStepsRemaining == Self.policy.pauseLength)
    }

    @Test("fast rounds grow a lowered block size back once the hold has passed")
    func fastRoundsGrowTheBlockSizeBack() {
        var controller = probedController()
        recordSlowRound(into: &controller)

        recordRounds(
            Self.policy.growthHoldRounds - 1, tokens: Self.fastRoundTokens, into: &controller)
        let heldBlockSize = controller.blockSize
        recordRounds(1, tokens: Self.fastRoundTokens, into: &controller)

        #expect(heldBlockSize == Self.blockSize - 1)
        #expect(controller.blockSize == Self.blockSize)
    }

    @Test("a slow verdict against a stale plain estimate probes plain decode first")
    func staleEstimateIsProbedFirst() {
        var controller = probedController()
        recordRounds(
            Self.policy.stalePlainEstimateRounds, tokens: Self.fastRoundTokens,
            into: &controller)
        #expect(controller.nextStep == .speculate(blockSize: Self.blockSize))

        recordSlowRound(into: &controller)

        #expect(controller.nextStep == .decodePlainly)
        #expect(controller.blockSize == Self.blockSize)
        #expect(controller.probeCount == 2)
    }

    @Test("a context that outgrew its plain estimate probes plain decode again at once")
    func grownContextIsProbedAgain() {
        var controller = MTPAdaptiveController(policy: Self.policy, blockSize: Self.blockSize)
        let probedContext = 100_000
        recordRounds(
            Self.policy.minimumRoundSamples, tokens: Self.fastRoundTokens, into: &controller)
        while controller.nextStep == .decodePlainly {
            controller.recordPlainStep(
                tokens: Self.plainTokens, seconds: Self.unitSeconds, contextLength: probedContext)
        }
        let grownContext =
            Int(
                Double(probedContext) * (1 + Self.policy.stalePlainContextGrowth)) + 1

        controller.recordRound(
            tokens: Self.fastRoundTokens, seconds: Self.unitSeconds, contextLength: grownContext)

        #expect(controller.nextStep == .decodePlainly)
        #expect(controller.probeCount == 2)
    }

    @Test("rounds that keep winning probe plain decode less often, up to the longest interval")
    func winningRoundsProbeLessOften() {
        var controller = probedController()
        var roundsBetweenProbes: [Int] = []

        for _ in 0 ..< 3 {
            var rounds = 0
            while controller.nextStep != .decodePlainly {
                controller.recordRound(tokens: Self.fastRoundTokens, seconds: Self.unitSeconds)
                rounds += 1
            }
            roundsBetweenProbes.append(rounds)
            recordPlainRun(into: &controller)
        }

        #expect(
            roundsBetweenProbes == [
                2 * Self.policy.probeInterval, Self.policy.maximumProbeInterval,
                Self.policy.maximumProbeInterval,
            ])
    }

    @Test("the first plain step of each run is a warm-up the estimate leaves out")
    func warmupStepIsLeftOut() {
        var policy = Self.policy
        policy.smoothing = 0.5
        var controller = MTPAdaptiveController(policy: policy, blockSize: Self.blockSize)
        recordRounds(
            policy.minimumRoundSamples, tokens: Self.fastRoundTokens,
            seconds: Self.slowRoundSeconds, into: &controller)
        controller.recordPlainStep(tokens: Self.plainTokens, seconds: Self.unitSeconds / 100)
        controller.recordPlainStep(tokens: Self.plainTokens, seconds: Self.unitSeconds)

        recordRounds(
            1, tokens: Self.fastRoundTokens, seconds: Self.slowRoundSeconds, into: &controller)

        #expect(controller.blockSizeChangeCount == 0)
    }

    @Test("a guard times each unit with its clock and reports it to its controller")
    func guardTimesEachUnit() throws {
        let clock = ScriptedClock(step: Self.unitSeconds)
        var adaptiveGuard = try #require(
            MTPAdaptiveGuard(
                policy: Self.policy, blockSize: Self.blockSize, clock: clock.adaptiveClock))

        for _ in 0 ..< Self.policy.minimumRoundSamples {
            #expect(adaptiveGuard.beginUnit() == .speculate(blockSize: Self.blockSize))
            adaptiveGuard.recordEmitted(Self.fastRoundTokens)
        }

        #expect(adaptiveGuard.beginUnit() == .decodePlainly)
        #expect(adaptiveGuard.controller.probeCount == 1)
    }

    @Test("a disabled policy makes no guard")
    func disabledPolicyMakesNoGuard() {
        #expect(MTPAdaptiveGuard(policy: .disabled, blockSize: Self.blockSize) == nil)
    }

    @Test(
        "a policy outside its ranges is refused",
        arguments: [
            MTPAdaptivePolicy(minimumRoundSamples: 0),
            MTPAdaptivePolicy(probeLength: 1),
            MTPAdaptivePolicy(probeInterval: 0),
            MTPAdaptivePolicy(probeInterval: 8, maximumProbeInterval: 4),
            MTPAdaptivePolicy(stalePlainEstimateRounds: 0),
            MTPAdaptivePolicy(stalePlainContextGrowth: -0.1),
            MTPAdaptivePolicy(requiredSpeedup: 0),
            MTPAdaptivePolicy(requiredSpeedup: 1.2, growthSpeedup: 1.1),
            MTPAdaptivePolicy(growthHoldRounds: -1),
            MTPAdaptivePolicy(pauseLength: 0),
            MTPAdaptivePolicy(pauseLength: 8, maximumPauseLength: 4),
            MTPAdaptivePolicy(smoothing: 0),
            MTPAdaptivePolicy(smoothing: 1.5),
        ])
    func invalidPolicyIsRefused(policy: MTPAdaptivePolicy) {
        #expect(throws: SpeculativeDecodingConfigurationError.self) {
            try policy.validate()
        }
    }

    @Test("the default policy is enabled and valid, and the disabled one is not enabled")
    func defaultPolicyIsValid() throws {
        try MTPAdaptivePolicy.default.validate()

        #expect(MTPAdaptivePolicy.default.isEnabled)
        #expect(!MTPAdaptivePolicy.disabled.isEnabled)
    }
}

/// A clock that moves a fixed step forward each time it is read.
private final class ScriptedClock: Sendable {

    /// The seconds each read moves the clock.
    private let step: TimeInterval

    /// The seconds of the last read.
    private let now = OSAllocatedUnfairLock<TimeInterval>(initialState: 0)

    /// - Parameter step: The seconds each read moves the clock.
    init(step: TimeInterval) {
        self.step = step
    }

    /// The clock of a guard, which reads this clock.
    var adaptiveClock: MTPAdaptiveClock {
        MTPAdaptiveClock { [self] in
            now.withLock { now in
                now += step
                return now
            }
        }
    }
}
