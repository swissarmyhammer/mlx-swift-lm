// Copyright © 2026 Apple Inc.
//
// MTP against plain decode of Qwen3.8-27B-4bit while busy
// threads keep a share of the CPU cores loaded, as the CPU-heavy work beside the ACP agent's
// generation does (test runs, indexing, other sessions' builds). Each round of MTP syncs the host
// with the GPU more often than plain decode, so CPU contention may cost it more. Run it in
// Release, or the throughput measures the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPSWEBenchScaleTests/decodeRateUnderSyntheticCPULoad(busyCorePercent:)`
//
// Every measurement line goes to the unified log under the `MTP SWE-BENCH SCALE:` prefix of
// `MTPSWEBenchScaleTests`.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Synchronization
import Testing

/// Threads that spin until they are stopped, each keeping one CPU core busy.
private final class BusyThreads: Sendable {

    /// Whether the threads stop spinning.
    private let isStopped = Atomic<Bool>(false)

    /// Starts `count` spinning threads.
    ///
    /// - Parameter count: The threads to start.
    func start(count: Int) {
        for _ in 0 ..< count {
            Thread { [self] in spin() }.start()
        }
    }

    /// Stops every thread ``start(count:)`` started.
    func stop() {
        isStopped.store(true, ordering: .relaxed)
    }

    /// Spins until ``stop()``.
    private func spin() {
        while !isStopped.load(ordering: .relaxed) {}
    }
}

extension MTPSWEBenchScaleTests {

    /// The shares of the CPU cores the synthetic load keeps busy, in percent: none, half and all.
    static let busyCorePercents = [0, 50, 100]

    /// The percent of a whole.
    private static let percentPerWhole = 100

    /// The context lengths of the CPU load sweep, in tokens of content: a short chat and the
    /// typical context of the ACP agent's SWE-bench sessions.
    private static let cpuLoadContextTokenCounts = [2_048, 32_768]

    /// Decodes at each context of ``cpuLoadContextTokenCounts`` with MTP and plain decode while
    /// `busyCorePercent` percent of the CPU cores spin, and logs the MTP rate as a share of
    /// plain decode's. Each sweep point holds what
    /// ``measureSweepPoint(_:runners:countTokens:)`` holds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: MTPSWEBenchScaleTests.busyCorePercents)
    func decodeRateUnderSyntheticCPULoad(busyCorePercent: Int) async throws {
        let busyThreadCount =
            ProcessInfo.processInfo.activeProcessorCount * busyCorePercent / Self.percentPerWhole
        let (runners, countTokens) = try await makeSweepRunners(on: .router)
        for contextTokenCount in Self.cpuLoadContextTokenCounts {
            let load = BusyThreads()
            load.start(count: busyThreadCount)
            defer { load.stop() }
            let point = try await measureSweepPoint(
                contextTokenCount, runners: runners, countTokens: countTokens)
            Self.log(
                """
                CPU load \(busyCorePercent)% (\(busyThreadCount) busy threads), sweep \
                \(contextTokenCount): plain \(point.plain) tok/s, MTP at \
                \(point.mtp / point.plain) of plain
                """)
        }
        await releaseAllGPUMemory()
    }
}

#endif
