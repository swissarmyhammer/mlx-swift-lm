// Copyright © 2026 Apple Inc.
//
// Holds the batched greedy MTP verification to the sequential one on Qwen3.5-4B-4bit with its
// 4-bit MTP head, and times a full decode of each path.
//
// Build for Release, or the timing measures the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPGreedyVerificationIntegrationTests`

import BenchmarkHelpers
import Foundation
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import Testing
import Tokenizers
import os

@_spi(Testing) @testable import MLXLMCommon

private let greedyMTPModelDirectory = hfSnapshotDir(
    modelId: "mlx-community/Qwen3.5-4B-4bit",
    revision: "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c")
private let greedyMTPDrafterDirectory = hfSnapshotDir(
    modelId: "mlx-community/Qwen3.5-4B-MTP-4bit",
    revision: "ab6f59bc6627196c611ab8851638651078170485")

/// Whether the files of the target and of its MTP head are local.
private let greedyMTPFilesAreLocal =
    greedyMTPModelDirectory != nil && greedyMTPDrafterDirectory != nil

/// The log every measurement line of this suite goes to.
private let measurementLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "MTPGreedyVerificationIntegration")

/// The tokens of a full decode: the longest parity run, and each timed run.
private let fullDecodeTokenCount = 128

/// A parity run that stops within the first rounds of drafts.
private let earlyStopTokenCount = 5

/// A parity run that stops part way into a later round.
private let midDecodeStopTokenCount = 13

/// The token counts where the parity runs stop: within the first rounds, part way into a
/// later round, and at the end of a full decode.
private let parityStopCounts = [earlyStopTokenCount, midDecodeStopTokenCount, fullDecodeTokenCount]

/// The rows of each verify block: one draft and its bonus row.
private let verifyBlockSize = 2

/// The timed runs of each path; odd, so the median is one run.
private let timedRepetitions = 5

private let attosecondsPerSecond = 1e18

private struct SequentialMTPProcessor: LogitProcessor {
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray { logits }
    mutating func didSample(token: MLXArray) {}
}

/// What one greedy MTP decode wrote, and how long it took.
private struct GreedyMTPRun {
    /// The emitted tokens.
    let tokens: [Int]
    /// The target caches and then the drafter caches, after the decode.
    let caches: [KVCache]
    /// The tokens the target cache has processed.
    let position: Int
    /// The drafts the target accepted.
    let accepted: Int
    /// The drafts the drafter proposed.
    let proposed: Int
    /// The wall-clock seconds of the decode and of the cache evaluation.
    let seconds: Double
}

/// Qwen3.5-4B-4bit, its MTP head, and the greedy prompt both tests decode.
private struct GreedyMTPFixture {
    let target: ModelContext
    let drafter: MTPDrafterContext
    let input: LMInput
    let parameters = GenerateParameters(maxTokens: fullDecodeTokenCount, temperature: 0)

    /// Loads the target and its MTP head from the local snapshots, and prepares the prompt.
    static func load() async throws -> GreedyMTPFixture {
        let targetDirectory = try #require(greedyMTPModelDirectory)
        let drafterDirectory = try #require(greedyMTPDrafterDirectory)
        let tokenizerLoader = #huggingFaceTokenizerLoader()
        let target = try await LLMModelFactory.shared.load(
            from: targetDirectory, using: tokenizerLoader)
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: drafterDirectory, using: tokenizerLoader)
        let input = try await target.processor.prepare(
            input: UserInput(chat: [
                .user("Why is the sky blue? Explain the physics, then give a simple analogy.")
            ]))
        return GreedyMTPFixture(target: target, drafter: drafter, input: input)
    }

    /// Decodes `stopAfter` tokens on fresh caches, through the batched verification or, with
    /// `batched` false, through the sequential one.
    func run(batched: Bool, stopAfter: Int) throws -> GreedyMTPRun {
        let caches = try target.model.newCache(parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target.model, drafter: drafter.model,
            mainCache: caches, parameters: parameters, blockSize: verifyBlockSize)
        #expect(
            iterator._processorForTesting == nil
                || iterator._processorForTesting is SuppressTokensProcessor)
        if !batched {
            let processors = iterator._processorForTesting.map { [$0] } ?? []
            iterator._setProcessorForTesting(
                ChainedLogitProcessor(processors: processors + [SequentialMTPProcessor()]))
        }
        Stream.defaultStream(.gpu).synchronize()
        let start = ContinuousClock.now
        var tokens = [Int]()
        while tokens.count < stopAfter, let token = iterator.next() { tokens.append(token) }
        iterator.finalizeGeneration()
        let finalCaches = iterator.mainCache + (iterator.drafterState?.cache ?? [])
        eval(finalCaches.flatMap { $0.state })
        let elapsed = (ContinuousClock.now - start).components
        #expect(iterator.passthroughReason == nil)
        return GreedyMTPRun(
            tokens: tokens, caches: finalCaches,
            position: iterator.mainCacheStorage.processedTokenCount,
            accepted: iterator.acceptedCount, proposed: iterator.proposedCount,
            seconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / attosecondsPerSecond)
    }
}

@Suite(.serialized)
struct MTPGreedyVerificationIntegrationTests {
    @Test(.enabled(if: greedyMTPFilesAreLocal))
    func tokenAndCacheParity() async throws {
        let fixture = try await GreedyMTPFixture.load()

        func expectCacheParity(_ actual: [KVCache], _ expected: [KVCache]) {
            #expect(actual.count == expected.count)
            for (actualCache, expectedCache) in zip(actual, expected) {
                #expect(actualCache.offset == expectedCache.offset)
                #expect(actualCache.state.count == expectedCache.state.count)
                for (a, b) in zip(actualCache.state, expectedCache.state) {
                    #expect(a.shape == b.shape)
                    #expect(a.dtype == b.dtype)
                    #expect(a.asArray(Float.self) == b.asArray(Float.self))
                }
            }
        }

        for stopAfter in parityStopCounts {
            let sequential = try fixture.run(batched: false, stopAfter: stopAfter)
            let batched = try fixture.run(batched: true, stopAfter: stopAfter)
            #expect(batched.tokens == sequential.tokens)
            #expect(batched.tokens.count == stopAfter)
            #expect(batched.accepted == sequential.accepted)
            #expect(batched.proposed == sequential.proposed)
            #expect(batched.position == sequential.position)
            #expect(batched.proposed > 0)
            expectCacheParity(batched.caches, sequential.caches)
            let measurement =
                "[MTPGREEDYPARITY] tokens=\(stopAfter) accepted=\(batched.accepted)/\(batched.proposed) cache=exact"
            measurementLog.info("\(measurement, privacy: .public)")
        }
    }

    /// Times full decodes of the sequential and of the batched verification, the two paths
    /// swapping their order from one repetition to the next, and holds each timed run to the
    /// tokens and acceptance of an untimed sequential reference run.
    @Test(.enabled(if: greedyMTPFilesAreLocal))
    func batchedVerificationDecodeSpeed() async throws {
        let fixture = try await GreedyMTPFixture.load()
        let reference = try fixture.run(batched: false, stopAfter: fullDecodeTokenCount)
        var sequentialSeconds = [Double]()
        var batchedSeconds = [Double]()
        for repetition in 0 ..< timedRepetitions {
            for batched in repetition.isMultiple(of: 2) ? [false, true] : [true, false] {
                let result = try fixture.run(batched: batched, stopAfter: fullDecodeTokenCount)
                #expect(result.tokens == reference.tokens)
                #expect(result.accepted == reference.accepted)
                #expect(result.proposed == reference.proposed)
                if batched {
                    batchedSeconds.append(result.seconds)
                } else {
                    sequentialSeconds.append(result.seconds)
                }
                let measurement =
                    "[MTPGREEDYRUN] batched=\(batched) seconds=\(result.seconds) tokens=\(result.tokens.count)"
                measurementLog.info("\(measurement, privacy: .public)")
            }
        }
        let sequential = BenchmarkStats(times: sequentialSeconds).median
        let batched = BenchmarkStats(times: batchedSeconds).median
        let tokens = Double(fullDecodeTokenCount)
        let measurement = String(
            format:
                "[MTPGREEDYE2E] Qwen3.5-4B-4bit block=%ld tokens=%ld sequential=%.2f tok/s batched=%.2f tok/s speedup=%.3fx",
            verifyBlockSize, fullDecodeTokenCount, tokens / sequential, tokens / batched,
            sequential / batched)
        measurementLog.info("\(measurement, privacy: .public)")
    }
}
