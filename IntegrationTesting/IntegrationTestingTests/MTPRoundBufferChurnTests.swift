// Copyright © 2026 Apple Inc.
//
// Measures the MLX buffers that one MTP round of Qwen3.8-27B-4bit with its 4-bit MTP head
// allocates and gives back, at 2k, 32k and 100k context tokens, on the text load path
// (`LLMModelFactory`). The cadence that clears the buffer pool is off, and the pool is cleared
// before each step, so the pool after a step holds the bytes the step freed and did not take
// back. Each round line also states the bytes of the sources the cache shapes predict: the
// recurrent snapshot slots of the gated delta net layers, their states after the verify, the
// copy of a selected slot, and the K/V buffers of the attention layers.
//
// Run it in Release, one context at a time:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES '-only-testing:IntegrationTestingTests/MTPRoundBufferChurnTests/bytesOfEachRound(contextTokenCount:)'`
//
// Every measurement line goes to standard output and to the unified log under the
// `MTP ROUND CHURN:` prefix.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXVLM
import Testing
import Tokenizers
import os

@testable import MLXLMCommon

/// The log every measurement line of this suite goes to.
private let churnLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "MTPRoundBufferChurn")

/// The prefix of every measurement line.
private let churnPrefix = "MTP ROUND CHURN:"

private let churnTargetID = "mlx-community/Qwen3.8-27B-4bit"
private let churnDrafterID = "mlx-community/Qwen3.8-27B-MTP-4bit"

/// Whether the target and the drafter are local. Asking loads nothing.
private let churnPairIsLocal =
    hfSnapshotDir(modelId: churnTargetID) != nil && hfSnapshotDir(modelId: churnDrafterID) != nil

private let bytesPerMebibyte = 1_048_576.0

/// The arrays of a gated delta net layer's `MambaCache`: its conv state, then its recurrent state.
private enum GatedDeltaStateSlot {
    static let conv = 0
    static let recurrent = 1
    static let count = 2
}

/// The bytes of one step of the iterator, from the MLX memory counters.
private struct StepBytes {
    /// The highest active memory of the step, above the active memory before it.
    let transient: Int
    /// The bytes the step freed into the pool and did not take back.
    let freed: Int
    /// The active memory after the step, less the active memory before it.
    let retained: Int
}

/// The bytes the cache shapes predict for one round.
private struct PredictedBytes {
    /// The snapshot slots of every gated delta net layer, for a verify of `tokens` tokens.
    let snapshots: Int
    /// The recurrent and conv states of every gated delta net layer.
    let states: Int
    /// The K/V buffers of every attention layer, as allocated.
    let keyValueBuffers: Int
    /// The K/V buffer of the largest attention layer.
    let largestKeyValueBuffer: Int

    init(cache: [KVCache], verifyTokens: Int) {
        var snapshots = 0
        var states = 0
        var keyValueBuffers = 0
        var largest = 0
        for layer in cache {
            if let mamba = layer as? MambaCache {
                let arrays = mamba.innerState()
                let stateBytes = arrays.reduce(0) { $0 + $1.nbytes }
                states += stateBytes
                if arrays.count == GatedDeltaStateSlot.count {
                    let conv = arrays[GatedDeltaStateSlot.conv]
                    let recurrent = arrays[GatedDeltaStateSlot.recurrent]
                    let convRowBytes = conv.nbytes / max(conv.dim(1), 1)
                    snapshots +=
                        verifyTokens * recurrent.nbytes
                        + (conv.dim(1) + verifyTokens) * convRowBytes
                }
            } else if let simple = layer as? KVCacheSimple {
                let bytes = (simple.keys?.nbytes ?? 0) + (simple.values?.nbytes ?? 0)
                keyValueBuffers += bytes
                largest = max(largest, bytes)
            }
        }
        self.snapshots = snapshots
        self.states = states
        self.keyValueBuffers = keyValueBuffers
        self.largestKeyValueBuffer = largest
    }
}

private func mebibytes(_ bytes: Int) -> String {
    String(format: "%.1f", Double(bytes) / bytesPerMebibyte)
}

/// The median of `values` as ``SampleMedian`` takes it, or 0 when no round of that kind ran.
private func median(_ values: [Int]) -> Int {
    values.isEmpty ? 0 : Int(SampleMedian.of(values.map(Double.init)))
}

private func log(_ line: String) {
    // swiftlint:disable:next no_direct_standard_out_logs — a measurement line, read from the test output.
    print("\(churnPrefix) \(line)")
    churnLog.info("\(churnPrefix, privacy: .public) \(line, privacy: .public)")
}

@Suite(.serialized)
struct MTPRoundBufferChurnTests {

    /// The context sizes of the measurement.
    static let contextTokenCounts = [2_048, 32_768, 100_000]

    /// The rounds before the measured rounds, so that every kernel is compiled.
    private static let warmUpRounds = 6

    /// The measured rounds.
    private static let measuredRounds = 32

    /// The prompt of about `tokenCount` tokens: a numbered list the model continues.
    private static func prompt(tokenCount: Int, countTokens: (String) -> Int) -> String {
        var lines: [String] = []
        var text = ""
        var index = 0
        // About twelve tokens a line; recount the whole text every thousand lines.
        while true {
            lines.append(
                "Item \(index): the stock count of bin \(index % 97) is \(index * 7 % 1000).")
            index += 1
            if index % 1000 == 0 || lines.count * 12 >= tokenCount {
                text = lines.joined(separator: "\n")
                if countTokens(text) >= tokenCount {
                    break
                }
            }
        }
        return text + "\n\nContinue the list with the next forty items, in the same format."
    }

    @Test(.enabled(if: churnPairIsLocal), arguments: contextTokenCounts)
    func bytesOfEachRound(contextTokenCount: Int) async throws {
        let context = try await LLMModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: churnTargetID))
        await Qwen35TextMTPRegistration.register()
        let drafter = try await MTPDrafterModelFactory.shared.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: churnDrafterID)
        ).model
        let tokenizer: any MLXLMCommon.Tokenizer = context.tokenizer
        let text = Self.prompt(
            tokenCount: contextTokenCount,
            countTokens: { tokenizer.encode(text: $0, addSpecialTokens: false).count })
        let input = try await context.processor.prepare(input: UserInput(chat: [.user(text)]))

        var parameters = GenerateParameters(temperature: 0)
        parameters.maxTokens = 4 * (Self.warmUpRounds + Self.measuredRounds) + 64
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: context.model, drafter: drafter, parameters: parameters,
            blockSize: SpeculativeDecodingConfig.defaultMTPBlockSize,
            minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
        iterator.bufferPoolClearing = GenerationBufferPoolClearing(clearPool: {})

        var transients: [Int] = []
        var freedBytes: [Int] = []
        var retained: [Int] = []
        var rejectingFreed: [Int] = []
        var acceptingFreed: [Int] = []
        var rounds = 0
        var finished = false
        while rounds < Self.warmUpRounds + Self.measuredRounds, !finished {
            let before = iterator.speculativeDecodingTelemetry
            Memory.clearCache()
            Memory.peakMemory = 0
            let activeBefore = Memory.activeMemory
            let token = iterator.next()
            let peak = Memory.peakMemory
            let pooled = Memory.cacheMemory
            let activeAfter = Memory.activeMemory
            guard token != nil else {
                finished = true
                break
            }
            let after = iterator.speculativeDecodingTelemetry
            guard (after?.roundCount ?? 0) > (before?.roundCount ?? 0) else { continue }
            rounds += 1
            guard rounds > Self.warmUpRounds else { continue }
            let drafted = (after?.draftTokenCount ?? 0) - (before?.draftTokenCount ?? 0)
            let accepted =
                (after?.acceptedDraftTokenCount ?? 0) - (before?.acceptedDraftTokenCount ?? 0)
            let predicted = PredictedBytes(cache: iterator.mainCache, verifyTokens: drafted + 1)
            let bytes = StepBytes(
                transient: max(peak - activeBefore, 0), freed: pooled,
                retained: activeAfter - activeBefore)
            transients.append(bytes.transient)
            freedBytes.append(bytes.freed)
            retained.append(bytes.retained)
            if accepted < drafted {
                rejectingFreed.append(bytes.freed)
            } else {
                acceptingFreed.append(bytes.freed)
            }
            log(
                """
                context \(contextTokenCount) round \(rounds - Self.warmUpRounds): drafted \
                \(drafted) accepted \(accepted); transient \(mebibytes(bytes.transient)) MiB, \
                freed \(mebibytes(bytes.freed)) MiB, retained \(mebibytes(bytes.retained)) MiB; \
                predicted GDN snapshot slots \(mebibytes(predicted.snapshots)) MiB, GDN states \
                \(mebibytes(predicted.states)) MiB, K/V buffers \
                \(mebibytes(predicted.keyValueBuffers)) MiB (largest layer \
                \(mebibytes(predicted.largestKeyValueBuffer)) MiB)
                """)
        }

        let predicted = PredictedBytes(
            cache: iterator.mainCache, verifyTokens: SpeculativeDecodingConfig.defaultMTPBlockSize)
        log(
            """
            context \(contextTokenCount) summary over \(freedBytes.count) rounds: median \
            transient \(mebibytes(median(transients))) MiB, median freed \
            \(mebibytes(median(freedBytes))) MiB (rejecting rounds \(rejectingFreed.count) \
            median \(mebibytes(median(rejectingFreed))) MiB, accepting rounds \
            \(acceptingFreed.count) median \(mebibytes(median(acceptingFreed))) MiB), median \
            retained \(mebibytes(median(retained))) MiB; full-block GDN snapshot slots \
            \(mebibytes(predicted.snapshots)) MiB, GDN states \(mebibytes(predicted.states)) \
            MiB, K/V buffers \(mebibytes(predicted.keyValueBuffers)) MiB, largest K/V layer \
            \(mebibytes(predicted.largestKeyValueBuffer)) MiB; active \
            \(mebibytes(Memory.activeMemory)) MiB
            """)
        #expect(freedBytes.count > 0)
    }

    // MARK: - Decode rate and resident memory

    /// The load paths of the decode rate measurement.
    enum LoadPath: String, CaseIterable, Sendable {
        /// `LLMModelFactory` and the text `Qwen35` target, with the text drafter.
        case text
        /// `VLMModelFactory` and the vision-language `Qwen35` target the FoundationModels router
        /// loads, with the vision-language drafter.
        case router
    }

    /// The tokens each decode of the rate measurement writes.
    private static let rateDecodeTokenCount = 512

    /// The tokens of the decode that compiles the kernels before the measurement.
    private static let rateWarmUpTokenCount = 64

    /// The rate of one decode and the memory it held.
    private struct DecodeMeasurement {
        let tokens: [Int]
        let tokensPerSecond: Double
        let peakBytes: Int
        let activeBytes: Int
    }

    /// Decodes `count` tokens of `iterator` after its prefill and reads the rate and memory.
    private static func decode<Tokens: IteratorProtocol>(
        _ iterator: inout Tokens, count: Int
    ) -> DecodeMeasurement where Tokens.Element == Int {
        Memory.peakMemory = 0
        let clock = ContinuousClock()
        let start = clock.now
        var tokens: [Int] = []
        while tokens.count < count, let token = iterator.next() {
            tokens.append(token)
        }
        let seconds =
            Double((clock.now - start).components.attoseconds) / 1e18
            + Double((clock.now - start).components.seconds)
        return DecodeMeasurement(
            tokens: tokens, tokensPerSecond: Double(tokens.count) / seconds,
            peakBytes: Memory.peakMemory, activeBytes: Memory.activeMemory)
    }

    /// Loads the target and the drafter of `path`.
    private static func load(_ path: LoadPath) async throws -> (
        context: ModelContext, drafter: any MTPDrafterModel
    ) {
        let configuration = ModelConfiguration(id: churnTargetID)
        let context: ModelContext
        let drafterFactory: MTPDrafterModelFactory
        switch path {
        case .text:
            context = try await LLMModelFactory.shared.load(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration)
            await Qwen35TextMTPRegistration.register()
            drafterFactory = .shared
        case .router:
            context = try await VLMModelFactory.shared.load(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration)
            await Qwen35VLMMTPRegistration.register()
            drafterFactory = .visionLanguage
        }
        let drafter = try await drafterFactory.load(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: churnDrafterID)
        ).model
        return (context, drafter)
    }

    /// Greedy MTP and plain decode rates and resident memory at each context of
    /// ``contextTokenCounts``, on `path`. Plain decode is the noise floor of a comparison of two
    /// builds.
    private func measureDecodeRates(on path: LoadPath) async throws {
        let (context, drafter) = try await Self.load(path)
        let tokenizer: any MLXLMCommon.Tokenizer = context.tokenizer
        let countTokens = { (text: String) in
            tokenizer.encode(text: text, addSpecialTokens: false).count
        }
        var parameters = GenerateParameters(temperature: 0)
        parameters.maxTokens = Self.rateDecodeTokenCount

        let warmUp = try await context.processor.prepare(
            input: UserInput(chat: [.user(Self.prompt(tokenCount: 256, countTokens: countTokens))]))
        var warmUpMTP = try MTPSpeculativeTokenIterator(
            input: warmUp, mainModel: context.model, drafter: drafter, parameters: parameters,
            blockSize: SpeculativeDecodingConfig.defaultMTPBlockSize,
            minimumDraftProbability: SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
        _ = Self.decode(&warmUpMTP, count: Self.rateWarmUpTokenCount)
        var warmUpPlain = try TokenIterator(
            input: warmUp, model: context.model, parameters: parameters)
        _ = Self.decode(&warmUpPlain, count: Self.rateWarmUpTokenCount)

        for contextTokenCount in Self.contextTokenCounts {
            let input = try await context.processor.prepare(
                input: UserInput(chat: [
                    .user(Self.prompt(tokenCount: contextTokenCount, countTokens: countTokens))
                ]))
            // Each decode holds its own cache, which goes with its iterator.
            let (mtp, telemetry) = try {
                Memory.clearCache()
                var iterator = try MTPSpeculativeTokenIterator(
                    input: input, mainModel: context.model, drafter: drafter,
                    parameters: parameters,
                    blockSize: SpeculativeDecodingConfig.defaultMTPBlockSize,
                    minimumDraftProbability:
                        SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
                let measurement = Self.decode(&iterator, count: Self.rateDecodeTokenCount)
                return (measurement, iterator.speculativeDecodingTelemetry)
            }()
            Memory.clearCache()
            var plainIterator = try TokenIterator(
                input: input, model: context.model, parameters: parameters)
            let plain = Self.decode(&plainIterator, count: Self.rateDecodeTokenCount)
            let firstDivergence =
                zip(mtp.tokens, plain.tokens).enumerated().first { $0.element.0 != $0.element.1 }?
                .offset
            log(
                """
                rate \(path) context \(contextTokenCount): MTP \(String(format: "%.2f", mtp.tokensPerSecond)) \
                tok/s, plain \(String(format: "%.2f", plain.tokensPerSecond)) tok/s; MTP peak \
                \(mebibytes(mtp.peakBytes)) MiB active \(mebibytes(mtp.activeBytes)) MiB, plain \
                peak \(mebibytes(plain.peakBytes)) MiB active \(mebibytes(plain.activeBytes)) MiB; \
                acceptance \(String(format: "%.3f", telemetry?.acceptanceRate ?? 0)) over \
                \(telemetry?.roundCount ?? 0) rounds; first divergence from plain \
                \(firstDivergence.map(String.init) ?? "none") of \(mtp.tokens.count)
                """)
            #expect(mtp.tokens.count == Self.rateDecodeTokenCount)
            #expect(plain.tokens.count == Self.rateDecodeTokenCount)
        }
    }

    @Test(.enabled(if: churnPairIsLocal))
    func textPathDecodeRate() async throws {
        try await measureDecodeRates(on: .text)
    }

    @Test(.enabled(if: churnPairIsLocal))
    func routerPathDecodeRate() async throws {
        try await measureDecodeRates(on: .router)
    }
}
