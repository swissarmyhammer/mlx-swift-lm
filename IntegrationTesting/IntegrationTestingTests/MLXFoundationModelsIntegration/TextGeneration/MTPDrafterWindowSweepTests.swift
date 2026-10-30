// Copyright © 2026 Apple Inc.
//
// Qwen3.8-27B-4bit with its 4-bit MTP head decoding after 32k, 64k and 100k tokens of an agent
// transcript, with the drafter attending to the whole context and to each window of
// `DrafterWindowSweep.windowTokenCounts` (`GenerateParameters.mtpDrafterAttentionWindow`), on the
// router's load path and on the text path, beside plain decode of the same prompt. Each line
// states the acceptance, the tokens of each verify, the milliseconds of each round and the decode
// rate, so the fastest window at each context picks the default. The drafter only proposes, so
// every window writes plain decode's tokens up to a near tie. Run it in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPDrafterWindowSweepTests`
//
// Every measurement line goes to the unified log under the `MTP DRAFTER WINDOW:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPDrafterWindow"'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import BenchmarkHelpers
import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// The time limit of each test, in minutes: for each of three contexts, six prefills of up to
/// 100k tokens (plain decode, and MTP with each drafter window) and six decodes of 1024 tokens.
private let drafterWindowSweepTimeLimitMinutes = 240

/// The drafter attention windows and the context lengths the drafter-window benchmarks sweep.
enum DrafterWindowSweep {

    /// The shortest context of the sweep, in tokens: 32k.
    private static let shortContextTokenCount = 32_768

    /// The middle context of the sweep, in tokens: 64k.
    private static let middleContextTokenCount = 65_536

    /// The longest context of the sweep, in tokens: 100k, the far end of a SWE-bench run.
    private static let longContextTokenCount = 100_000

    /// The widest drafter window of the sweep, in tokens: 16k.
    private static let widestWindowTokenCount = 16_384

    /// The second drafter window of the sweep, in tokens: 8k.
    private static let wideWindowTokenCount = 8_192

    /// The third drafter window of the sweep, in tokens: 4k.
    private static let narrowWindowTokenCount = 4_096

    /// The narrowest drafter window of the sweep, in tokens: 2k.
    private static let narrowestWindowTokenCount = 2_048

    /// The context lengths of the sweep, in tokens.
    static let contextTokenCounts = [
        shortContextTokenCount, middleContextTokenCount, longContextTokenCount,
    ]

    /// The drafter windows of the sweep, in tokens; `nil` attends to the whole context.
    static let windowTokenCounts: [Int?] = [
        nil, widestWindowTokenCount, wideWindowTokenCount, narrowWindowTokenCount,
        narrowestWindowTokenCount,
    ]

    /// The greedy parameters of a stream whose drafter attends to `window` tokens, with the
    /// prefix `maxKVSize` keeps, or to the whole context when `window` is `nil`.
    ///
    /// - Parameter window: The drafter window, in tokens.
    /// - Returns: The parameters.
    /// - Throws: ``KVCacheConfigurationError`` for a window that cannot hold the kept prefix.
    static func parameters(window: Int?) throws -> GenerateParameters {
        var parameters = GenerateParameters(temperature: 0)
        parameters.mtpDrafterAttentionWindow = try window.map {
            try KVCacheConfiguration.Capacity(maxTokens: $0)
        }
        return parameters
    }

    /// The name of `window` in a measurement line.
    static func label(of window: Int?) -> String {
        window.map { "\($0)" } ?? "full"
    }
}

/// What one decode of a sweep point wrote, and how fast.
private struct SweepDecode: Sendable {

    /// The tokens, with the two best candidates of each step when plain decode wrote them.
    let tokens: [MTPChatToken]

    /// The seconds from the first token to the last.
    let seconds: Double

    /// The speculation of an MTP decode, or `nil` for plain decode.
    let telemetry: SpeculativeDecodingTelemetry?

    /// Why an MTP decode stopped speculating, or `nil`.
    let passthroughReason: String?

    /// The tokens per second after the first token.
    var tokensPerSecond: Double { Double(max(tokens.count - 1, 0)) / max(seconds, .ulpOfOne) }

    /// The milliseconds of each round of an MTP decode.
    var millisecondsPerRound: Double {
        seconds * MTPDrafterWindowSweepTests.millisecondsPerSecond
            / Double(max(telemetry?.roundCount ?? 0, 1))
    }
}

/// The decodes of one context length: plain decode, then MTP with each drafter window.
private struct SweepPoint: Sendable {

    /// The tokens of the rendered prompt.
    let promptTokenCount: Int

    /// Plain decode, which reports the two best candidates of each step.
    let plain: SweepDecode

    /// MTP with each window of ``DrafterWindowSweep/windowTokenCounts``, in order.
    let mtp: [SweepDecode]
}

/// Decodes after 32k, 64k and 100k tokens with plain decode and with MTP over each drafter window,
/// on the router's load path and on the text path.
@Suite(.serialized, .timeLimit(.minutes(drafterWindowSweepTimeLimitMinutes)))
struct MTPDrafterWindowSweepTests {

    /// The milliseconds of one second.
    static let millisecondsPerSecond = 1_000.0

    /// The tokens each decode writes at most.
    private static let decodeTokenCount = 1_024

    /// The most history rounds a context may draw on.
    private static let maximumHistoryRoundCount = 512

    /// The candidates plain decode reports for each step: the two of a near tie.
    private static let topCandidateCount = 2

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP DRAFTER WINDOW:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPDrafterWindow")

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: Qwen38FourBitMTPModel.LoadPath.allCases)
    func decodeRateByDrafterWindow(path: Qwen38FourBitMTPModel.LoadPath) async throws {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        let loaded = try await Qwen38FourBitMTPModel.loadWithDrafter(path: path)
        let tokenizer = await loaded.container.tokenizer
        let speculation = loaded.speculation
        for contextTokenCount in DrafterWindowSweep.contextTokenCounts {
            let history = try Self.history(
                minimumTokenCount: contextTokenCount,
                countTokens: { tokenizer.encode(text: $0, addSpecialTokens: false).count })
            let point = try await loaded.perform { context, drafter in
                try await Self.measurePoint(
                    history: history, context: context, drafter: drafter,
                    speculation: speculation)
            }
            try expectPoint(point, label: "\(path) context \(contextTokenCount)")
        }
        await loaded.model.evict()
        await releaseAllGPUMemory()
    }

    /// The text of the history exchanges, in order, until they hold `minimumTokenCount` tokens.
    ///
    /// - Parameters:
    ///   - minimumTokenCount: The tokens the history must reach.
    ///   - countTokens: The tokens of a text.
    /// - Returns: The history.
    /// - Throws: When ``maximumHistoryRoundCount`` exchanges do not reach the count.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func history(
        minimumTokenCount: Int, countTokens: (String) -> Int
    ) throws -> String {
        var exchanges: [String] = []
        var tokenCount = 0
        for index in 0 ..< maximumHistoryRoundCount where tokenCount < minimumTokenCount {
            let exchange = SWEBenchTranscript.exchangeText(index)
            exchanges.append(exchange)
            tokenCount += countTokens(exchange)
        }
        try #require(tokenCount >= minimumTokenCount, "the history is too short")
        return exchanges.joined(separator: "\n\n")
    }

    /// Renders the agent prompt over `history`, then decodes it plainly and with MTP over each
    /// drafter window, each from a new cache.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func measurePoint(
        history: String, context: ModelContext, drafter: any MTPDrafterModel,
        speculation: MLXSpeculativeDecoding
    ) async throws -> SweepPoint {
        let input = try await context.processor.prepare(
            input: UserInput(chat: [
                .system(SWEBenchTranscript.systemPrompt),
                .user(history + "\n\n" + MTPSWEBenchScaleTests.longDecodePrompt),
            ]))
        let stopTokens = context.configuration.eosTokenIds.union(
            [context.tokenizer.eosTokenId].compactMap { $0 })
        let plain = try decodePlainly(input: input, context: context, stopTokens: stopTokens)
        let mtp = try DrafterWindowSweep.windowTokenCounts.map { window in
            try decodeWithMTP(
                input: input, context: context, drafter: drafter, speculation: speculation,
                parameters: try DrafterWindowSweep.parameters(window: window),
                stopTokens: stopTokens)
        }
        return SweepPoint(promptTokenCount: input.text.tokens.size, plain: plain, mtp: mtp)
    }

    /// Plain greedy decode of `input`, which reports the two best candidates of each step.
    private static func decodePlainly(
        input: LMInput, context: ModelContext, stopTokens: Set<Int>
    ) throws -> SweepDecode {
        var parameters = try DrafterWindowSweep.parameters(window: nil)
        parameters.logProbabilities = topCandidateCount
        var iterator = try TokenIterator(
            input: input, model: context.model, parameters: parameters)
        let decode = timedDecode(stopTokens: stopTokens) {
            iterator.next().map { token in
                MTPChatToken(
                    token: token,
                    candidates: iterator.lastLogProbabilities?.materialize().topLogProbabilities
                        ?? [])
            }
        }
        Memory.clearCache()
        return SweepDecode(
            tokens: decode.tokens, seconds: decode.seconds, telemetry: nil, passthroughReason: nil)
    }

    /// Greedy MTP decode of `input` with the block size and the confidence floor of
    /// `speculation`, under `parameters`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func decodeWithMTP(
        input: LMInput, context: ModelContext, drafter: any MTPDrafterModel,
        speculation: MLXSpeculativeDecoding, parameters: GenerateParameters, stopTokens: Set<Int>
    ) throws -> SweepDecode {
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: context.model, drafter: drafter, parameters: parameters,
            blockSize: speculation.blockSize,
            minimumDraftProbability: speculation.minimumDraftProbability)
        let decode = timedDecode(stopTokens: stopTokens) {
            iterator.next().map { MTPChatToken(token: $0) }
        }
        let result = SweepDecode(
            tokens: decode.tokens, seconds: decode.seconds,
            telemetry: iterator.speculativeDecodingTelemetry,
            passthroughReason: iterator.passthroughReason)
        Memory.clearCache()
        return result
    }

    /// Reads tokens from `next` until a stop token or ``decodeTokenCount`` tokens, timing every
    /// token after the first, so the time of the prefill stays out of the rate.
    private static func timedDecode(
        stopTokens: Set<Int>, next: () -> MTPChatToken?
    ) -> (tokens: [MTPChatToken], seconds: Double) {
        guard let first = next() else { return ([], 0) }
        let start = Date()
        var tokens = [first]
        while tokens.count < decodeTokenCount, !stopTokens.contains(tokens[tokens.count - 1].token),
            let token = next()
        {
            tokens.append(token)
        }
        return (tokens, Date().timeIntervalSince(start))
    }

    /// Logs each MTP decode of `point` against plain decode and against the whole-context drafter,
    /// and holds that each one speculated and wrote plain decode's tokens up to a near tie.
    private func expectPoint(_ point: SweepPoint, label: String) throws {
        let full = try #require(point.mtp.first)
        Self.log(
            """
            \(label): prompt \(point.promptTokenCount) tokens, plain \(point.plain.tokens.count) \
            tokens at \(point.plain.tokensPerSecond) tok/s
            """)
        for (window, decode) in zip(DrafterWindowSweep.windowTokenCounts, point.mtp) {
            let windowLabel = "\(label) drafter window \(DrafterWindowSweep.label(of: window))"
            let match = MTPChatTokenMatch(mtp: decode.tokens, plain: point.plain.tokens)
            Self.log(
                "\(windowLabel): \(Self.describe(decode, plain: point.plain, full: full)); \(match.summary)"
            )
            #expect((decode.telemetry?.draftTokenCount ?? 0) > 0, "\(windowLabel)")
            #expect(decode.passthroughReason == nil, "\(windowLabel)")
            #expect(match.holds, "\(windowLabel): \(match.summary)")
        }
    }

    /// The numbers of one MTP decode against plain decode and the whole-context drafter.
    private static func describe(
        _ decode: SweepDecode, plain: SweepDecode, full: SweepDecode
    ) -> String {
        let telemetry = decode.telemetry
        return """
            \(decode.tokens.count) tokens at \(decode.tokensPerSecond) tok/s, \
            \(decode.tokensPerSecond / plain.tokensPerSecond)x plain, \
            \(decode.tokensPerSecond / full.tokensPerSecond)x the whole-context drafter; \
            acceptance \(telemetry?.acceptanceRate ?? 0), tokens per verify \
            \(telemetry?.meanEmittedTokensPerTargetCall ?? 0), \(decode.millisecondsPerRound) ms \
            per round over \(telemetry?.roundCount ?? 0) rounds
            """
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
