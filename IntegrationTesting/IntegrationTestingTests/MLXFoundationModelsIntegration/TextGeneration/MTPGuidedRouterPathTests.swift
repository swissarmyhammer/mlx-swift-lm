// Copyright © 2026 Apple Inc.
//
// One schema turn of Qwen3.8-27B-4bit with its 4-bit MTP head after 8k tokens of an agent
// transcript, on the FoundationModels router's load path, beside the same turn without the head.
// The turn speculates, reports no fallback, and writes the plain tokens up to a near tie. The text
// path's turn of this prompt parts from plain at a gap of 0.125, past the near tie this suite
// allows, so ``MTPGuidedRouterPathTests`` holds the router path only. Run it in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPGuidedRouterPathTests`
//
// The measurement line goes to the unified log under the `MTP GUIDED ROUTER:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPGuidedRouterPath"'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXVLM
import Testing
import os

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The items of ``GuidedRouterInventory``.
private let guidedRouterItemCount = 60

/// The time limit of each test, in minutes: two prefills of 8k tokens, two decodes of up to 16k.
private let guidedRouterTimeLimitMinutes = 30

/// One item of the inventory a turn writes.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct GuidedRouterItem {
    @Guide(description: "The file or function the item is about")
    var subject: String

    @Guide(description: "What the transcript says about it, in one sentence")
    var finding: String

    @Guide(description: "The next step, in one sentence")
    var nextStep: String
}

/// The inventory a turn writes.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct GuidedRouterInventory {
    @Guide(.count(guidedRouterItemCount))
    var items: [GuidedRouterItem]
}

/// A schema turn on the router's load path speculates, and writes the plain tokens.
@Suite(.serialized, .timeLimit(.minutes(guidedRouterTimeLimitMinutes)))
struct MTPGuidedRouterPathTests {

    /// The tokens of context before the request.
    private static let contextTokenCount = 8_192

    /// The tokens the turn writes at most.
    private static let responseTokenBudget = 16_384

    /// The most history rounds the context may draw on.
    private static let maximumHistoryRoundCount = 256

    /// The widest gap between the two best constrained logits of plain decoding at which the MTP
    /// output may first part from the plain output.
    private static let nearTieLogitGap: Float = 1e-3

    /// The request after the transcript.
    private static let request = """
        List sixty things in the transcript above that someone should follow up on, each \
        with its subject, what the transcript says about it, and the next step.
        """

    /// Prefix that makes the measurement line greppable in the log.
    private static let measurementPrefix = "MTP GUIDED ROUTER:"

    /// The log the measurement line goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPGuidedRouterPath")

    /// What one turn recorded, and how long it took.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Turn {
        let sink: GuidedGenerationDiagnosticSink
        let seconds: Double

        /// The tokens of the turn per second, the prefill included.
        var tokensPerSecond: Double {
            Double(sink.generatedTokenCount) / max(seconds, .ulpOfOne)
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: [Qwen38FourBitMTPModel.LoadPath.router])
    func aSchemaTurnSpeculatesToThePlainTokens(path: Qwen38FourBitMTPModel.LoadPath) async throws {
        await releaseAllGPUMemory()
        let plainModel = try Qwen38FourBitMTPModel.make(speculates: false, guides: true, path: path)
        let speculativeModel = try Qwen38FourBitMTPModel.make(
            speculates: true, guides: true, path: path)
        let container = try await plainModel.loadContainer()
        let isVisionTarget = await container.perform { $0.model is MLXVLM.Qwen35 }
        let tokenizer = await container.tokenizer
        let prompt = try Self.prompt(
            countTokens: { tokenizer.encode(text: $0, addSpecialTokens: false).count })

        let plain = try await Self.respond(to: prompt, model: plainModel)
        let speculative = try await Self.respond(to: prompt, model: speculativeModel)

        let divergence = Self.firstDivergence(of: speculative.sink, from: plain.sink)
        Self.log(path: path, plain: plain, speculative: speculative, divergence: divergence)
        #expect(isVisionTarget == (path == .router))
        #expect(speculative.sink.draftedTokenCount > 0)
        #expect(speculative.sink.speculationFallbacks.isEmpty)
        #expect(
            divergence.map { ($0.margin ?? .infinity) < Self.nearTieLogitGap } ?? true,
            "MTP parted from plain away from a near tie: \(String(describing: divergence))")

        await speculativeModel.evict()
        await plainModel.evict()
        await releaseAllGPUMemory()
    }

    /// The transcript until it holds ``contextTokenCount`` tokens, then ``request``.
    ///
    /// - Parameter countTokens: The tokens of a text.
    /// - Returns: The prompt.
    /// - Throws: When ``maximumHistoryRoundCount`` exchanges do not reach the count.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(countTokens: (String) -> Int) throws -> String {
        var exchanges: [String] = []
        var tokenCount = 0
        for index in 0 ..< maximumHistoryRoundCount where tokenCount < contextTokenCount {
            let exchange = SWEBenchTranscript.exchangeText(index)
            exchanges.append(exchange)
            tokenCount += countTokens(exchange)
        }
        try #require(tokenCount >= contextTokenCount, "the history is too short")
        return (exchanges + [request]).joined(separator: "\n\n")
    }

    /// One greedy schema turn of `model` in a new session, which records its decisions.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respond(to prompt: String, model: MLXLanguageModel) async throws -> Turn {
        let sink = GuidedGenerationDiagnosticSink()
        let start = ContinuousClock.now
        _ = try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            try await LanguageModelSession(model: model).respond(
                to: prompt, generating: GuidedRouterInventory.self,
                options: GenerationOptions(
                    samplingMode: .greedy, maximumResponseTokens: responseTokenBudget))
        }
        let elapsed = (ContinuousClock.now - start).components
        return Turn(
            sink: sink, seconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) * 1e-18)
    }

    /// Where the decisions of `run` first part from the decisions of `plain`, with plain's margin
    /// there, or `nil` when they write the same tokens.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func firstDivergence(
        of run: GuidedGenerationDiagnosticSink, from plain: GuidedGenerationDiagnosticSink
    ) -> (index: Int, margin: Float?)? {
        let tokens = run.sampledTokenIDs
        let plainTokens = plain.sampledTokenIDs
        let common = min(tokens.count, plainTokens.count)
        guard
            let index = (0 ..< common).first(where: { tokens[$0] != plainTokens[$0] })
                ?? (tokens.count == plainTokens.count ? nil : common)
        else { return nil }
        let margins = plain.decisionMargins
        return (index, margins.indices.contains(index) ? margins[index] : nil)
    }

    /// Logs the two turns.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func log(
        path: Qwen38FourBitMTPModel.LoadPath, plain: Turn, speculative: Turn,
        divergence: (index: Int, margin: Float?)?
    ) {
        let nearTies = plain.sink.decisionMargins.filter { $0 < nearTieLogitGap }.count
        let line = """
            \(measurementPrefix) \(path) plain \(plain.sink.generatedTokenCount) tokens in \
            \(plain.seconds) s (\(plain.tokensPerSecond) tok/s); MTP \
            \(speculative.sink.generatedTokenCount) tokens in \(speculative.seconds) s \
            (\(speculative.tokensPerSecond) tok/s), drafted \(speculative.sink.draftedTokenCount), \
            accepted \(speculative.sink.acceptedDraftTokenCount), rollbacks \
            \(speculative.sink.rollbacks.count), fallbacks \
            \(speculative.sink.speculationFallbacks); \
            sampled \(plain.sink.sampledTokenIDs.count) and \
            \(speculative.sink.sampledTokenIDs.count), divergence \
            \(divergence.map { "\($0.index) margin \($0.margin ?? .infinity)" } ?? "none"), \
            plain near ties \(nearTies)
            """
        measurementLog.notice("\(line, privacy: .public)")
    }
}

#endif
