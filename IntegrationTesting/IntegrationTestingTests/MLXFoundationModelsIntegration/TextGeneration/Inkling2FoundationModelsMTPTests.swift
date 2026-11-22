// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of Inkling 2 Mini MTP through `MLXLanguageModel` and
// `LanguageModelSession`, loaded the way the Router and Extras load it: the
// `mlx-community/Inkling-2-Mini-4bit` target through `VLMModelFactory`, and the DSpark head of
// `mlx-community/Inkling-2-Mini-oQ4e-mtp` through `MLXSpeculativeDecoding` with the
// `.matchingTarget(.qwen35Text, .qwen35VLM, .inkling2)` registration, the sum of the
// `*.safetensors` files of the drafter snapshot as `drafterBytes`, and the
// `.recommendedWorkingSet` memory policy.
//
// A text call, a `@Generable` call and a two-step tool loop with its follow-up turn each draft
// in every call, report no fallback, and write the output of the same calls without MTP. The
// rounds of the tool loop after a tool result and the follow-up turn reuse the prompt cache of
// the session with MTP on. Each kind of call runs in alternating plain and MTP pairs, and the
// suite logs the decode rate of both.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2FoundationModelsMTPTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Synchronization
import Testing
import Tokenizers

@testable import MLXFoundationModels

// MARK: - The guided output

/// The facts of a city, the output of the guided calls.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct Inkling2CityFacts {
    @Guide(description: "The name of the city")
    var name: String

    @Guide(description: "The country of the city")
    var country: String

    @Guide(description: "Three famous landmarks of the city", .count(3))
    var landmarks: [String]

    @Guide(description: "One sentence about the city")
    var summary: String
}

// MARK: - What one call reported

/// What one `respond` call of a session reported to the executor's observer: the MTP outcome
/// of its response entry, the completion of each generation-loop pass, and its usage.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct Inkling2MTPCall {
    /// The metadata of the response entry of the call, which carries its MTP outcome.
    var metadata: [String: GeneratedContent] = [:]

    /// The completion of each generation-loop pass of the call, in order.
    var completions: [GenerateCompletionInfo] = []

    /// The prompt tokens of the call, from its usage: the reused tokens and the fed tokens.
    var renderedTokenCount = 0

    /// The prompt tokens the call reused from the prompt cache, from its usage.
    var cachedTokenCount = 0

    /// The tokens the call generated, from its usage.
    var generatedTokenCount = 0

    /// The drafts of the call, from its metadata.
    var draftedTokenCount: Int {
        (try? metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount]?.value(
            Int.self)) ?? 0
    }

    /// The accepted drafts of the call, from its metadata.
    var acceptedDraftTokenCount: Int {
        (try? metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount]?.value(
            Int.self)) ?? 0
    }

    /// The fallback reason of the call, from its metadata.
    var fallbackReason: String? {
        try? metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason]?.value(String.self)
    }

    /// One line with the counts of the call.
    var summary: String {
        "drafted \(draftedTokenCount), accepted \(acceptedDraftTokenCount), fallback "
            + "\(fallbackReason ?? "none"), prompt \(renderedTokenCount) (cached "
            + "\(cachedTokenCount)), generated \(generatedTokenCount), passes "
            + "\(completions.map { String(format: "%.1f tok/s", $0.tokensPerSecond) })"
    }
}

/// Splits the observer events of a session into its calls: each call of a model with
/// speculative decoding ends with the metadata of its response entry.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class Inkling2MTPCallLog: Sendable {
    /// The calls the observer has seen so far.
    private struct Calls {
        /// The calls that wrote their outcome, in order.
        var finished: [Inkling2MTPCall] = []

        /// What the observer has seen since the last call that wrote its outcome.
        var current = Inkling2MTPCall()
    }

    /// The calls so far. The executor reports on the generation path and the test reads from
    /// its own task, thus a mutex guards them.
    private let calls = Mutex(Calls())

    /// Reads one observer event.
    func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
        calls.withLock { calls in
            switch event {
            case .completion(let info):
                calls.current.completions.append(info)
            case .updateUsage(let input, let output, _):
                calls.current.renderedTokenCount = input.totalTokenCount
                calls.current.cachedTokenCount = input.cachedTokenCount
                calls.current.generatedTokenCount = output.totalTokenCount
            case .updateMetadata(let values, _)
            where values[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] != nil:
                calls.current.metadata = values.mapValues(\.generatedContent)
                calls.finished.append(calls.current)
                calls.current = Inkling2MTPCall()
            case .updateMetadata, .appendText, .toolCall:
                break
            }
        }
    }

    /// The calls that wrote their outcome, then what the observer saw after them: a model
    /// without speculative decoding writes no outcome, so its calls all land there.
    var finished: [Inkling2MTPCall] {
        calls.withLock { calls in
            calls.current.completions.isEmpty && calls.current.generatedTokenCount == 0
                ? calls.finished : calls.finished + [calls.current]
        }
    }
}

// MARK: - The models

/// Inkling 2 Mini as a FoundationModels language model, with its DSpark head or without it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private enum Inkling2MTPLanguageModel {

    /// The bytes Extras passes as `drafterBytes`: the sum of every `*.safetensors` file of the
    /// drafter snapshot, the oQ4e target weights included.
    static func snapshotSafetensorsBytes() async throws -> Int {
        let directory = try await #hubDownloader().download(
            id: inkling2MiniDrafter.name, revision: inkling2MiniOQ4eMTPRevision,
            matching: ["*.safetensors", "*.json"], useLatest: false, progressHandler: { _ in })
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey])
        return try files.filter { $0.pathExtension == "safetensors" }.reduce(0) {
            $0
                + (try $1.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey])
                    .fileSize ?? 0)
        }
    }

    /// The model, without reasoning, so that each call answers without a thinking frame.
    ///
    /// - Parameter drafterBytes: the `drafterBytes` of the speculative decoding, or `nil` for
    ///   the model without MTP.
    static func make(drafterBytes: Int?) throws -> MLXLanguageModel {
        let speculation = try drafterBytes.map {
            try MLXSpeculativeDecoding(
                drafterConfiguration: inkling2MiniDrafter, from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                registration: .matchingTarget(.qwen35Text, .qwen35VLM, .inkling2),
                drafterBytes: $0, memoryPolicy: .recommendedWorkingSet)
        }
        return MLXLanguageModel(
            configuration: inkling2Mini4Bit, capabilities: [.guidedGeneration, .toolCalling],
            speculativeDecoding: speculation,
            weightsLocation: {
                testWeightsLocation(modelID: $0, revision: inkling2Mini4BitRevision)
            },
            load: { _, _ in try await Inkling2MiniCheckpoint.shared.container() })
    }
}

/// The two models of the suite, made once per test process; both share the loaded target.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private actor Inkling2MTPModels {
    /// The models of the test process.
    static let shared = Inkling2MTPModels()

    /// The model without MTP and the model with it, once the first test made them.
    private var models: (plain: MLXLanguageModel, mtp: MLXLanguageModel)?

    /// The model without MTP and the model with it.
    func pair() async throws -> (plain: MLXLanguageModel, mtp: MLXLanguageModel) {
        if let models { return models }
        let bytes = try await Inkling2MTPLanguageModel.snapshotSafetensorsBytes()
        let made = (
            plain: try Inkling2MTPLanguageModel.make(drafterBytes: nil),
            mtp: try Inkling2MTPLanguageModel.make(drafterBytes: bytes)
        )
        models = made
        return made
    }
}

// MARK: - Tests

/// The minutes the suite may run: two loads of the checkpoints and the alternating pairs of
/// every kind of call.
private let suiteTimeLimitMinutes = 60

/// Inkling 2 Mini with its DSpark head through `LanguageModelSession`: text, guided, tool-loop
/// and sampled calls each draft in every call, report no fallback, and write the output of the
/// same calls without MTP.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct Inkling2FoundationModelsMTPTests {

    /// The tokens each call may generate.
    private static let maxTokens = 256

    /// The alternating plain and MTP pairs of each kind of call.
    private static let pairCount = 2

    /// The prompt of the text calls.
    private static let textPrompt =
        "Write a detailed story about a lighthouse keeper and a storm, in three paragraphs."

    /// The prompt of the guided calls.
    private static let guidedPrompt = "Give the facts of the city of Lisbon."

    /// Greedy options with the call budget.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var options: GenerationOptions {
        GenerationOptions(samplingMode: .greedy, temperature: 0, maximumResponseTokens: maxTokens)
    }

    /// Runs `body` with an observer that splits what it reports into calls.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func observing<Value>(
        _ body: () async throws -> Value
    ) async throws -> (value: Value, calls: [Inkling2MTPCall], seconds: Double) {
        let log = Inkling2MTPCallLog()
        let observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void = {
            log.record($0)
        }
        let start = ContinuousClock.now
        let value = try await MLXLanguageModel.Executor.$generationObserver.withValue(observer) {
            try await body()
        }
        return (value, log.finished, (ContinuousClock.now - start).seconds)
    }

    /// The decode rate of the generation-loop passes of `calls`: their generated tokens over
    /// their decode seconds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func decodeRate(_ calls: [Inkling2MTPCall]) -> Double {
        let passes = calls.flatMap(\.completions)
        let tokens = passes.reduce(0) { $0 + $1.generationTokenCount }
        let seconds = passes.reduce(0) { $0 + $1.generateTime }
        return seconds > 0 ? Double(tokens) / seconds : 0
    }

    /// The rate of the guided calls `calls`, which took `seconds`: a guided pass reports no
    /// completion, so the rate is the generated tokens over the seconds of the calls, the
    /// prefill included.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func wallRate(_ calls: [Inkling2MTPCall], seconds: Double) -> Double {
        seconds > 0 ? Double(calls.reduce(0) { $0 + $1.generatedTokenCount }) / seconds : 0
    }

    /// Records an issue for each call of `calls` that did not draft or that fell back.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func expectEveryCallDrafts(_ calls: [Inkling2MTPCall], _ label: String) {
        #expect(!calls.isEmpty, "\(label): no call reported its MTP outcome")
        for (index, call) in calls.enumerated() {
            #expect(call.draftedTokenCount > 0, "\(label) call \(index): \(call.summary)")
            #expect(call.fallbackReason == nil, "\(label) call \(index): \(call.summary)")
        }
    }

    /// A text call drafts, reports no fallback and writes the plain text.
    @Test
    func aTextCallDraftsAndWritesThePlainText() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let models = try await Inkling2MTPModels.shared.pair()
            var rates: [(plain: Double, mtp: Double)] = []
            for pair in 0 ..< Self.pairCount {
                let plain = try await Self.observing {
                    try await LanguageModelSession(model: models.plain).respond(
                        to: Self.textPrompt, options: Self.options
                    ).content
                }
                let mtp = try await Self.observing {
                    try await LanguageModelSession(model: models.mtp).respond(
                        to: Self.textPrompt, options: Self.options
                    ).content
                }
                let rate = (plain: Self.decodeRate(plain.calls), mtp: Self.decodeRate(mtp.calls))
                rates.append(rate)
                Inkling2ResultLog.report(
                    "INKLING2 FM MTP",
                    "text pair \(pair): plain \(rate.plain) tok/s, MTP \(rate.mtp) tok/s; "
                        + "\(mtp.calls.map(\.summary))")
                Self.expectEveryCallDrafts(mtp.calls, "text pair \(pair)")
                #expect(mtp.value == plain.value, "text pair \(pair): <<<\(mtp.value)>>>")
            }
            Inkling2ResultLog.report("INKLING2 FM MTP", "text rates \(rates)")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// A `@Generable` call drafts under the grammar, reports no fallback and writes the plain
    /// output.
    @Test
    func aGuidedCallDraftsAndWritesThePlainOutput() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let models = try await Inkling2MTPModels.shared.pair()
            var rates: [(plain: Double, mtp: Double)] = []
            for pair in 0 ..< Self.pairCount {
                let plain = try await Self.observing {
                    try await LanguageModelSession(model: models.plain).respond(
                        to: Self.guidedPrompt, generating: Inkling2CityFacts.self,
                        options: Self.options
                    ).rawContent.jsonString
                }
                let mtp = try await Self.observing {
                    try await LanguageModelSession(model: models.mtp).respond(
                        to: Self.guidedPrompt, generating: Inkling2CityFacts.self,
                        options: Self.options
                    ).rawContent.jsonString
                }
                // The schema of each call orders its properties anew, so the two calls compare
                // as JSON objects.
                let sameObject =
                    Self.jsonObject(mtp.value) == Self.jsonObject(plain.value)
                let rate = (
                    plain: Self.wallRate(plain.calls, seconds: plain.seconds),
                    mtp: Self.wallRate(mtp.calls, seconds: mtp.seconds)
                )
                rates.append(rate)
                Inkling2ResultLog.report(
                    "INKLING2 FM MTP",
                    "guided pair \(pair): plain \(rate.plain) tok/s, MTP \(rate.mtp) tok/s "
                        + "(call wall time); \(mtp.calls.map(\.summary)) <<<\(mtp.value)>>>")
                Self.expectEveryCallDrafts(mtp.calls, "guided pair \(pair)")
                #expect(
                    sameObject, "guided pair \(pair): <<<\(mtp.value)>>> <<<\(plain.value)>>>")
            }
            Inkling2ResultLog.report("INKLING2 FM MTP", "guided rates \(rates)")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// Each call of a two-step tool loop and of its follow-up turn drafts, reports no fallback
    /// and writes the plain output, and the calls after the first reuse the prompt cache of the
    /// session with MTP on.
    @Test
    func everyCallOfAToolLoopDraftsAndReusesThePromptCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let models = try await Inkling2MTPModels.shared.pair()
            var rates: [(plain: Double, mtp: Double)] = []
            for pair in 0 ..< Self.pairCount {
                let plain = try await Self.toolLoop(models.plain)
                let mtp = try await Self.toolLoop(models.mtp)
                let rate = (plain: Self.decodeRate(plain.calls), mtp: Self.decodeRate(mtp.calls))
                rates.append(rate)
                Inkling2ResultLog.report(
                    "INKLING2 FM MTP",
                    "tool pair \(pair): plain \(rate.plain) tok/s, MTP \(rate.mtp) tok/s; "
                        + "calls \(mtp.value.calls); \(mtp.calls.map(\.summary)) "
                        + "<<<\(mtp.value.answers)>>>")
                Self.expectEveryCallDrafts(mtp.calls, "tool pair \(pair)")
                #expect(mtp.value.loop, "tool pair \(pair): calls \(mtp.value.calls)")
                #expect(mtp.value.answers == plain.value.answers, "tool pair \(pair)")
                for (index, call) in mtp.calls.enumerated().dropFirst() {
                    #expect(
                        call.cachedTokenCount > 0,
                        "tool pair \(pair) call \(index) reused no prompt token: \(call.summary)")
                }
            }
            Inkling2ResultLog.report("INKLING2 FM MTP", "tool rates \(rates)")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// The JSON object of `json`, comparable whatever the order of its keys.
    private static func jsonObject(_ json: String) -> NSDictionary? {
        (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? NSDictionary
    }

    /// With no sampling mode, as the Router makes its sessions, a text call and every call of
    /// a tool loop sample at the default temperature, draft, and report no fallback.
    @Test
    func sampledCallsDraft() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let models = try await Inkling2MTPModels.shared.pair()
            let options = GenerationOptions(maximumResponseTokens: Self.maxTokens)
            var rates: [(plain: Double, mtp: Double)] = []
            for pair in 0 ..< Self.pairCount {
                let plain = try await Self.observing {
                    try await LanguageModelSession(model: models.plain).respond(
                        to: Self.textPrompt, options: options
                    ).content
                }
                let mtp = try await Self.observing {
                    try await LanguageModelSession(model: models.mtp).respond(
                        to: Self.textPrompt, options: options
                    ).content
                }
                let plainTools = try await Self.toolLoop(models.plain, options: options)
                let mtpTools = try await Self.toolLoop(models.mtp, options: options)
                let plainCalls = plain.calls + plainTools.calls
                let mtpCalls = mtp.calls + mtpTools.calls
                rates.append((Self.decodeRate(plainCalls), Self.decodeRate(mtpCalls)))
                Inkling2ResultLog.report(
                    "INKLING2 FM MTP",
                    "sampled pair \(pair): text plain \(Self.decodeRate(plain.calls)) / MTP "
                        + "\(Self.decodeRate(mtp.calls)) tok/s, tools plain "
                        + "\(Self.decodeRate(plainTools.calls)) / MTP "
                        + "\(Self.decodeRate(mtpTools.calls)) tok/s; "
                        + "\(mtpCalls.map(\.summary))")
                Self.expectEveryCallDrafts(mtpCalls, "sampled pair \(pair)")
            }
            Inkling2ResultLog.report("INKLING2 FM MTP", "sampled rates \(rates)")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// The two-step loop and its follow-up turn in one session of `model`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func toolLoop(
        _ model: MLXLanguageModel,
        options: GenerationOptions = Inkling2FoundationModelsMTPTests.options
    ) async throws -> (
        value: (answers: [String], calls: String, loop: Bool), calls: [Inkling2MTPCall],
        seconds: Double
    ) {
        let log = Inkling2ToolCallLog()
        let session = LanguageModelSession(
            model: model, tools: [FindCustomerTool(log: log), GetBalanceTool(log: log)],
            instructions: Inkling2BankFixture.instructions)
        return try await observing {
            let answer = try await session.respond(
                to: Inkling2BankFixture.question, options: options
            ).content
            let followUp = try await session.respond(
                to: Inkling2BankFixture.followUp, options: options
            ).content
            return ([answer, followUp], "\(log.calls)", log.isTheTwoStepLoop)
        }
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
