// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of guided (`@Generable`) generation of Inkling 2 Mini through
// `MLXLanguageModel` and `LanguageModelSession`, without MTP: the Router "rank" call is such
// a pass, a selection among candidates with a reason.
//
// Each test logs the tokens the model sampled under the grammar and the tokens the grammar
// forced, and checks that the string values are sentences of the answer and not JSON debris
// (`"}"`), and that the selection is the candidate a clear prompt names.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2GuidedGenerationTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Tokenizers
import MLXLMCommon
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

// MARK: - The guided output

/// The candidate tools of the ranker.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
enum Inkling2RankedTool: String {
    case readFile
    case sendEmail
    case runTests
    case searchWeb
    case resizeImage
}

/// The output of the ranker: the pick, why, and what the task asks.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct Inkling2ToolRanking {
    @Guide(description: "What the task asks for, in one short sentence")
    var task: String

    @Guide(description: "Why the chosen tool fits the task, in one short sentence")
    var reason: String

    @Guide(description: "The tool that fits the task best")
    var selection: Inkling2RankedTool
}

// MARK: - Tests

@Suite(.serialized, .timeLimit(.minutes(30)))
struct Inkling2GuidedGenerationTests {

    /// The tokens a guided call may generate.
    private static let maxTokens = 256

    /// The prompts of the ranker, each with the candidate it names.
    private static let rankings: [(prompt: String, expected: String)] = [
        (
            "Pick the tool for this task: run the unit tests of the Swift package and report "
                + "which tests fail.", "runTests"
        ),
        (
            "Pick the tool for this task: email the quarterly report to the finance team.",
            "sendEmail"
        ),
    ]

    /// String values that are JSON debris and not text.
    private static let debris: Set<String> = ["", "}", "{", "\"", ",", "]", "["]

    /// The model without MTP and without reasoning, as the Router makes it.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func model() -> MLXLanguageModel {
        MLXLanguageModel(
            configuration: inkling2Mini4Bit, capabilities: [.guidedGeneration, .toolCalling],
            weightsLocation: {
                testWeightsLocation(modelID: $0, revision: inkling2Mini4BitRevision)
            },
            load: { _, _ in try await Inkling2MiniCheckpoint.shared.container() })
    }

    /// The text of `tokenIDs` in the tokenizer of the checkpoint, special tokens kept.
    private static func decode(_ tokenIDs: [Int]) async throws -> String {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        return await container.perform { context in
            context.tokenizer.decode(tokenIds: tokenIDs, skipSpecialTokens: false)
        }
    }

    /// A greedy ranker call picks the candidate the prompt names and writes sentences.
    @Test
    func aGuidedRankingPicksTheNamedToolAndWritesText() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let model = Self.model()
            for ranking in Self.rankings {
                let sink = GuidedGenerationDiagnosticSink()
                let response = try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
                    try await LanguageModelSession(model: model).respond(
                        to: ranking.prompt, generating: Inkling2ToolRanking.self,
                        options: GenerationOptions(
                            samplingMode: .greedy, maximumResponseTokens: Self.maxTokens))
                }
                let sampled = try await Self.decode(sink.sampledTokenIDs)
                let line =
                    "sampled \(sink.sampledTokenIDs.prefix(32)) <<<\(sampled)>>> forced "
                    + "\(sink.fastForwardTokenIDs.count) output "
                    + "<<<\(response.rawContent.jsonString)>>>"
                Inkling2ResultLog.report("INKLING2 GUIDED", line)
                print("[INKLING2 GUIDED] \(line)")
                let output = response.content
                #expect(
                    !Self.debris.contains(output.task.trimmingCharacters(in: .whitespaces)),
                    "task <<<\(response.rawContent.jsonString)>>>")
                #expect(
                    !Self.debris.contains(output.reason.trimmingCharacters(in: .whitespaces)),
                    "reason <<<\(response.rawContent.jsonString)>>>")
                #expect(
                    output.selection.rawValue == ranking.expected,
                    "selection <<<\(response.rawContent.jsonString)>>>")
            }
        } else {
            Issue.record(.unsupportedSystem)
        }
    }
}

#endif  // FoundationModelsIntegration
