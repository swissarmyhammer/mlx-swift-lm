// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of guided (`@Generable`) generation of Inkling 2 Mini through
// `MLXLanguageModel` and `LanguageModelSession`, without MTP: the Router "rank" call is such
// a pass, a selection among candidates with a reason.
//
// Each test logs the tokens the model sampled under the grammar and the tokens the grammar
// forced, and checks that the string values are sentences of the answer and not JSON debris
// (`"}"`) or a placeholder (`"..."`), and that the selection is the candidate a clear prompt
// names. Two causes made that debris: the grammar started before the header of the answer frame
// (`<|message_model|><|content_text|>`), and the hard closing zone of the city schema covered
// the whole budget of 256 tokens.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2GuidedGenerationTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
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

/// The facts of a city, the schema of `Inkling2FoundationModelsMTPTests`, whose strings all
/// came out as `"}"` before the fix.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct Inkling2GuidedCityFacts {
    @Guide(description: "The name of the city")
    var name: String

    @Guide(description: "The country of the city")
    var country: String

    @Guide(description: "Three famous landmarks of the city", .count(3))
    var landmarks: [String]

    @Guide(description: "One sentence about the city")
    var summary: String
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

    /// The placeholder a model writes for a value it does not know, which is not text.
    private static let placeholder = "..."

    /// The special tokens of the header of the answer frame, in order.
    private static let answerHeaderTokens = ["<|message_model|>", "<|content_text|>"]

    /// The prompt of the city call.
    private static let cityPrompt = "Give the facts of the city of Lisbon."

    /// The country the city call must name.
    private static let cityCountry = "Portugal"

    /// Whether `value` is JSON debris or a placeholder: a value with no letter (`"}"`, `"]"`,
    /// `","`, the empty string) or the placeholder `"..."`.
    private static func isDebris(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        return trimmed == placeholder
            || !trimmed.unicodeScalars.contains(where: \.properties.isAlphabetic)
    }

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

    /// One greedy call of `model` for `prompt` under the schema of `type`, which logs the tokens
    /// the model sampled under the grammar, the count the grammar forced, and the output.
    ///
    /// - Returns: the output and its JSON.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respond<Output: Generable>(
        _ model: MLXLanguageModel, to prompt: String, generating type: Output.Type
    ) async throws -> (content: Output, json: String) {
        let sink = GuidedGenerationDiagnosticSink()
        let result: Result<LanguageModelSession.Response<Output>, any Error>
        do {
            result = .success(
                try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
                    try await LanguageModelSession(model: model).respond(
                        to: prompt, generating: type,
                        options: GenerationOptions(
                            samplingMode: .greedy, maximumResponseTokens: maxTokens))
                })
        } catch {
            result = .failure(error)
        }
        // The sink holds the tokens of a call that threw too, thus the log shows what the model
        // wrote before the error.
        let sampled = try await decode(sink.sampledTokenIDs)
        let json = (try? result.get())?.rawContent.jsonString ?? "\(result)"
        let line =
            "sampled \(sink.sampledTokenIDs.count) tokens <<<\(sampled)>>> forced "
            + "\(sink.fastForwardTokenIDs.count), budget hit \(sink.incompleteOutput), output "
            + "<<<\(json)>>>"
        Inkling2ResultLog.report("INKLING2 GUIDED", line)
        print("[INKLING2 GUIDED] \(line)")
        #expect(!sink.incompleteOutput, "the budget cut the output <<<\(json)>>>")
        return (try result.get().content, json)
    }

    /// The tokenizer of the checkpoint encodes the answer opening as the two special tokens of
    /// the header of the answer frame, and not as the characters of their names.
    @Test
    func theAnswerOpeningEncodesAsTheTwoHeaderTokens() async throws {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        let (encoded, header) = await container.perform { context in
            (
                context.tokenizer.encode(
                    text: InklingReasoningProtocol.answerOpening, addSpecialTokens: false),
                Self.answerHeaderTokens.map { context.tokenizer.convertTokenToId($0) }
            )
        }

        #expect(header.allSatisfy { $0 != nil }, "header ids \(header)")
        #expect(encoded == header.compactMap { $0 })
    }

    /// A greedy ranker call picks the candidate the prompt names and writes sentences.
    @Test
    func aGuidedRankingPicksTheNamedToolAndWritesText() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let model = Self.model()
            for ranking in Self.rankings {
                let output = try await Self.respond(
                    model, to: ranking.prompt, generating: Inkling2ToolRanking.self)
                #expect(!Self.isDebris(output.content.task), "task <<<\(output.json)>>>")
                #expect(!Self.isDebris(output.content.reason), "reason <<<\(output.json)>>>")
                #expect(
                    output.content.selection.rawValue == ranking.expected,
                    "selection <<<\(output.json)>>>")
            }
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// A greedy city call writes the facts of the city, not JSON debris.
    @Test
    func aGuidedCityCallWritesTheFacts() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let output = try await Self.respond(
                Self.model(), to: Self.cityPrompt, generating: Inkling2GuidedCityFacts.self)
            let facts = output.content
            let strings = [facts.name, facts.country, facts.summary] + facts.landmarks
            #expect(!strings.contains(where: Self.isDebris), "<<<\(output.json)>>>")
            #expect(facts.country.contains(Self.cityCountry), "<<<\(output.json)>>>")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }
}

#endif  // FoundationModelsIntegration
