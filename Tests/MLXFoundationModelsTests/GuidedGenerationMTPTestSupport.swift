// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The schemas of the guided generation integration tests, which the loop tests decode on the
/// fixture: one activity, three activities, a day plan, and unbounded strings.
enum GuidedFixtureSchema: String, CaseIterable, Sendable {
    case activity
    case threeActivities
    case dayPlan
    case unbounded

    /// The activity object the first three schemas share.
    private static let activityDefinition = """
        "Activity": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["sightseeing", "foodAndDining", "shopping", "hotelAndLodging"]
                },
                "title": { "type": "string", "maxLength": 40 },
                "description": { "type": "string", "maxLength": 40 }
            },
            "required": ["type", "title", "description"],
            "additionalProperties": false
        }
        """

    /// The JSON schema.
    var json: String {
        switch self {
        case .activity:
            return """
                { "$defs": { \(Self.activityDefinition) }, "$ref": "#/$defs/Activity" }
                """
        case .threeActivities:
            return """
                {
                    "$defs": { \(Self.activityDefinition) },
                    "type": "array",
                    "items": { "$ref": "#/$defs/Activity" },
                    "minItems": 3,
                    "maxItems": 3
                }
                """
        case .dayPlan:
            return """
                {
                    "$defs": {
                        \(Self.activityDefinition),
                        "DayPlan": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string", "maxLength": 60 },
                                "destination": { "type": "string", "maxLength": 60 },
                                "activities": {
                                    "type": "array",
                                    "items": { "$ref": "#/$defs/Activity" },
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "required": ["title", "destination", "activities"],
                            "additionalProperties": false
                        }
                    },
                    "$ref": "#/$defs/DayPlan"
                }
                """
        case .unbounded:
            return """
                {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "summary": { "type": "string" },
                        "conclusion": { "type": "string" }
                    },
                    "required": ["title", "summary", "conclusion"],
                    "additionalProperties": false
                }
                """
        }
    }
}

/// The grammar of one schema over ``ToolCallAlphabetTokenizer``, with the rules of a guided
/// generation, for the tests that drive a draft chain without a model.
struct GuidedFixtureGrammar {

    /// The token budget of the rules.
    private static let maxTokens = 64

    /// The tokenizer the grammar reads.
    let tokenizer = ToolCallAlphabetTokenizer()

    /// The rules of a generation under the grammar.
    let rules: GuidedDecisionRules

    /// Compiles `schema`.
    ///
    /// - Parameter schema: A JSON schema.
    /// - Throws: The error of the grammar compiler.
    init(schema: String) throws {
        let parts = try GuidedFixtureRun.Grammar(schema: schema, tokenizer: tokenizer)
        rules = GuidedDecisionRules(
            constraint: parts.constraint, maxTokens: Self.maxTokens,
            vocabSize: parts.vocabSize, logitDim: Qwen35MTPFixture.vocabularySize,
            stopTokenIDs: Set([tokenizer.eosTokenId].compactMap { $0 }),
            unknownTokenID: tokenizer.unknownTokenId,
            completionReserve: 0, hardReserve: 0, closingBias: nil, whitespaceBias: nil,
            whitespaceTokenIDs: [])
    }

    /// Drafts each character of `characters` into `chain`, as a drafter's processor sees each
    /// step: the logits of the step favor the character, then the chain learns of it.
    ///
    /// - Parameters:
    ///   - characters: Characters of the tokenizer's alphabet.
    ///   - chain: The chain the drafts go into.
    /// - Throws: The error the chain met.
    func draft(_ characters: [Character], into chain: GuidedDraftChain) throws {
        var processor = chain.processor
        for character in characters {
            let token = try #require(ToolCallAlphabetTokenizer.alphabet.firstIndex(of: character))
            var logits = [Float](repeating: 0, count: Qwen35MTPFixture.vocabularySize)
            logits[token] = 1
            _ = processor.process(logits: MLXArray(logits)[.newAxis])
            processor.didSample(token: MLXArray([Int32(token)]))
        }
        if let failure = chain.failure {
            throw failure
        }
    }
}

/// What one guided generation on the fixture's target wrote.
struct GuidedFixtureRun {

    /// The token budget of each generation.
    static let maxTokens = 96

    /// The part of the budget whose closing bias nudges the generation to close.
    static let completionReserve = 48

    /// The part of the budget whose bias forces closing tokens.
    static let hardReserve = 16

    /// The text the generation emitted.
    let text: String

    /// The tokens the generation counted, or `nil` when it ran out of budget.
    let tokenCount: Int?

    /// Whether the generation ran out of budget before the grammar closed.
    let incomplete: Bool

    /// What the generation recorded.
    let sink: GuidedGenerationDiagnosticSink

    /// The grammar, the tokenizer vocabulary and the biases of one schema over a tokenizer.
    struct Grammar {
        let constraint: GrammarConstraint
        let vocabSize: Int
        let closingBias: MLXArray
        let whitespaceBias: MLXArray
        let whitespaceTokenIDs: Set<Int>

        /// Compiles `schema` over `tokenizer`, with fast-forward on.
        init(schema: String, tokenizer: any Tokenizer) throws {
            let vocabulary = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary.vocab, vocabType: vocabulary.vocabType,
                eosTokenId: Int32(tokenizer.eosTokenId ?? 0))
            constraint = try GrammarConstraint(
                tokenizer: grammarTokenizer, jsonSchema: schema, fastForward: true,
                hostTokenizer: tokenizer)
            vocabSize = Int(grammarTokenizer.vocabSize)
            closingBias = ClosingTokenBias.compute(
                tokenizer: tokenizer, eosTokenId: tokenizer.eosTokenId)
            (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(tokenizer: tokenizer)
        }
    }

    /// Runs one guided generation of `schema` after `prompt` on the fixture's target.
    ///
    /// - Parameters:
    ///   - schema: The JSON schema.
    ///   - prompt: The prompt tokens.
    ///   - fixture: The fixture whose target decodes, and whose drafter drafts.
    ///   - blockSize: The tokens of each MTP round, or `nil` to decode plainly.
    ///   - drafter: The drafter of the MTP rounds, or `nil` for the fixture's own.
    /// - Returns: What the generation wrote.
    /// - Throws: An error of the generation other than running out of budget.
    static func generate(
        schema: String, prompt: [Int], fixture: Qwen35MTPFixture, blockSize: Int?,
        drafter: (any MTPDrafterModel)? = nil
    ) throws -> GuidedFixtureRun {
        let tokenizer = ToolCallAlphabetTokenizer()
        var context = fixture.makeModelContext()
        context.tokenizer = tokenizer
        let grammar = try Grammar(schema: schema, tokenizer: tokenizer)
        let speculation = blockSize.map {
            MTPSpeculation(
                drafter: drafter ?? fixture.drafter, blockSize: $0, minimumDraftProbability: 0)
        }
        let sink = GuidedGenerationDiagnosticSink()
        var text = ""
        var tokenCount: Int?
        var incomplete = false
        do {
            tokenCount = try GuidedGenerationDiagnosticSink.$current.withValue(sink) {
                try GuidedGenerationLoop.run(
                    input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), context: context,
                    constraint: grammar.constraint, maxTokens: maxTokens,
                    vocabSize: grammar.vocabSize, completionReserve: completionReserve,
                    hardReserve: hardReserve, closingBias: grammar.closingBias,
                    whitespaceBias: grammar.whitespaceBias,
                    whitespaceTokenIDs: grammar.whitespaceTokenIDs, speculation: speculation
                ) { delta in
                    text += delta
                    return true
                }
            }
        } catch GuidedGenerationError.incompleteOutput {
            incomplete = true
        }
        return GuidedFixtureRun(
            text: text, tokenCount: tokenCount, incomplete: incomplete, sink: sink)
    }
}

/// The fixture's target as two FoundationModels models over ``ToolCallAlphabetTokenizer``: one
/// that decodes plainly, and one with the fixture's MTP drafter.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct GuidedExecutorModels {

    /// The tokens each turn may generate.
    static let responseTokenBudget = 64

    /// The greedy options of a schema turn.
    static let greedy = GenerationOptions(
        samplingMode: .greedy, maximumResponseTokens: responseTokenBudget)

    /// The tokenizer of both models.
    let tokenizer = ToolCallAlphabetTokenizer()

    /// The model that decodes plainly.
    let plain: MLXLanguageModel

    /// The model that speculates.
    let speculative: MLXLanguageModel

    /// What one model's turn streamed, and what its guided pass recorded.
    struct Turn {
        let turn: MTPTurn
        let sink: GuidedGenerationDiagnosticSink
    }

    /// Makes both models over the fixture's target.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the models available.
    ///   - capabilities: The capabilities both models declare.
    /// - Throws: The error of the fixture or of the speculative decoding.
    init(weights: URL, capabilities: [LanguageModelCapabilities.Capability]) throws {
        let fixture = try Qwen35MTPFixture()
        plain = fixture.makeLanguageModel(
            weights: weights, capabilities: capabilities, tokenizer: tokenizer)
        speculative = fixture.makeLanguageModel(
            weights: weights, capabilities: capabilities, tokenizer: tokenizer,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())
    }

    /// A transcript of one prompt of `prompt` tokens.
    func transcript(prompt: [Int]) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [
                        .text(
                            Transcript.TextSegment(
                                content: tokenizer.decode(
                                    tokenIds: prompt, skipSpecialTokens: false)))
                    ],
                    responseFormat: nil))
        ])
    }

    /// Runs `request` on the plain model, then on the speculative model.
    ///
    /// - Parameter request: The request of both passes.
    /// - Returns: What each pass streamed and recorded.
    /// - Throws: The error of an executor's initializer.
    func respond(
        to request: LanguageModelExecutorGenerationRequest
    ) async throws -> (plain: Turn, speculative: Turn) {
        (
            try await Self.respond(to: request, model: plain),
            try await Self.respond(to: request, model: speculative)
        )
    }

    /// Runs `request` on `model` with a diagnostic sink bound.
    private static func respond(
        to request: LanguageModelExecutorGenerationRequest, model: MLXLanguageModel
    ) async throws -> Turn {
        let sink = GuidedGenerationDiagnosticSink()
        let turn = try await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            try await MTPExecutorTurns.respond(to: request, model: model)
        }
        return Turn(turn: turn, sink: sink)
    }
}

#endif
