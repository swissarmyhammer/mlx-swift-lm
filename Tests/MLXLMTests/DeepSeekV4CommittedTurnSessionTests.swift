// Copyright © 2026 Apple Inc.
//
// A weight-free `ChatSession` test of the DSML cache splice of card `^v7z7v99`.
//
// `DSMLCommittedTurnRuleTests` examines the decision table of the rule. This
// file examines the two facts the SESSION owns, which a pure rule test cannot
// reach: that the session records the render of each prefill, and that it hands
// that render to the policy on the next turn.
//
// The scripted tokenizer below writes the shape the published checkpoint writes.
// The model writes a body of its own and closes its turn with the commit marker;
// the render of that same turn writes a DIFFERENT body — the real checkpoint
// abbreviates its DSML closing tags and splits ordinary words into other tokens
// — and then the same commit marker. Thus the ledger of the session stops being
// a prefix of the next render, exactly as the real weights show.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Examines the DSML splice through a real ``ChatSession``.
final class DeepSeekV4CommittedTurnSessionTests: XCTestCase {

    /// The token identifiers of the scripted DeepSeek-V4 conversation.
    private enum Token {
        /// Opens a user turn.
        static let user = 0
        /// Opens an assistant turn, and ends a render as the generation tail.
        static let assistant = 1
        /// Closes an assistant turn. Stands for `<｜end▁of▁sentence｜>`.
        static let commit = 2
        /// The text of the first user turn.
        static let firstQuestion = 8
        /// The text of the second user turn.
        static let secondQuestion = 9
        /// The body the model writes in its first turn.
        static let generatedBody = 10
        /// The body the RENDER writes for that same turn. It differs from
        /// ``generatedBody``, which is what makes the ledger unreproducible.
        static let renderedBody = 11
        /// The body the model writes in its second turn.
        static let secondGeneratedBody = 12
        /// One past the largest identifier above.
        static let count = 13
    }

    /// The render of round 1: the user turn and the generation tail.
    private static let firstRender = [Token.user, Token.firstQuestion, Token.assistant]

    /// The tokens the model writes in round 1, closing its turn.
    private static let firstGeneration = [Token.generatedBody, Token.commit]

    /// The render of round 2. It holds ``firstRender`` whole, then the render's
    /// own version of the committed assistant turn, then the new user turn.
    private static let secondRender =
        firstRender + [Token.renderedBody, Token.commit]
        + [Token.user, Token.secondQuestion, Token.assistant]

    /// The tokens the session must feed in round 2: the new user turn and the
    /// generation tail, and nothing the cache already holds.
    private static let secondSuffixLength = 3

    /// A tokenizer that renders the scripted conversation and resolves the
    /// commit marker.
    ///
    /// The conformance is `@unchecked` because it keeps NO stored property.
    /// Every member is a computed property or a pure function that reads the
    /// immutable static tokens of the enclosing suite, thus no task can observe
    /// a mutation and no synchronization is necessary.
    private final class ScriptedTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        var vocabularySize: Int { Token.count }
        var bosToken: String? { nil }
        /// The generation stops on this marker. `Tokenizer.eosTokenId` reads it
        /// through ``convertTokenToId(_:)``.
        var eosToken: String? { DeepSeekV4ChatEncoder.SpecialToken.endOfSentence }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            let userTurns = messages.count { ($0["role"] as? String) == "user" }
            return userTurns > 1
                ? DeepSeekV4CommittedTurnSessionTests.secondRender
                : DeepSeekV4CommittedTurnSessionTests.firstRender
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { $0 == Token.commit ? "" : "text" }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            token == DeepSeekV4ChatEncoder.SpecialToken.endOfSentence ? Token.commit : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            id == Token.commit ? DeepSeekV4ChatEncoder.SpecialToken.endOfSentence : "text"
        }
    }

    /// A model that writes one scripted token for each forward pass and records
    /// what each pass received.
    ///
    /// The conformance is `@unchecked` because ``passes`` and `step` are mutable
    /// and carry no lock. The invariant that makes it safe: one ``ChatSession``
    /// owns this model, and a session runs its forward passes ONE AT A TIME, thus
    /// every mutation happens inside ``callAsFunction(_:cache:)`` and no two
    /// calls overlap. The test reads ``passes`` only after both `respond(to:)`
    /// calls return, thus the read never races a write.
    private final class ScriptedModel: Module, LLMModel, KVCacheDimensionProvider,
        @unchecked Sendable
    {
        /// What one forward pass received.
        struct Pass {
            /// The position the cache held when the pass began.
            var offset: Int
            /// The number of tokens the pass fed.
            var tokenCount: Int
        }

        let kvHeads: [Int] = [1]
        let vocabularySize: Int
        /// Every forward pass, in order.
        private(set) var passes: [Pass] = []

        private let script: [Int]
        private var step = 0

        /// - Parameters:
        ///   - script: one token for each forward pass, in order.
        ///   - vocabularySize: the size of the scripted vocabulary.
        init(script: [Int], vocabularySize: Int) {
            self.script = script
            self.vocabularySize = vocabularySize
            super.init()
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let tokenCount = inputs.dim(-1)
            passes.append(Pass(offset: cache?.first?.offset ?? 0, tokenCount: tokenCount))

            let entry = MLXArray.zeros([1, 1, tokenCount, 1])
            _ = cache?.first?.update(keys: entry, values: entry)

            let token = step < script.count ? script[step] : Token.commit
            step += 1
            var row = [Float](repeating: -30, count: vocabularySize)
            row[token] = 30
            let logits = MLXArray(row, [1, 1, vocabularySize])
            if tokenCount == 1 {
                return logits
            }
            return concatenated(Array(repeating: logits, count: tokenCount), axis: 1)
        }

        var toolCallFormat: ToolCallFormat? { .dsml }
        var loraLayers: [Module] { [] }
    }

    /// The session feeds only the new tail of round 2, spliced onto the tokens
    /// the model wrote.
    ///
    /// Card `^v7z7v99` measured the failure this guards on
    /// `mlx-community/DeepSeek-V4-Flash-4bit`: the round after a committed turn
    /// fed every one of its rendered tokens and skipped none, because the render
    /// cannot write the tokens the model wrote.
    func testTheRoundAfterACommittedTurnFeedsOnlyItsNewTail() async throws {
        let tokenizer = ScriptedTokenizer()
        let configuration = ModelConfiguration(id: "deepseek-v4-test", toolCallFormat: .dsml)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let model = ScriptedModel(
            script: Self.firstGeneration + [Token.secondGeneratedBody, Token.commit],
            vocabularySize: tokenizer.vocabularySize)
        let context = ModelContext(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 8, temperature: 0))

        _ = try await session.respond(to: "first")
        _ = try await session.respond(to: "second")

        let ledgerLength = Self.firstRender.count + Self.firstGeneration.count
        XCTAssertTrue(
            model.passes.contains {
                $0.offset == ledgerLength && $0.tokenCount == Self.secondSuffixLength
            },
            """
            round 2 must feed its \(Self.secondSuffixLength)-token tail onto the \
            \(ledgerLength) tokens the cache holds. The passes were \(model.passes).
            """)
    }
}
