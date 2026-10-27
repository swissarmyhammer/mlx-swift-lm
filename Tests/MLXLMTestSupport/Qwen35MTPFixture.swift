// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// A tiny Qwen 3.5 text target with a matching MTP drafter, on seeded random weights.
///
/// The target is a hybrid: one Gated DeltaNet (linear, recurrent) layer, then one
/// full-attention layer, as the layers of Qwen 3.8 alternate. The drafter is the stateful
/// `Qwen35MTPDraftModel` that the MTP iterator and `ChatSession` run for Qwen 3.8. Both models
/// take their weights from a seed, thus every run of a test decodes the same tokens.
///
/// ``DrafterMode`` picks a drafter the target mostly accepts or mostly rejects, so a test can
/// force accept rounds and reject rounds. The target is the same in both modes.
public struct Qwen35MTPFixture {

    /// How the drafter of the fixture proposes tokens.
    public enum DrafterMode: Sendable, CaseIterable {

        /// The drafter copies the full-attention layer of the target, and the target accepts
        /// most of its proposals.
        ///
        /// The Qwen MTP drafter shares the embedding and the LM head of its target. In this
        /// mode its input projection passes the embedding of the token through and drops the
        /// target hidden state, and its one layer takes the weights of the target's
        /// full-attention layer. Its cache holds the same tokens one position earlier, which
        /// rotary attention does not see. The drafter thus runs the target without the linear
        /// layer and without the first prompt token, and those two differences make the reject
        /// rounds.
        case copiesTarget

        /// The drafter has its own random weights, and the target rejects most of its
        /// proposals.
        case independent
    }

    /// The seed of the weights when a test names none.
    public static let defaultSeed: UInt64 = 35

    /// The number of tokens of the vocabulary.
    public static let vocabularySize = 64

    /// The width of the hidden state of the target and of the drafter. At 32 the greedy
    /// decode of the target does not fall into a short cycle, as it does at 16.
    public static let hiddenSize = 32

    /// The number of decoder layers of the target: one linear layer, then one full-attention
    /// layer.
    public static let hiddenLayers = 2

    /// One full-attention layer for each two layers, thus layer 0 is linear and layer 1 is
    /// attention.
    public static let fullAttentionInterval = 2

    /// The number of layers of the drafter.
    public static let mtpLayers = 1

    /// The tokens of each MTP round: one drafted token and the bonus token. The Qwen MTP
    /// drafter drafts one token for each round.
    public static let blockSize = 2

    /// The seed of the independent drafter is this far from the seed of the target, thus its
    /// weights differ from the target's.
    private static let drafterSeedOffset: UInt64 = 1

    /// The parameter key of the target's token embeddings.
    private static let embeddingKey = "model.embed_tokens.weight"

    /// The parameter key prefix of the target's full-attention layer, the last layer of the
    /// first interval.
    private static let targetAttentionLayerPrefix =
        "model.layers.\(fullAttentionInterval - 1)."

    /// The parameter key prefix of the drafter's layer.
    private static let drafterLayerPrefix = "mtp.layers.0."

    /// The parameter key of the drafter's input projection.
    private static let inputProjectionKey = "mtp.fc.weight"

    /// The configuration of the model, as a loaded model would carry it.
    public let modelConfiguration = ModelConfiguration(id: "tiny-qwen35-mtp")

    /// The target model.
    public let target: Qwen35TextModel

    /// The MTP drafter of the target.
    public let drafter: Qwen35MTPDraftModel

    /// The tokenizer of the target.
    public let tokenizer = ScalarTokenizer(vocabularySize: vocabularySize)

    /// Makes the target and the drafter from `seed`.
    ///
    /// - Parameters:
    ///   - mode: How the drafter proposes tokens.
    ///   - seed: The seed of the weights.
    /// - Throws: The error of the configuration decoder or of a weight update.
    public init(drafter mode: DrafterMode = .copiesTarget, seed: UInt64 = defaultSeed) throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: Self.mtpLayers, hiddenLayers: Self.hiddenLayers,
                    fullAttentionInterval: Self.fullAttentionInterval,
                    hiddenSize: Self.hiddenSize, vocabularySize: Self.vocabularySize,
                    tieWordEmbeddings: false
                ).utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35TextModel(configuration)
        }
        let drafter = withRandomState(
            MLXRandom.RandomState(seed: seed &+ Self.drafterSeedOffset)
        ) {
            Qwen35MTPDraftModel(configuration)
        }
        try Self.normalizeEmbeddings(of: target)
        if mode == .copiesTarget {
            try Self.copyAttentionLayer(of: target, into: drafter)
        }
        eval(target, drafter)
        self.target = target
        self.drafter = drafter
    }

    /// Scales each token embedding of `target` to a root mean square of one.
    ///
    /// The drafter normalizes the embedding it reads, and the target adds the raw embedding
    /// to its residual. With unit embeddings the two agree, which
    /// ``DrafterMode/copiesTarget`` relies on.
    private static func normalizeEmbeddings(of target: Qwen35TextModel) throws {
        let normalized = target.parameters().flattened()
            .filter { key, _ in key == embeddingKey }
            .map { key, embeddings in
                (
                    key,
                    embeddings
                        * rsqrt(
                            mean(square(embeddings), axis: -1, keepDims: true)
                                + Qwen35FixtureJSON.rmsNormEpsilon)
                )
            }
        try target.update(
            parameters: ModuleParameters.unflattened(normalized), verify: .noUnusedKeys)
    }

    /// Gives the layer of `drafter` the weights of the full-attention layer of `target`, and
    /// makes its input projection pass the token embedding through.
    private static func copyAttentionLayer(
        of target: Qwen35TextModel, into drafter: Qwen35MTPDraftModel
    ) throws {
        let layer = target.parameters().flattened()
            .filter { key, _ in key.hasPrefix(targetAttentionLayerPrefix) }
            .map { key, weight in
                (drafterLayerPrefix + key.dropFirst(targetAttentionLayerPrefix.count), weight)
            }
        let embeddingPassThrough = concatenated(
            [MLXArray.eye(hiddenSize), MLXArray.zeros([hiddenSize, hiddenSize])], axis: 1)
        try drafter.update(
            parameters: ModuleParameters.unflattened(
                layer + [(inputProjectionKey, embeddingPassThrough)]),
            verify: .noUnusedKeys)
    }

    /// A model context over the target, with the tokenizer and the input processor of the
    /// fixture.
    ///
    /// The tokenizer of the fixture has no end-of-sequence token, thus a generation runs until
    /// its token budget unless `eosTokenIds` names tokens that stop it.
    ///
    /// - Parameter eosTokenIds: The tokens that end a generation.
    /// - Returns: The context.
    public func makeModelContext(eosTokenIds: Set<Int> = []) -> ModelContext {
        var configuration = modelConfiguration
        configuration.eosTokenIds = eosTokenIds
        return ModelContext(
            configuration: configuration, model: target,
            processor: ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer)
    }

    /// A model container over ``makeModelContext()``.
    ///
    /// - Returns: The container.
    public func makeModelContainer() -> ModelContainer {
        ModelContainer(context: makeModelContext())
    }

    /// The MTP configuration of a `ChatSession`: the drafter, with rounds of ``blockSize``.
    ///
    /// - Returns: The configuration.
    /// - Throws: The error of the configuration when the drafter refuses the block size.
    public func makeSpeculativeDecodingConfig() throws -> SpeculativeDecodingConfig {
        try SpeculativeDecodingConfig(
            mtpDrafter: MTPDrafterContainer(
                context: MTPDrafterContext(configuration: modelConfiguration, model: drafter)),
            blockSize: Self.blockSize)
    }

    /// A greedy chat session over the target, with MTP or without it.
    ///
    /// - Parameters:
    ///   - speculative: Whether the session drafts with the drafter of the fixture.
    ///   - maxTokens: The tokens each turn generates.
    /// - Returns: The session.
    /// - Throws: The error of ``makeSpeculativeDecodingConfig()``.
    public func makeChatSession(speculative: Bool, maxTokens: Int) throws -> ChatSession {
        ChatSession(
            makeModelContext(),
            speculativeDecoding: speculative ? try makeSpeculativeDecodingConfig() : nil,
            generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
    }

    /// The text of `tokens`, which the tokenizer of the fixture encodes back to `tokens`.
    ///
    /// - Parameter tokens: Tokens of the vocabulary.
    /// - Returns: The text.
    public func text(for tokens: [Int]) -> String {
        tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
    }

    /// The tokens and the completion information of one greedy decode.
    public struct GreedyDecode: Sendable {

        /// The generated tokens, in order.
        public let tokens: [Int]

        /// The completion information of the decode.
        public let info: GenerateCompletionInfo
    }

    /// The error of a decode that ends without its completion information.
    public struct MissingCompletionInfo: Error {}

    /// Decodes `maxTokens` greedy tokens after `prompt` from fresh caches, with MTP or without
    /// it.
    ///
    /// - Parameters:
    ///   - prompt: The prompt tokens.
    ///   - maxTokens: The tokens to generate.
    ///   - speculative: Whether the decode drafts with the drafter of the fixture.
    /// - Returns: The generated tokens and the completion information.
    /// - Throws: The error of the token iterator, or ``MissingCompletionInfo``.
    public func decodeGreedily(prompt: [Int], maxTokens: Int, speculative: Bool) async throws
        -> GreedyDecode
    {
        let context = makeModelContext()
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        let stream =
            speculative
            ? try generateTokens(
                input: input, parameters: parameters, context: context, mtpDrafter: drafter,
                blockSize: Self.blockSize)
            : try generateTokens(input: input, parameters: parameters, context: context)
        return try await Self.collect(stream)
    }

    /// Reads every event of a token stream.
    ///
    /// - Parameter stream: The stream of one generation.
    /// - Returns: The generated tokens and the completion information.
    /// - Throws: ``MissingCompletionInfo`` when the stream ends without its completion
    ///   information.
    public static func collect(_ stream: AsyncStream<TokenGeneration>) async throws
        -> GreedyDecode
    {
        var tokens: [Int] = []
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let completion): info = completion
            }
        }
        guard let info else { throw MissingCompletionInfo() }
        return GreedyDecode(tokens: tokens, info: info)
    }
}
