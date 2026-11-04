// Copyright © 2026 Apple Inc.
//
// The router path loads Qwen3.8 as the Qwen 3.5 vision-language model. A
// speculative verify of 3 to 8 tokens there must read its key cache through
// ``VerifyAttentionKernel``, as it does in the text model: the decoder gives
// each full-attention layer the causal mask mode, which the kernel reads, and
// not a causal mask array, which only the SDPA of MLX reads.
//
// The attention layer gives the kernel output bit for bit with the causal
// mode, and stays inside the bfloat16 tolerance of the causal mask array it
// read before. The verify forward of the whole model still gives the logits
// that plain decode gives for the same tokens, one at a time, inside the
// bfloat16 tolerance, and the same greedy token in each row up to a near tie.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Qwen35VLMVerifyAttentionTests {

    /// The number of query heads: six for each KV head, as in Qwen3.8.
    private static let queryHeads = 6

    /// The number of KV heads.
    private static let kvHeads = 1

    /// The width of one attention head: the only width the kernel reads.
    private static let headDim = 256

    /// The width of the residual stream.
    private static let hiddenSize = 32

    /// The number of tokens of the vocabulary.
    private static let vocabularySize = 32

    /// The number of random tensors of a filled cache: the keys and the
    /// values.
    private static let keysAndValues = 2

    /// The dtype of the weights, the tokens and the cache, as the published
    /// checkpoints hold them.
    private static let dtype = DType.bfloat16

    /// The cached keys ahead of the verify: the fewest keys the kernel reads,
    /// and some more.
    private static let cachedKeys = DecodeAttentionKernel.fewestKeys + 76

    /// The tokens of the model verify: the bonus token and four drafts, a
    /// block of five.
    private static let modelVerifyTokens = 5

    /// Each verify size the kernel reads.
    private static let kernelVerifyTokenCounts = Array(
        VerifyAttentionKernel.fewestQueries ... VerifyAttentionKernel.mostQueries)

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 20_261_030

    /// The seed of the filled cache.
    private static let cacheSeed: UInt64 = 20_261_031

    /// The seed of the new tokens of a step, or of the token ids of the model
    /// test.
    private static let tokenSeed: UInt64 = 20_261_032

    /// A small dense Qwen 3.5 text configuration with one full-attention
    /// layer: layer 1 of an interval of 2.
    private static let textConfigurationJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": \(hiddenSize),
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": \(queryHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "linear_num_value_heads": 2,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": \(Qwen35FixtureJSON.rmsNormEpsilon),
          "vocab_size": \(vocabularySize),
          "tie_word_embeddings": true,
          "full_attention_interval": 2,
          "rope_parameters": {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25
          }
        }
        """

    /// The new tokens of one attention step, before and after the rotary
    /// position.
    private struct Step {
        /// The input rows, shape `(1, tokens, hiddenSize)`.
        let x: MLXArray
        /// The M-RoPE positions of the tokens, after the cached keys.
        let positionIds: MLXArray
        /// The queries at their rotary positions.
        let rotatedQueries: MLXArray
        /// The output gate of the layer.
        let gate: MLXArray
    }

    /// A cache of the model after its prompt, and the state the prompt left.
    private struct Prefilled {
        /// The caches of the layers.
        let cache: [KVCache?]
        /// The state of the prompt call, which holds the rope deltas.
        let state: LMOutput.State
    }

    // MARK: - The attention layer

    /// A verify of each size the kernel reads gives the kernel output with
    /// the causal mask mode, bit for bit.
    @Test(arguments: kernelVerifyTokenCounts)
    func theAttentionLayerGivesTheVerifyKernelOutputWithTheCausalMode(tokens: Int) throws {
        let attention = try Self.attention()
        let cache = Self.filledCache()
        let step = Self.step(attention, tokens: tokens, cache: cache)

        let output = attention(step.x, mask: .causal, cache: cache, positionIds: step.positionIds)
        let state = cache.state
        let operands = AttentionKernelOperands(
            queries: step.rotatedQueries, keys: state[0], values: state[1], scale: attention.scale)
        try #require(
            operands.runs {
                VerifyAttentionKernel.canRun(queries: $0, keys: $1, values: $2, mask: .causal)
            }, "the verify kernel must read \(operands.testDescription)")
        let expected = attention.mergeHeadsAndProject(
            attention: operands.expectedRouteOutput(
                with: .causal, picksFrom: VerifyAttentionKernelTests.kernels),
            gate: step.gate)

        Qwen35AttentionLayerFixture.expectTheStep(
            output, gives: expected, cache: cache, keys: Self.cachedKeys + tokens)
    }

    /// A verify of each size the kernel reads gives, with the causal mask
    /// mode, the output of the causal mask array inside the bfloat16
    /// tolerance.
    @Test(arguments: kernelVerifyTokenCounts)
    func theCausalModeGivesTheOutputOfTheCausalMaskArray(tokens: Int) throws {
        let attention = try Self.attention()
        let causalCache = Self.filledCache()
        let arrayCache = Self.filledCache()
        let step = Self.step(attention, tokens: tokens, cache: causalCache)

        let causal = attention(
            step.x, mask: .causal, cache: causalCache, positionIds: step.positionIds)
        let array = attention(
            step.x, mask: .array(createCausalMask(n: tokens, offset: Self.cachedKeys)),
            cache: arrayCache, positionIds: step.positionIds)

        let gap = AttentionKernelTolerance.worstGap(causal, array)
        #expect(gap <= AttentionKernelTolerance.bfloat16, "worst gap \(gap)")
    }

    // MARK: - The model

    /// The verify forward of a bfloat16 model over a long cache, as the MTP
    /// iterator runs it, gives the logits of plain decode over the same
    /// tokens inside the bfloat16 tolerance, and the greedy token of plain
    /// decode in each row up to a near tie.
    @Test func theVerifyForwardGivesThePlainDecodeLogits() throws {
        let model = try Self.model()
        let tokens = MLXRandom.randInt(
            0 ..< Self.vocabularySize, [1, Self.cachedKeys + Self.modelVerifyTokens],
            key: MLXRandom.key(Self.tokenSeed))
        let context = tokens[0..., ..<Self.cachedKeys]
        let draft = tokens[0..., Self.cachedKeys...]

        let verifyPrompt = try Self.prefilled(model, context)
        let decodePrompt = try Self.prefilled(model, context)
        let verify = VerifyForwardLogits.verify(
            draft, after: verifyPrompt.state, forward: Self.forward(model, over: verifyPrompt))
        let decode = VerifyForwardLogits.decode(
            draft, after: decodePrompt.state, forward: Self.forward(model, over: decodePrompt))

        let gap = AttentionKernelTolerance.worstGap(verify, decode)
        #expect(gap <= AttentionKernelTolerance.bfloat16, "worst gap \(gap)")
        AttentionKernelTolerance.expectTheGreedyTokensUpToANearTie(verify: verify, decode: decode)
    }

    // MARK: - Fixture

    /// The text configuration of every module of these tests.
    private static func textConfiguration() throws -> Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(textConfigurationJSON.utf8))
    }

    /// The attention of the full-attention layer, on seeded weights in
    /// ``dtype``.
    private static func attention() throws -> Qwen35Language.Attention {
        let configuration = try textConfiguration()
        return withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            let attention = Qwen35Language.Attention(configuration)
            attention.update(parameters: attention.parameters().mapValues { $0.asType(dtype) })
            eval(attention)
            return attention
        }
    }

    /// The vision-language model of the text configuration, on seeded
    /// weights in ``dtype``.
    private static func model() throws -> Qwen35 {
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: textConfigurationJSON).utf8))
        return withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            let model = Qwen35(configuration)
            model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
            eval(model)
            return model
        }
    }

    /// A key cache that holds ``cachedKeys`` random keys and values.
    private static func filledCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        let rows = MLXRandom.normal(
            [keysAndValues, 1, kvHeads, cachedKeys, headDim], key: MLXRandom.key(cacheSeed)
        ).asType(dtype)
        _ = cache.update(keys: rows[0], values: rows[1])
        eval(cache.state)
        return cache
    }

    /// The random new tokens of a step after the cache, with their rotated
    /// queries. Call it before the step writes its keys, while the cache
    /// offset is the position of the first new token.
    ///
    /// - Parameters:
    ///   - attention: The attention of the layer.
    ///   - tokens: The number of new tokens.
    ///   - cache: The cache before the step.
    /// - Returns: The step.
    private static func step(
        _ attention: Qwen35Language.Attention, tokens: Int, cache: KVCacheSimple
    ) -> Step {
        let x = MLXRandom.normal([1, tokens, hiddenSize], key: MLXRandom.key(tokenSeed))
            .asType(dtype)
        let (queries, gate, keys, values) = attention.projectPreRope(x)
        let positionIds = qwen35MTPPositionIds(
            offset: cache.offset, length: tokens, batchSize: 1, positionDeltas: nil)
        let (cosValues, sinValues) = attention.rotaryEmbedding(x: values, positionIds: positionIds)
        let (rotatedQueries, _) = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: queries, k: keys, cos: cosValues, sin: sinValues)
        return Step(x: x, positionIds: positionIds, rotatedQueries: rotatedQueries, gate: gate)
    }

    /// The caches of `model` after `context`, and the state the prompt left.
    private static func prefilled(_ model: Qwen35, _ context: MLXArray) throws -> Prefilled {
        let cache: [KVCache?] = try model.newCache(parameters: nil)
        let prompt = model.languageModel(context, cache: cache, state: nil)
        eval(prompt.logits)
        return Prefilled(cache: cache, state: prompt.state ?? LMOutput.State())
    }

    /// One forward of the language model of `model` over the caches of
    /// `prefilled`.
    private static func forward(
        _ model: Qwen35, over prefilled: Prefilled
    ) -> VerifyForwardLogits.Forward {
        { tokens, state in model.languageModel(tokens, cache: prefilled.cache, state: state) }
    }
}
