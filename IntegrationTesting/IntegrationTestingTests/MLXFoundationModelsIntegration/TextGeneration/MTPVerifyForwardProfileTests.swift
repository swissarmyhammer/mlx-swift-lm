// Copyright © 2026 Apple Inc.
//
// Where the time of one MTP verify forward of Qwen3.8-27B-4bit goes at 32k and at 100k tokens of
// context, by layer kind, at each row count of a verify. `MTPRoundCostBreakdownTests` saw each row
// past the first add 5-7 ms at 32k and 10-13 ms at 100k, while the weights are read once for all
// rows. The profile times, over every layer of a kind, each with its own weights and its own cache:
//
//  - the gated delta net mixers (their projections, and their recurrence apart),
//  - the q/k/v/o projections of the full-attention layers,
//  - the attention of the full-attention layers over their caches, on the route of a verify, and
//    on the route of the router's load path, whose mask is an array,
//  - the feed-forward blocks,
//
// and the whole verify forward, whose rest is the embedding, the norms and the vocabulary
// projection. Run it in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPVerifyForwardProfileTests`
//
// Every measurement line goes to the unified log under the `MTP VERIFY PROFILE:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPVerifyProfile"'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXNN
import Testing
import os

@testable import MLXFoundationModels
@testable import MLXLLM
@testable import MLXLMCommon

/// The time limit of the test, in minutes: two prefills of 32k and 100k tokens, and a few hundred
/// timed forwards and layer passes.
private let verifyProfileTimeLimitMinutes = 60

/// The MLX language model, apart from the FoundationModels one of the same name.
private typealias TargetModel = MLXLMCommon.LanguageModel

/// The time of one verify forward and of each layer kind in it, at one row count, in milliseconds.
private struct VerifyLayerKindCost: Sendable {

    /// The tokens the verify reads.
    let rows: Int

    /// The whole verify forward, with its logits and its emitted hidden rows.
    let verify: Double

    /// The gated delta net mixers: their projections and their recurrence.
    let gatedDeltaNet: Double

    /// The input and output projections of the gated delta net mixers.
    let gatedDeltaProjections: Double

    /// The q/k/v/o projections of the full-attention layers.
    let attentionProjections: Double

    /// The attention of the full-attention layers over their caches, on the route of a verify.
    let attention: Double

    /// The same attention through the SDPA of MLX with the causal mask as an array, the route
    /// of the router's load path.
    let maskArrayAttention: Double

    /// The feed-forward blocks of every layer.
    let feedForward: Double

    /// The recurrence of the gated delta net mixers: the conv, the norms, the delta rule.
    var gatedDeltaRecurrence: Double { gatedDeltaNet - gatedDeltaProjections }

    /// The rest of the verify forward: the embedding, the norms and the vocabulary projection.
    var rest: Double { verify - gatedDeltaNet - attentionProjections - attention - feedForward }

    /// The parts in a measurement line.
    var summary: String {
        let parts: [(String, Double)] = [
            ("verify", verify), ("GDN", gatedDeltaNet), ("GDN projections", gatedDeltaProjections),
            ("GDN recurrence", gatedDeltaRecurrence),
            ("attention projections", attentionProjections), ("attention", attention),
            ("mask-array attention", maskArrayAttention), ("feed-forward", feedForward),
            ("rest", rest),
        ]
        return "rows \(rows): "
            + parts.map { "\($0.0) \(String(format: "%.2f", $0.1)) ms" }.joined(separator: ", ")
    }
}

/// The layers of the target by kind, each with its cache.
private struct LayersByKind {

    /// The gated delta net mixers and their caches.
    let gatedDeltaNets: [(mixer: Qwen35GatedDeltaNet, cache: MambaCache)]

    /// The full-attention layers, their attention modules and their caches.
    let attentionLayers: [(layer: Qwen35DecoderLayer, attention: Qwen35Attention, cache: KVCache)]

    /// Every decoder layer.
    let all: [Qwen35DecoderLayer]

    /// Splits the layers of `model` and the caches of `cache` by kind.
    init(_ model: Qwen35TextModel, cache: [KVCache]) {
        let layers = model.model.layers
        all = layers
        gatedDeltaNets = zip(layers, cache).compactMap { layer, cache in
            guard let mixer = layer.linearAttn, let cache = cache as? MambaCache else {
                return nil
            }
            return (mixer, cache)
        }
        attentionLayers = zip(layers, cache).compactMap { layer, cache in
            layer.selfAttn.map { (layer, $0, cache) }
        }
    }
}

/// Times one verify forward of Qwen3.8-27B-4bit at 32k and 100k tokens by layer kind.
///
/// The target loads on the text path, whose verify runs through the compiled verify schedule.
/// The layer passes run inside the speculative-verify scope, thus each projection takes the
/// multi-row matvec of a verify. Each part is the median of ``timedRepeats`` runs, each with a GPU
/// sync before and after it. The verify forward appends to the cache, thus the context grows by
/// a few tokens over the runs, which is noise next to 32k.
@Suite(.serialized, .timeLimit(.minutes(verifyProfileTimeLimitMinutes)))
struct MTPVerifyForwardProfileTests {

    /// The context lengths of the profile, in tokens.
    private static let contextTokenCounts = [32_768, 100_000]

    /// The tokens of each chunk of the prefill.
    private static let prefillChunkTokenCount = 2_048

    /// The most history rounds the context may draw on.
    private static let maximumHistoryRoundCount = 512

    /// The runs of each timed part.
    private static let timedRepeats = 8

    /// The seed of the random queries of the attention passes.
    private static let querySeed: UInt64 = 20_261_030

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP VERIFY PROFILE:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPVerifyProfile")

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: Qwen38FourBitMTPModel.isLocal))
    func verifyForwardByLayerKindAtThirtyTwoAndOneHundredThousandTokens() async throws {
        await releaseAllGPUMemory()
        let mtpModel = try Qwen38FourBitMTPModel.make(
            speculates: true, path: .text, adaptivePolicy: .disabled)
        let blockSize = try #require(mtpModel.speculativeDecoding).blockSize
        let container = try await mtpModel.loadContainer()
        let tokenizer = await container.tokenizer
        let historyTokens = (0 ..< Self.maximumHistoryRoundCount).flatMap {
            tokenizer.encode(text: SWEBenchTranscript.exchangeText($0), addSpecialTokens: false)
        }
        for contextTokenCount in Self.contextTokenCounts {
            let context = Array(historyTokens.prefix(contextTokenCount))
            try #require(context.count == contextTokenCount, "the history is too short")
            let costs = try await container.perform { modelContext in
                try Self.profile(target: modelContext.model, context: context, blockSize: blockSize)
            }
            for cost in costs {
                Self.log("context \(contextTokenCount) tokens, \(cost.summary)")
            }
            #expect(costs.map(\.rows) == Array(1 ... blockSize))
            #expect(costs.allSatisfy { $0.verify > 0 && $0.attention > 0 && $0.gatedDeltaNet > 0 })
        }
        await mtpModel.evict()
        await releaseAllGPUMemory()
    }

    /// The attention of a verify alone, over random caches of the Qwen3.8 shape in every
    /// full-attention layer, at 32k and 100k keys: the route of a verify, and the SDPA of MLX with
    /// the causal mask that the route replaced. It loads no model.
    @Test func verifyAttentionAtThirtyTwoAndOneHundredThousandKeys() {
        MLXRandom.seed(Self.querySeed)
        for keyCount in Self.contextTokenCounts {
            let caches = (0 ..< Self.attentionLayerCount).map { _ in
                MLXRandom.normal([
                    Self.keysAndValues, 1, Self.kvHeadCount, keyCount, Self.headWidth,
                ]).asType(.bfloat16)
            }
            eval(caches)
            let routes = (1 ... Self.mostVerifyRows).map { rows in
                let queries = caches.map { _ in
                    MLXRandom.normal([1, Self.queryHeadCount, rows, Self.headWidth])
                        .asType(.bfloat16)
                }
                eval(queries)
                let operands = zip(queries, caches).map { ($0, $1[0], $1[1], Self.attentionScale) }
                return (
                    route: Self.median { Self.attentionMilliseconds(operands, maskArray: false) },
                    sdpa: Self.median { Self.sdpaMilliseconds(operands) })
            }
            let format = { (values: [Double]) in
                values.map { String(format: "%.2f", $0) }.joined(separator: ", ")
            }
            Self.log(
                """
                \(keyCount) keys, \(Self.attentionLayerCount) layers, rows 1...: verify route \
                [\(format(routes.map { $0.route }))] ms, SDPA of MLX [\(format(routes.map { $0.sdpa }))] ms
                """)
            #expect(routes.allSatisfy { $0.route > 0 && $0.sdpa > 0 })
        }
    }

    /// The full-attention layers of Qwen3.8.
    private static let attentionLayerCount = 16

    /// The KV heads of Qwen3.8.
    private static let kvHeadCount = 4

    /// The query heads of Qwen3.8.
    private static let queryHeadCount = 24

    /// The width of one attention head of Qwen3.8.
    private static let headWidth = 256

    /// The tensors of one cache: the keys and the values.
    private static let keysAndValues = 2

    /// The most tokens of a verify: the bonus token and four drafts.
    private static let mostVerifyRows = 5

    /// The score factor of a head of 256 numbers.
    private static let attentionScale = 1 / Float(headWidth).squareRoot()

    /// The milliseconds of the SDPA of MLX over each operand, with the causal mask as `.causal`
    /// for more than one query token.
    private static func sdpaMilliseconds(
        _ operands: [(MLXArray, MLXArray, MLXArray, Float)]
    ) -> Double {
        GPUWallTime.milliseconds {
            eval(
                operands.map { queries, keys, values, scale in
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: keys, values: values, scale: scale,
                        mask: queries.dim(HeadRowsAxis.row) > 1 ? .causal : .none)
                })
        }
    }

    /// Prefills `context` into a target cache, then times a verify forward and each layer kind at
    /// each row count up to `blockSize`.
    ///
    /// - Parameters:
    ///   - target: The target, as the text path loads it.
    ///   - context: The tokens of the context.
    ///   - blockSize: The most tokens of a verify.
    /// - Returns: The cost at each row count, from one row up.
    /// - Throws: When the target is not the text Qwen3.5 model.
    private static func profile(
        target: any TargetModel, context: [Int], blockSize: Int
    ) throws -> [VerifyLayerKindCost] {
        let model = try #require(
            (target as? Qwen35Model)?.languageModel ?? (target as? Qwen35TextModel),
            "the text path loads the text Qwen3.5 model")
        let cache = try target.newCache(parameters: nil)
        let bonus = prefill(model, context: context, cache: cache)
        let layers = LayersByKind(model, cache: cache)
        MLXRandom.seed(querySeed)
        return (1 ... blockSize).map { rows in
            let tokens = Array(repeating: bonus, count: rows)
            return VerifyLayerKindCost(
                rows: rows,
                verify: median { verifyMilliseconds(target, tokens: tokens, cache: cache) },
                gatedDeltaNet: median { gatedDeltaNetMilliseconds(model, layers, tokens) },
                gatedDeltaProjections: median {
                    gatedDeltaProjectionMilliseconds(model, layers, tokens)
                },
                attentionProjections: median {
                    attentionProjectionMilliseconds(model, layers, tokens)
                },
                attention: median { attentionMilliseconds(layers, rows: rows, maskArray: false) },
                maskArrayAttention: median {
                    attentionMilliseconds(layers, rows: rows, maskArray: true)
                },
                feedForward: median { feedForwardMilliseconds(model, layers, tokens) })
        }
    }

    /// Prefills `context` into `cache` in chunks, each projecting only its last token.
    ///
    /// - Returns: The greedy token after the context, the bonus token of the first round.
    private static func prefill(_ model: Qwen35TextModel, context: [Int], cache: [KVCache]) -> Int {
        var bonus = 0
        for start in stride(from: 0, to: context.count, by: prefillChunkTokenCount) {
            let end = min(start + prefillChunkTokenCount, context.count)
            let output = model.nextTokenLogits(
                LMInput.Text(tokens: tokenArray(Array(context[start ..< end]))), cache: cache,
                state: nil)
            eval(cache.flatMap(\.state))
            bonus = end < context.count ? context[end] : argMax(output.logits[0, -1]).item(Int.self)
        }
        return bonus
    }

    /// The token ids `tokens` as a batch of one.
    private static func tokenArray(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map(Int32.init))[.newAxis]
    }

    /// The milliseconds of one verify forward of `tokens`, as the MTP iterator runs it: the
    /// drafter state asked of the target and the recurrent snapshots kept.
    private static func verifyMilliseconds(
        _ target: any TargetModel, tokens: [Int], cache: [KVCache]
    ) -> Double {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        return GPUWallTime.milliseconds {
            let output = target(LMInput.Text(tokens: tokenArray(tokens)), cache: cache, state: state)
            eval([output.logits] + (output.state?[mtpLastHiddenStatesKey].map { [$0] } ?? []))
        }
    }

    /// The milliseconds of `body` inside the speculative-verify scope, with its arrays evaluated.
    private static func verifyScopeMilliseconds(_ body: () -> [MLXArray]) -> Double {
        GPUWallTime.milliseconds {
            MultiRowQuantizedMatvec.run(asSpeculativeVerify: true) { eval(body()) }
        }
    }

    /// The embedded `tokens`, the input of each layer pass, evaluated.
    private static func embedded(_ model: Qwen35TextModel, _ tokens: [Int]) -> MLXArray {
        let x = model.model.embedTokens(tokenArray(tokens))
        eval(x)
        return x
    }

    /// The milliseconds of every gated delta net mixer over `tokens`, from each mixer's states,
    /// keeping the snapshots of a verify.
    private static func gatedDeltaNetMilliseconds(
        _ model: Qwen35TextModel, _ layers: LayersByKind, _ tokens: [Int]
    ) -> Double {
        let x = embedded(model, tokens)
        return verifyScopeMilliseconds {
            layers.gatedDeltaNets.flatMap { mixer, cache -> [MLXArray] in
                let zero = mixer.zeroStates(batch: 1, dtype: x.dtype)
                let step = mixer.forward(
                    x, convState: cache[0] ?? zero.conv, recState: cache[1], mask: nil,
                    keepsSnapshots: true)
                return [step.output, step.convState, step.recurrentState]
            }
        }
    }

    /// The milliseconds of the input and output projections of every gated delta net mixer.
    private static func gatedDeltaProjectionMilliseconds(
        _ model: Qwen35TextModel, _ layers: LayersByKind, _ tokens: [Int]
    ) -> Double {
        let x = embedded(model, tokens)
        return verifyScopeMilliseconds {
            layers.gatedDeltaNets.flatMap { mixer, _ -> [MLXArray] in
                let inputs = mixer.projectInputs(x, batch: 1, sequence: tokens.count)
                let output = mixer.projectOutput(inputs.z.reshaped(1, tokens.count, -1))
                return [inputs.qkv, inputs.b, inputs.a, output]
            }
        }
    }

    /// The milliseconds of the q/k/v/o projections of every full-attention layer.
    private static func attentionProjectionMilliseconds(
        _ model: Qwen35TextModel, _ layers: LayersByKind, _ tokens: [Int]
    ) -> Double {
        let x = embedded(model, tokens)
        return verifyScopeMilliseconds {
            layers.attentionLayers.flatMap { layer, _, _ -> [MLXArray] in
                let projected = layer.projectAttention(x)
                let merged = layer.mergeAttention(projected.queries, gate: projected.gate)
                return [merged, projected.keys, projected.values]
            }
        }
    }

    /// The milliseconds of the attention of `rows` random queries over the cache of every
    /// full-attention layer: on the route of a verify, the causal mask as `.causal`, or through
    /// the SDPA of MLX with the causal mask as an array.
    private static func attentionMilliseconds(
        _ layers: LayersByKind, rows: Int, maskArray: Bool
    ) -> Double {
        let operands = layers.attentionLayers.map { _, module, cache in
            let state = cache.state
            let queries = MLXRandom.normal(
                [state[0].dim(0), module.attentionHeads, rows, state[0].dim(-1)]
            ).asType(state[0].dtype)
            return (queries, state[0], state[1], module.scale)
        }
        eval(operands.map(\.0))
        return attentionMilliseconds(operands, maskArray: maskArray)
    }

    /// The milliseconds of the attention of each operand: on the route of a verify, the causal
    /// mask as `.causal`, or through the SDPA of MLX with the causal mask as an array.
    private static func attentionMilliseconds(
        _ operands: [(MLXArray, MLXArray, MLXArray, Float)], maskArray: Bool
    ) -> Double {
        GPUWallTime.milliseconds {
            eval(
                operands.map { queries, keys, values, scale in
                    attention(
                        queries: queries, keys: keys, values: values, scale: scale,
                        maskArray: maskArray)
                })
        }
    }

    /// The attention of `queries` over the keys, the last query on the last key: on the route of
    /// a verify, or through the SDPA of MLX with the causal mask as an array.
    private static func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, maskArray: Bool
    ) -> MLXArray {
        let rows = queries.dim(HeadRowsAxis.row)
        guard rows > 1 else {
            return PrefillAttentionKernel.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        }
        guard maskArray else {
            return PrefillAttentionKernel.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
        }
        let mask = createCausalMask(n: rows, offset: keys.dim(HeadRowsAxis.row) - rows)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))
    }

    /// The milliseconds of the feed-forward block of every layer.
    private static func feedForwardMilliseconds(
        _ model: Qwen35TextModel, _ layers: LayersByKind, _ tokens: [Int]
    ) -> Double {
        let x = embedded(model, tokens)
        return verifyScopeMilliseconds { layers.all.map { $0.feedForward(x) } }
    }

    /// The median of ``timedRepeats`` runs of `sample`.
    private static func median(_ sample: () -> Double) -> Double {
        GPUWallTime.median(repeats: timedRepeats, sample)
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
