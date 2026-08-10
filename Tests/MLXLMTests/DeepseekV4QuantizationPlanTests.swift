// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4-Flash-4bit ships a mixed quantization plan: a 4-bit affine
// default at group size 64, and mxfp4 at group size 32 on the three routed
// expert projections of every layer. These tests show that the plumbing this
// repository already carries resolves that plan, applies it, and runs the
// mxfp4 layers.
//
// The fixture `Resources/DeepSeek-V4-Flash-4bit-config.json` is a copy of the
// published `config.json`. Its `quantization` block holds three scalar keys
// (`group_size`, `bits`, `mode`) and 641 per-layer keys: 129 mxfp4 keys, which
// are `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj` for N in 0...42, and
// 512 affine keys. Among the affine keys the block names a compressor on the 41
// layers 2 to 42 -- 82 keys, `wgate` and `wkv` on each -- and an indexer on the
// 21 even layers 2 to 42 -- 84 keys, `wq_b`, `weights_proj`, `compressor.wgate`
// and `compressor.wkv` on each. Layers 0 and 1 hold a compress ratio of 0, thus
// the block names neither a compressor nor an indexer for them.
//
// Vacuity. The affine per-layer keys hold the same three values as the
// default, thus a resolver that dropped every per-layer key would still give
// the correct answer for them. The tests thus lean on the mxfp4 keys, which
// differ from the default in both group size and mode, and they read the
// per-layer dictionary directly to show that an affine key is an entry of its
// own and not the fallback.
//
// The plan names checkpoint key paths. `quantize(model:filter:)` gives its
// filter the flattened module path, thus the two agree only when the Swift
// module tree carries the same `@ModuleInfo` keys. `ProbeModel` below is a
// small tree with those keys, and it stands for the naming contract the
// DeepSeek-V4 model must meet.

import Foundation
import MLX
import MLXNN
import Testing

// `QuantizedSwitchLinear` keeps `weight`, `scales` and `biases` internal, and
// the tests below read all three, thus the import is `@testable`.
@testable import MLXLMCommon

/// Builds a stand-in weight dictionary holding a `scales` array for each path.
///
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// quantizes a layer only when the weight files hold `<path>.scales` for it,
/// and ``quantizationParameters(forPath:weights:quantization:perLayerQuantization:)``
/// is the gate that decides it. The tests below run that same function, thus
/// they need a weight dictionary to run it against. Only the presence of a key
/// counts, thus the arrays hold nothing of interest.
///
/// This repository holds no DeepSeek-V4 weight file, thus which paths the
/// published checkpoint truly holds a `scales` array for is an assumption. The
/// tests state their results as "given these scales, the filter gives this",
/// and never as a statement about the published weight file.
///
/// - Parameter paths: The layer paths whose weights are quantized.
/// - Returns: A weight dictionary holding one `scales` array for each path.
private func stubScales(for paths: [String]) -> [String: MLXArray] {
    Dictionary(uniqueKeysWithValues: paths.map { ("\($0).scales", MLXArray.zeros([1])) })
}

@Suite(.serialized)
struct DeepseekV4QuantizationPlanTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixture facts

    /// Decoder layer count of DeepSeek-V4-Flash, and thus the number of layers
    /// the `quantization` block gives mxfp4 expert projections.
    private static let layerCount = 43

    /// First layer the block names a compressor for. Layers 0 and 1 hold a
    /// compress ratio of 0, thus they hold no compressor.
    private static let firstCompressorLayer = 2

    /// Compressor keys the block holds: `wgate` and `wkv` on each of the 41
    /// layers 2 to 42.
    private static let compressorKeyCount = 82

    /// Indexer keys the block holds: `wq_b`, `weights_proj`,
    /// `compressor.wgate` and `compressor.wkv` on each of the 21 even layers
    /// 2 to 42, which are the layers whose compress ratio is 4.
    private static let indexerKeyCount = 84

    /// The three routed expert projections `SwitchGLU` carries, named as the
    /// DeepSeek-V4 checkpoint names them.
    private static let expertProjections = ["gate_proj", "up_proj", "down_proj"]

    /// The two projections a compressor carries.
    private static let compressorProjections = ["wgate", "wkv"]

    /// Group size of the affine default, and of every affine per-layer entry.
    private static let affineGroupSize = 64

    /// Group size the mxfp4 entries state. MXFP4 shares one power-of-two scale
    /// across a block of 32 values.
    private static let mxfp4GroupSize = 32

    /// Bit width of every entry, affine and mxfp4 alike.
    private static let quantizationBits = 4

    // MARK: - Probe module sizes

    /// Hidden size of the probe tree. A multiple of ``affineGroupSize``, thus
    /// every affine probe layer is quantizable.
    private static let probeHiddenSize = 64

    /// Per-expert intermediate size of the probe tree. A multiple of
    /// ``mxfp4GroupSize``, thus the probe's expert projections are quantizable.
    private static let probeExpertHiddenSize = 32

    /// Expert count of the probe tree and of the forward-pass layers below.
    private static let probeExpertCount = 4

    /// Vocabulary size of the probe tree.
    private static let probeVocabularySize = 8

    /// Decoder layers the probe tree holds. Three, so that the tree reaches
    /// layer ``firstCompressorLayer`` -- the first layer the block names a
    /// compressor for.
    private static let probeLayerCount = 3

    /// Tokens fed through the forward-pass layers below.
    private static let forwardTokenCount = 2

    /// Experts each token routes to in the forward-pass layers below.
    private static let forwardExpertsPerToken = 2

    // MARK: - Helpers

    private func decodeFixturePlan() throws -> BaseConfiguration.PerLayerQuantization {
        let name = "DeepSeek-V4-Flash-4bit-config"
        let url = try #require(
            Bundle.module.url(forResource: name, withExtension: "json"),
            "Missing fixture: \(name).json")
        let configuration = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: url))
        return try #require(
            configuration.perLayerQuantization,
            "The fixture holds a quantization block, thus the plan decodes")
    }

    /// Every path the fixture gives mxfp4.
    private static var expertProjectionPaths: [String] {
        (0 ..< layerCount).flatMap { layer in
            expertProjections.map { "model.layers.\(layer).ffn.switch_mlp.\($0)" }
        }
    }

    /// Quantizes a probe tree through the production load-time filter and gives
    /// its flattened leaf modules.
    ///
    /// The filter is
    /// ``quantizationParameters(forPath:weights:quantization:perLayerQuantization:)``
    /// itself, which is what `loadWeights` runs, thus these tests pin the gate
    /// the load path takes rather than a stand-in for it.
    ///
    /// - Parameters:
    ///   - plan: The per-layer plan the filter resolves paths against.
    ///   - scalePaths: The layer paths whose weights hold a `scales` array.
    /// - Returns: The leaf modules of the quantized probe tree, by path.
    private func leavesAfterLoadFilter(
        plan: BaseConfiguration.PerLayerQuantization, scalePaths: [String]
    ) -> [String: Module] {
        let probe = ProbeModel(
            hiddenSize: Self.probeHiddenSize,
            expertHiddenSize: Self.probeExpertHiddenSize,
            expertCount: Self.probeExpertCount,
            vocabularySize: Self.probeVocabularySize,
            layerCount: Self.probeLayerCount,
            firstCompressorLayer: Self.firstCompressorLayer)
        let weights = stubScales(for: scalePaths)

        quantize(model: probe) { path, _ in
            quantizationParameters(
                forPath: path, weights: weights, quantization: nil,
                perLayerQuantization: plan)
        }

        return Dictionary(uniqueKeysWithValues: probe.leafModules().flattened())
    }

    /// Builds a quantized switch layer of the given mode, plus the input and
    /// the routing indices its forward pass takes.
    private func makeQuantizedSwitchLinear(
        mode: QuantizationMode, groupSize: Int
    ) throws -> (layer: QuantizedSwitchLinear, x: MLXArray, indices: MLXArray) {
        let full = SwitchLinear(
            inputDims: Self.probeHiddenSize,
            outputDims: Self.probeExpertHiddenSize,
            numExperts: Self.probeExpertCount,
            bias: false)
        let quantized = try #require(
            full.toQuantized(
                groupSize: groupSize, bits: Self.quantizationBits, mode: mode)
                as? QuantizedSwitchLinear,
            "`toQuantized` gives a `QuantizedSwitchLinear`")

        // `SwitchGLU` expands the token axis twice before it calls a switch
        // layer, and gives the routing indices unexpanded. Do the same here.
        let tokens = MLXArray(
            0 ..< (Self.forwardTokenCount * Self.probeHiddenSize)
        )
        .asType(.float32)
        .reshaped(Self.forwardTokenCount, Self.probeHiddenSize)
        let x = MLX.expandedDimensions(tokens, axes: [-2, -3])
        let indices = MLXArray(
            (0 ..< (Self.forwardTokenCount * Self.forwardExpertsPerToken)).map {
                Int32($0 % Self.probeExpertCount)
            }
        )
        .reshaped(Self.forwardTokenCount, Self.forwardExpertsPerToken)

        return (quantized, x, indices)
    }

    // MARK: - Plan resolution

    @Test func fixtureGivesEveryExpertProjectionMxfp4() throws {
        let plan = try decodeFixturePlan()

        for path in Self.expertProjectionPaths {
            let resolved = try #require(
                plan.quantization(layer: path), "\(path) resolves")
            #expect(resolved.mode == .mxfp4, "\(path) mode")
            #expect(resolved.groupSize == Self.mxfp4GroupSize, "\(path) group size")
            #expect(resolved.bits == Self.quantizationBits, "\(path) bits")
        }

        let mxfp4Count = plan.perLayerQuantization.values.filter { option in
            guard case .quantize(let quantization) = option else { return false }
            return quantization.mode == .mxfp4
        }.count
        #expect(mxfp4Count == Self.expertProjectionPaths.count)
    }

    @Test func fixtureGivesTheDefaultAffineAtGroupSize64() throws {
        let plan = try decodeFixturePlan()

        let fallback = try #require(plan.quantization, "the block states a default")
        #expect(fallback.mode == .affine)
        #expect(fallback.groupSize == Self.affineGroupSize)
        #expect(fallback.bits == Self.quantizationBits)
    }

    @Test func fixtureNamesTheAffineLayersAsEntriesOfTheirOwn() throws {
        let plan = try decodeFixturePlan()

        let affinePaths = [
            "model.embed_tokens",
            "lm_head",
            "model.layers.0.attn.wq_a",
            // Layers 0 and 1 hold a compress ratio of 0 and thus carry no
            // compressor. Layer 2 is the first one that does.
            "model.layers.2.attn.compressor.wkv",
            "model.layers.2.attn.indexer.wq_b",
            "model.layers.0.ffn.shared_experts.gate_proj",
        ]
        for path in affinePaths {
            let entry = try #require(
                plan.perLayerQuantization[path],
                "\(path) is an entry of its own, not the fallback")
            guard case .quantize(let quantization) = entry else {
                Issue.record("\(path) is a skip, not a quantization")
                continue
            }
            #expect(quantization.mode == .affine, "\(path) mode")
            #expect(quantization.groupSize == Self.affineGroupSize, "\(path) group size")
            #expect(quantization.bits == Self.quantizationBits, "\(path) bits")
        }
    }

    @Test func fixtureLeavesTheRouterOutOfThePlan() throws {
        let plan = try decodeFixturePlan()

        // The router picks the experts. DeepSeek-V4 keeps it in high
        // precision, thus the block names no entry for it.
        #expect(plan.perLayerQuantization["model.layers.0.ffn.gate"] == nil)
    }

    @Test func fixtureNamesCompressorKeysFromLayerTwoUp() throws {
        let plan = try decodeFixturePlan()

        for layer in 0 ..< Self.layerCount {
            let named = layer >= Self.firstCompressorLayer
            for projection in Self.compressorProjections {
                let path = "model.layers.\(layer).attn.compressor.\(projection)"
                #expect(
                    (plan.perLayerQuantization[path] != nil) == named,
                    "\(path) named: \(named)")
            }
        }

        let compressorKeys = plan.perLayerQuantization.keys.filter {
            $0.contains(".attn.compressor.")
        }
        #expect(compressorKeys.count == Self.compressorKeyCount)
    }

    @Test func fixtureNamesIndexerKeysOnTheEvenLayersFromTwoUp() throws {
        let plan = try decodeFixturePlan()

        for layer in 0 ..< Self.layerCount {
            let named = layer >= Self.firstCompressorLayer && layer.isMultiple(of: 2)
            let path = "model.layers.\(layer).attn.indexer.wq_b"
            #expect(
                (plan.perLayerQuantization[path] != nil) == named,
                "\(path) named: \(named)")
        }

        let indexerKeys = plan.perLayerQuantization.keys.filter {
            $0.contains(".attn.indexer.")
        }
        #expect(indexerKeys.count == Self.indexerKeyCount)
    }

    // MARK: - Plan application through the production load-time filter

    @Test func loadFilterAppliesMxfp4ToExpertsAndAffineToEverythingElse() throws {
        let plan = try decodeFixturePlan()
        let leaves = leavesAfterLoadFilter(
            plan: plan, scalePaths: Array(plan.perLayerQuantization.keys))

        for projection in Self.expertProjections {
            let path = "model.layers.0.ffn.switch_mlp.\(projection)"
            let layer = try #require(
                leaves[path] as? QuantizedSwitchLinear, "\(path) is quantized")
            #expect(layer.mode == .mxfp4, "\(path) mode")
            #expect(layer.groupSize == Self.mxfp4GroupSize, "\(path) group size")
            #expect(layer.bits == Self.quantizationBits, "\(path) bits")
            #expect(layer.biases == nil, "\(path) holds scales and no biases")
        }

        let affinePaths = [
            "model.layers.0.attn.wq_a",
            "model.layers.0.ffn.shared_experts.gate_proj",
            "model.layers.\(Self.firstCompressorLayer).attn.compressor.wkv",
        ]
        for path in affinePaths {
            let layer = try #require(
                leaves[path] as? QuantizedLinear, "\(path) is quantized")
            #expect(layer.mode == .affine, "\(path) mode")
            #expect(layer.groupSize == Self.affineGroupSize, "\(path) group size")
            #expect(layer.bits == Self.quantizationBits, "\(path) bits")
        }

        let embedding = try #require(
            leaves["model.embed_tokens"] as? QuantizedEmbedding, "embeddings are quantized")
        #expect(embedding.mode == .affine)
        #expect(embedding.groupSize == Self.affineGroupSize)
    }

    @Test func loadFilterLeavesAPathWhoseWeightsHoldNoScalesAlone() throws {
        let plan = try decodeFixturePlan()
        let routerPath = "model.layers.0.ffn.gate"
        #expect(
            plan.perLayerQuantization[routerPath] == nil,
            "the premise: the plan names no entry for the router")

        // The plan names every path that holds a `scales` array here, and the
        // router is not among them, thus the first part of the gate keeps the
        // router in high precision.
        let leaves = leavesAfterLoadFilter(
            plan: plan, scalePaths: Array(plan.perLayerQuantization.keys))

        let router = try #require(leaves[routerPath], "the router survives")
        #expect(router is Linear)
        #expect(!(router is QuantizedLinear), "no scales, thus no quantization")
    }

    @Test func loadFilterGivesTheDefaultToAPathThePlanDoesNotName() throws {
        let plan = try decodeFixturePlan()
        let routerPath = "model.layers.0.ffn.gate"
        #expect(
            plan.perLayerQuantization[routerPath] == nil,
            "the premise: the plan names no entry for the router")

        // Add a `scales` array for the router. The first part of the gate now
        // passes, and the second part gives it the plan's own default, because
        // the plan names no entry of its own for that path.
        let leaves = leavesAfterLoadFilter(
            plan: plan, scalePaths: Array(plan.perLayerQuantization.keys) + [routerPath])

        let router = try #require(
            leaves[routerPath] as? QuantizedLinear,
            "a path the plan does not name takes the default when its weights hold scales")
        #expect(router.mode == .affine)
        #expect(router.groupSize == Self.affineGroupSize)
        #expect(router.bits == Self.quantizationBits)
    }

    @Test func loadFilterQuantizesNothingWhenNoWeightsHoldScales() throws {
        let plan = try decodeFixturePlan()
        let leaves = leavesAfterLoadFilter(plan: plan, scalePaths: [])

        for (path, module) in leaves {
            #expect(!(module is Quantized), "\(path) stays in high precision")
        }
    }

    // MARK: - mxfp4 forward pass

    @Test func mxfp4SwitchLinearRunsWithScalesAndNoBiases() throws {
        let (layer, x, indices) = try makeQuantizedSwitchLinear(
            mode: .mxfp4, groupSize: Self.mxfp4GroupSize)

        #expect(layer.mode == .mxfp4)
        #expect(layer.groupSize == Self.mxfp4GroupSize)
        #expect(layer.bits == Self.quantizationBits)
        #expect(layer.biases == nil, "mxfp4 holds scales and no biases")

        let out = layer(x, indices)
        eval(out)

        #expect(
            out.shape == [
                Self.forwardTokenCount, Self.forwardExpertsPerToken, 1,
                Self.probeExpertHiddenSize,
            ])
        #expect(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test func mxfp4SwitchLinearMatchesItsOwnDequantizedWeights() throws {
        let (layer, x, indices) = try makeQuantizedSwitchLinear(
            mode: .mxfp4, groupSize: Self.mxfp4GroupSize)

        // Dequantize by hand and run the plain gather-matmul. This is the
        // reference the fused quantized kernel must reproduce, thus it shows
        // that the layer hands its own group size, bit width and mode to
        // `gatherQuantizedMM` rather than a default.
        let weight = MLX.dequantized(
            layer.weight,
            scales: layer.scales,
            biases: layer.biases,
            groupSize: Self.mxfp4GroupSize,
            bits: Self.quantizationBits,
            mode: .mxfp4)
        let expected = MLX.gatherMM(
            x, weight.swappedAxes(-1, -2), rhsIndices: indices)

        let out = layer(x, indices)
        eval(out, expected)
        #expect(MLX.allClose(out, expected).item(Bool.self))
    }

    @Test func affineSwitchLinearStillRunsOnTheSharedPath() throws {
        let (layer, x, indices) = try makeQuantizedSwitchLinear(
            mode: .affine, groupSize: Self.mxfp4GroupSize)

        #expect(layer.mode == .affine)
        #expect(layer.biases != nil, "affine holds both scales and biases")

        let out = layer(x, indices)
        eval(out)

        #expect(
            out.shape == [
                Self.forwardTokenCount, Self.forwardExpertsPerToken, 1,
                Self.probeExpertHiddenSize,
            ])
        #expect(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}

// MARK: - Probe module tree

/// The routed and shared halves of one DeepSeek-V4 FFN block.
private final class ProbeFFN: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: ProbeSharedExperts
    @ModuleInfo(key: "gate") var gate: Linear

    init(hiddenSize: Int, expertHiddenSize: Int, expertCount: Int) {
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenSize, hiddenDims: expertHiddenSize, numExperts: expertCount)
        self._sharedExperts.wrappedValue = ProbeSharedExperts(hiddenSize: hiddenSize)
        self._gate.wrappedValue = Linear(hiddenSize, expertCount, bias: false)
        super.init()
    }
}

/// The dense expert every token passes through, beside the routed ones.
private final class ProbeSharedExperts: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._downProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }
}

/// The low-rank query and key/value projections of one DeepSeek-V4 attention
/// block, named as the checkpoint names them.
///
/// The compressor is present only on a layer the `quantization` block names one
/// for, which is a layer whose compress ratio is more than 0.
private final class ProbeAttention: Module {
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "compressor") var compressor: ProbeCompressor?

    init(hiddenSize: Int, hasCompressor: Bool) {
        self._wqA.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._wkv.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._compressor.wrappedValue =
            hasCompressor ? ProbeCompressor(hiddenSize: hiddenSize) : nil
        super.init()
    }
}

/// The key/value compressor nested inside an attention block.
private final class ProbeCompressor: Module {
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear

    init(hiddenSize: Int) {
        self._wgate.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._wkv.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }
}

/// One decoder layer of the probe tree.
private final class ProbeLayer: Module {
    @ModuleInfo(key: "attn") var attn: ProbeAttention
    @ModuleInfo(key: "ffn") var ffn: ProbeFFN

    init(hiddenSize: Int, expertHiddenSize: Int, expertCount: Int, hasCompressor: Bool) {
        self._attn.wrappedValue = ProbeAttention(
            hiddenSize: hiddenSize, hasCompressor: hasCompressor)
        self._ffn.wrappedValue = ProbeFFN(
            hiddenSize: hiddenSize, expertHiddenSize: expertHiddenSize,
            expertCount: expertCount)
        super.init()
    }
}

/// The embedding table and the decoder stack, under the `model` key the
/// checkpoint uses.
private final class ProbeInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [ProbeLayer]

    init(
        hiddenSize: Int, expertHiddenSize: Int, expertCount: Int, vocabularySize: Int,
        layerCount: Int, firstCompressorLayer: Int
    ) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: hiddenSize)
        self._layers.wrappedValue = (0 ..< layerCount).map { layer in
            ProbeLayer(
                hiddenSize: hiddenSize, expertHiddenSize: expertHiddenSize,
                expertCount: expertCount, hasCompressor: layer >= firstCompressorLayer)
        }
        super.init()
    }
}

/// A stand-in for the DeepSeek-V4 model whose flattened module paths equal
/// checkpoint key paths the published `quantization` block names.
private final class ProbeModel: Module {
    @ModuleInfo(key: "model") var model: ProbeInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(
        hiddenSize: Int, expertHiddenSize: Int, expertCount: Int, vocabularySize: Int,
        layerCount: Int, firstCompressorLayer: Int
    ) {
        self._model.wrappedValue = ProbeInner(
            hiddenSize: hiddenSize, expertHiddenSize: expertHiddenSize,
            expertCount: expertCount, vocabularySize: vocabularySize,
            layerCount: layerCount, firstCompressorLayer: firstCompressorLayer)
        self._lmHead.wrappedValue = Linear(hiddenSize, vocabularySize, bias: false)
        super.init()
    }
}
