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
// 512 affine keys.
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

/// A stand-in for the checkpoint keys that carry a `scales` array.
///
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// quantizes a layer only when the weight file holds `<path>.scales` for it,
/// and DeepSeek-V4-Flash-4bit stores those arrays for exactly the layers its
/// `quantization` block names. The probe run below reproduces that gate, thus
/// the router -- which the block does not name -- stays in high precision
/// rather than picking up the affine default.
private func quantizationTuple(
    for path: String,
    plan: BaseConfiguration.PerLayerQuantization
) -> (Int, Int, QuantizationMode)? {
    guard plan.perLayerQuantization[path] != nil else { return nil }
    return plan.quantization(layer: path)?.asTuple
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

    /// The three routed expert projections `SwitchGLU` carries, named as the
    /// DeepSeek-V4 checkpoint names them.
    private static let expertProjections = ["gate_proj", "up_proj", "down_proj"]

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

    // MARK: - Plan application over the checkpoint's own key paths

    @Test func planAppliesMxfp4ToExpertsAndAffineToEverythingElse() throws {
        let plan = try decodeFixturePlan()
        let probe = ProbeModel(
            hiddenSize: Self.probeHiddenSize,
            expertHiddenSize: Self.probeExpertHiddenSize,
            expertCount: Self.probeExpertCount,
            vocabularySize: Self.probeVocabularySize)

        quantize(model: probe) { path, _ in
            quantizationTuple(for: path, plan: plan)
        }

        let leaves = Dictionary(uniqueKeysWithValues: probe.leafModules().flattened())

        for projection in Self.expertProjections {
            let path = "model.layers.0.ffn.switch_mlp.\(projection)"
            let layer = try #require(
                leaves[path] as? QuantizedSwitchLinear, "\(path) is quantized")
            #expect(layer.mode == .mxfp4, "\(path) mode")
            #expect(layer.groupSize == Self.mxfp4GroupSize, "\(path) group size")
            #expect(layer.bits == Self.quantizationBits, "\(path) bits")
            #expect(layer.biases == nil, "\(path) carries scales but no biases")
        }

        for path in ["model.layers.0.attn.wq_a", "model.layers.0.ffn.shared_experts.gate_proj"] {
            let layer = try #require(
                leaves[path] as? QuantizedLinear, "\(path) is quantized")
            #expect(layer.mode == .affine, "\(path) mode")
            #expect(layer.groupSize == Self.affineGroupSize, "\(path) group size")
        }

        let embedding = try #require(
            leaves["model.embed_tokens"] as? QuantizedEmbedding, "embeddings are quantized")
        #expect(embedding.mode == .affine)
        #expect(embedding.groupSize == Self.affineGroupSize)

        let router = try #require(leaves["model.layers.0.ffn.gate"], "the router survives")
        #expect(router is Linear)
        #expect(!(router is QuantizedLinear), "the router stays in high precision")
    }

    // MARK: - mxfp4 forward pass

    @Test func mxfp4SwitchLinearRunsWithScalesAndNoBiases() throws {
        let (layer, x, indices) = try makeQuantizedSwitchLinear(
            mode: .mxfp4, groupSize: Self.mxfp4GroupSize)

        #expect(layer.mode == .mxfp4)
        #expect(layer.groupSize == Self.mxfp4GroupSize)
        #expect(layer.bits == Self.quantizationBits)
        #expect(layer.biases == nil, "mxfp4 carries scales but no biases")

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
        #expect(layer.biases != nil, "affine carries both scales and biases")

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
private final class ProbeAttention: Module {
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "compressor") var compressor: ProbeCompressor

    init(hiddenSize: Int) {
        self._wqA.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._wkv.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._compressor.wrappedValue = ProbeCompressor(hiddenSize: hiddenSize)
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

    init(hiddenSize: Int, expertHiddenSize: Int, expertCount: Int) {
        self._attn.wrappedValue = ProbeAttention(hiddenSize: hiddenSize)
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

    init(hiddenSize: Int, expertHiddenSize: Int, expertCount: Int, vocabularySize: Int) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabularySize, dimensions: hiddenSize)
        self._layers.wrappedValue = [
            ProbeLayer(
                hiddenSize: hiddenSize, expertHiddenSize: expertHiddenSize,
                expertCount: expertCount)
        ]
        super.init()
    }
}

/// A one-layer stand-in for the DeepSeek-V4 model whose flattened module paths
/// equal the checkpoint key paths the published `quantization` block names.
private final class ProbeModel: Module {
    @ModuleInfo(key: "model") var model: ProbeInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(hiddenSize: Int, expertHiddenSize: Int, expertCount: Int, vocabularySize: Int) {
        self._model.wrappedValue = ProbeInner(
            hiddenSize: hiddenSize, expertHiddenSize: expertHiddenSize,
            expertCount: expertCount, vocabularySize: vocabularySize)
        self._lmHead.wrappedValue = Linear(hiddenSize, vocabularySize, bias: false)
        super.init()
    }
}
