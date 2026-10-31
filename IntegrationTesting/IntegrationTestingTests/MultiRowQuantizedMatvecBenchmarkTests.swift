// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// One quantized projection of a Qwen3.8-27B decode step, and the number of
/// times one step runs it.
struct StepProjection {
    let inputs: Int
    let outputs: Int
    let count: Int
}

/// The widths and the layer counts of Qwen3.8-27B.
private enum Qwen38 {
    /// The width of the hidden state.
    static let hiddenSize = 5120

    /// The fused input projection of a gated delta net: queries, keys,
    /// values, the output gate, and the beta and alpha of each value head.
    static let gatedDeltaInputWidth = 16480

    /// The value width of a gated delta net, the input of its `out_proj`.
    static let gatedDeltaValueWidth = 6144

    /// The queries of a full-attention layer with their output gate.
    static let attentionQueryWidth = 12288

    /// The keys, or the values, of a full-attention layer.
    static let attentionKeyValueWidth = 1024

    /// The heads of a full-attention layer, the input of its `o_proj`.
    static let attentionOutputWidth = 6144

    /// The hidden width of each MLP.
    static let intermediateSize = 17408

    /// The tokens of the vocabulary, the outputs of the LM head.
    static let vocabularySize = 248_320

    /// The gated delta net layers.
    static let gatedDeltaLayers = 48

    /// The full-attention layers.
    static let attentionLayers = 16

    /// Every decoder layer, each with one MLP.
    static let layers = gatedDeltaLayers + attentionLayers

    /// The key and the value projection of each full-attention layer.
    static let keyValueProjectionsPerLayer = 2

    /// The gate and the up projection of each MLP.
    static let gateUpProjectionsPerMLP = 2
}

/// The quantized projections of one Qwen3.8-27B step: the fused input and the
/// output projection of each gated delta net layer, the four projections of
/// each full-attention layer, the three projections of each MLP, and the LM
/// head.
let qwen38StepProjections = [
    StepProjection(
        inputs: Qwen38.hiddenSize, outputs: Qwen38.gatedDeltaInputWidth,
        count: Qwen38.gatedDeltaLayers),
    StepProjection(
        inputs: Qwen38.gatedDeltaValueWidth, outputs: Qwen38.hiddenSize,
        count: Qwen38.gatedDeltaLayers),
    StepProjection(
        inputs: Qwen38.hiddenSize, outputs: Qwen38.attentionQueryWidth,
        count: Qwen38.attentionLayers),
    StepProjection(
        inputs: Qwen38.hiddenSize, outputs: Qwen38.attentionKeyValueWidth,
        count: Qwen38.attentionLayers * Qwen38.keyValueProjectionsPerLayer),
    StepProjection(
        inputs: Qwen38.attentionOutputWidth, outputs: Qwen38.hiddenSize,
        count: Qwen38.attentionLayers),
    StepProjection(
        inputs: Qwen38.hiddenSize, outputs: Qwen38.intermediateSize,
        count: Qwen38.layers * Qwen38.gateUpProjectionsPerMLP),
    StepProjection(
        inputs: Qwen38.intermediateSize, outputs: Qwen38.hiddenSize, count: Qwen38.layers),
    StepProjection(inputs: Qwen38.hiddenSize, outputs: Qwen38.vocabularySize, count: 1),
]

/// A quantization format of the Qwen3.8-27B checkpoints.
struct QuantizationFormat {
    let name: String
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
}

/// The bits of each weight of both Qwen3.8-27B checkpoints.
private let checkpointBits = 4

/// The group size of the affine checkpoint.
private let affineGroupSize = 64

/// The group size of mxfp4, fixed by the format.
private let mxfp4GroupSize = 32

let qwen38Formats = [
    QuantizationFormat(
        name: "4-bit affine g64", groupSize: affineGroupSize, bits: checkpointBits,
        mode: .affine),
    QuantizationFormat(
        name: "mxfp4", groupSize: mxfp4GroupSize, bits: checkpointBits, mode: .mxfp4),
]

/// The verify rows a speculative step runs: one plain decode row up to eight.
private let measuredRows = 1 ... 8
/// The one measured row count that stays with `qmv_wide` of MLX, whose rows
/// differ in the last bits from a row alone.
private let qmvWideRows = 2
/// The largest difference of the `qmv_wide` rows from the MLX matvec of each row alone, as a
/// share of the largest output: they round as bfloat16 does, about 2^-7.
private let qmvWideTolerance: Float = 0.01
private let warmupRuns = 2
private let timedRuns = 7
/// The median of an odd count of samples stands at half the count.
private let medianDivisor = 2
private let millisecondsPerSecond = 1e3
private let attosecondsPerMillisecond = 1e15
/// The bound of the random weights, about the scale of a trained projection.
private let weightRange: Float = 0.02

/// One way to run a quantized projection over a batch of rows.
typealias RowProjection = (MLXArray, QuantizedLinear) -> MLXArray

/// Times the quantized matmuls of one Qwen3.8-27B step for 1 to 8 rows, through
/// the matvec of MLX and through ``MultiRowQuantizedMatvec``, with distinct
/// random weights for every layer so no weight stays in cache from one
/// projection to the next.
@Suite(.serialized)
struct MultiRowQuantizedMatvecBenchmarkTests {

    @Test func qwen38StepMatmulCostByRows() {
        for format in qwen38Formats {
            let projections = qwen38StepProjections.map { makeLayers($0, format) }
            let layers = projections.flatMap { $0 }
            let mlxOneRow = medianStepMilliseconds(layers, rows: 1) { $1($0) }
            for rows in measuredRows {
                let mlx = medianStepMilliseconds(layers, rows: rows) { $1($0) }
                let kernel = medianStepMilliseconds(layers, rows: rows) {
                    MultiRowQuantizedMatvec.quantizedMatmulRows($0, $1)
                }
                for layer in projections.compactMap(\.first) {
                    expectRowsMatchTheirOwnMatvec(rows: rows, layer)
                }
                report(
                    String(
                        format:
                            "[MULTIROW] %@ M=%d: MLX %.2f ms (%.2fx), kernel %.2f ms (%.2fx M=1)",
                        format.name, rows, mlx, mlx / mlxOneRow, kernel, kernel / mlxOneRow))
            }
        }
    }

    /// Every layer of `projection`, each with distinct random weights.
    func makeLayers(
        _ projection: StepProjection, _ format: QuantizationFormat
    ) -> [QuantizedLinear] {
        (0 ..< projection.count).map { _ in
            let weight = MLXRandom.uniform(
                low: -weightRange, high: weightRange, [projection.outputs, projection.inputs]
            ).asType(.bfloat16)
            let layer = QuantizedLinear(
                weight: weight, bias: nil, groupSize: format.groupSize, bits: format.bits,
                mode: format.mode)
            eval(layer)
            return layer
        }
    }

    func medianStepMilliseconds(
        _ layers: [QuantizedLinear], rows: Int, project: RowProjection
    ) -> Double {
        let inputs = Dictionary(
            Set(layers.map { $0.shape.1 }).map { width in
                (width, MLXRandom.normal([rows, width]).asType(.bfloat16))
            },
            uniquingKeysWith: { first, _ in first })
        eval(Array(inputs.values))

        let clock = ContinuousClock()
        var samples: [Double] = []
        for run in 0 ..< (warmupRuns + timedRuns) {
            let start = clock.now
            let outputs = layers.compactMap { layer in
                inputs[layer.shape.1].map { project($0, layer) }
            }
            eval(outputs)
            let elapsed = clock.now - start
            if run >= warmupRuns {
                samples.append(
                    Double(elapsed.components.attoseconds) / attosecondsPerMillisecond
                        + Double(elapsed.components.seconds) * millisecondsPerSecond)
            }
        }
        return median(samples)
    }

    /// The output of ``MultiRowQuantizedMatvec`` over `x`, and the matvec of MLX
    /// over each row alone, both as float32.
    private func kernelAndRowByRowOutputs(
        _ x: MLXArray, _ layer: QuantizedLinear
    ) -> (got: [Float], want: [Float]) {
        let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer).asType(.float32)
        let want = concatenated((0 ..< x.dim(0)).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0)
            .asType(.float32)
        return (got.asArray(Float.self), want.asArray(Float.self))
    }

    /// Checks `rows` random rows through `layer` against the matvec of MLX over each
    /// row alone. The rows of `qmv_wide` sum in another order than a row alone, so
    /// they hold to ``qmvWideTolerance``; every other row count keeps the bits.
    func expectRowsMatchTheirOwnMatvec(rows: Int, _ layer: QuantizedLinear) {
        let x = MLXRandom.normal([rows, layer.shape.1]).asType(.bfloat16)
        if rows == qmvWideRows {
            #expect(
                relativeOutputError(x, layer) <= qmvWideTolerance, "\(layer.shape) M=\(rows)")
        } else {
            #expect(mismatchedRowValues(x, layer) == 0, "\(layer.shape) M=\(rows)")
        }
    }

    /// The count of values where the kernel output differs in bits from the
    /// matvec of MLX over each row alone.
    private func mismatchedRowValues(_ x: MLXArray, _ layer: QuantizedLinear) -> Int {
        let (got, want) = kernelAndRowByRowOutputs(x, layer)
        return zip(got, want).filter { $0.bitPattern != $1.bitPattern }.count
    }

    /// The largest difference between the kernel output and the matvec of MLX
    /// over each row alone, as a share of the largest output, or infinity when
    /// an output is not finite, which no tolerance accepts.
    private func relativeOutputError(_ x: MLXArray, _ layer: QuantizedLinear) -> Float {
        let (got, want) = kernelAndRowByRowOutputs(x, layer)
        let largest = want.map { abs($0) }.max() ?? 0
        let difference = zip(got, want).map { abs($0 - $1) }.max() ?? 0
        return got.allSatisfy { $0.isFinite } ? difference / largest : .infinity
    }

    private func median(_ samples: [Double]) -> Double {
        let sorted = samples.sorted()
        return sorted[sorted.count / medianDivisor]
    }

    func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
