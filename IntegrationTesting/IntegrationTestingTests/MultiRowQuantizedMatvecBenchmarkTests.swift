// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// One quantized projection of a Qwen3.8-27B decode step, and the number of
/// times one step runs it.
private struct StepProjection {
    let inputs: Int
    let outputs: Int
    let count: Int
}

/// The quantized projections of one Qwen3.8-27B step: the fused input and the
/// output projection of 48 gated delta net layers, the four projections of 16
/// full-attention layers, the three projections of 64 MLPs, and the LM head.
private let qwen38StepProjections = [
    StepProjection(inputs: 5120, outputs: 16480, count: 48),
    StepProjection(inputs: 6144, outputs: 5120, count: 48),
    StepProjection(inputs: 5120, outputs: 12288, count: 16),
    StepProjection(inputs: 5120, outputs: 1024, count: 32),
    StepProjection(inputs: 6144, outputs: 5120, count: 16),
    StepProjection(inputs: 5120, outputs: 17408, count: 128),
    StepProjection(inputs: 17408, outputs: 5120, count: 64),
    StepProjection(inputs: 5120, outputs: 248_320, count: 1),
]

/// A quantization format of the Qwen3.8-27B checkpoints.
private struct QuantizationFormat {
    let name: String
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
}

private let qwen38Formats = [
    QuantizationFormat(name: "4-bit affine g64", groupSize: 64, bits: 4, mode: .affine),
    QuantizationFormat(name: "mxfp4", groupSize: 32, bits: 4, mode: .mxfp4),
]

/// The verify rows a speculative step runs: one plain decode row up to eight.
private let measuredRows = 1 ... 8
private let warmupRuns = 2
private let timedRuns = 7
/// The median of an odd count of samples stands at half the count.
private let medianDivisor = 2
private let millisecondsPerSecond = 1e3
private let attosecondsPerMillisecond = 1e15
/// The bound of the random weights, about the scale of a trained projection.
private let weightRange: Float = 0.02

/// One way to run a quantized projection over a batch of rows.
private typealias RowProjection = (MLXArray, QuantizedLinear) -> MLXArray

/// Times the quantized matmuls of one Qwen3.8-27B step for 1 to 8 rows, through
/// the matvec of MLX and through ``MultiRowQuantizedMatvec``, with distinct
/// random weights for every layer so no weight stays in cache from one
/// projection to the next.
@Suite(.serialized)
struct MultiRowQuantizedMatvecBenchmarkTests {

    @Test func qwen38StepMatmulCostByRows() {
        for format in qwen38Formats {
            let layers = makeStepLayers(format)
            let mlxOneRow = medianStepMilliseconds(layers, rows: 1) { $1($0) }
            for rows in measuredRows {
                let mlx = medianStepMilliseconds(layers, rows: rows) { $1($0) }
                let kernel = medianStepMilliseconds(layers, rows: rows) {
                    MultiRowQuantizedMatvec.quantizedMatmulRows($0, $1)
                }
                #expect(mismatchedRowValues(layers, rows: rows) == 0)
                report(
                    String(
                        format:
                            "[MULTIROW] %@ M=%d: MLX %.2f ms (%.2fx), kernel %.2f ms (%.2fx M=1)",
                        format.name, rows, mlx, mlx / mlxOneRow, kernel, kernel / mlxOneRow))
            }
        }
    }

    private func makeStepLayers(_ format: QuantizationFormat) -> [QuantizedLinear] {
        qwen38StepProjections.flatMap { projection in
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
    }

    private func medianStepMilliseconds(
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

    /// The count of values where the kernel output of the LM head (the last
    /// layer) differs in bits from the matvec of MLX over each row alone.
    private func mismatchedRowValues(_ layers: [QuantizedLinear], rows: Int) -> Int {
        guard let layer = layers.last else { return 0 }
        let x = MLXRandom.normal([rows, layer.shape.1]).asType(.bfloat16)
        let got = MultiRowQuantizedMatvec.quantizedMatmulRows(x, layer).asType(.float32)
        let want = concatenated((0 ..< rows).map { layer(x[$0 ..< ($0 + 1)]) }, axis: 0)
            .asType(.float32)
        return zip(got.asArray(Float.self), want.asArray(Float.self))
            .filter { $0.bitPattern != $1.bitPattern }
            .count
    }

    private func median(_ samples: [Double]) -> Double {
        let sorted = samples.sorted()
        return sorted[sorted.count / medianDivisor]
    }

    private func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
