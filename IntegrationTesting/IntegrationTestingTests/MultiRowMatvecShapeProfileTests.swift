// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// Times each projection shape of one Qwen3.8-27B step alone, for the rows of a verify, through
/// the matvec of MLX and through ``MultiRowQuantizedMatvec``: which shape adds the cost of a row,
/// and which kernel each row count runs.
@Suite(.serialized)
struct MultiRowMatvecShapeProfileTests {

    /// The rows of a verify at the default block size and one above it.
    private static let measuredRows = 1 ... 6

    /// The measured row counts the kernel runs.
    private static let kernelRows = 3 ... 6

    @Test func qwen38ProjectionCostByShapeAndRows() throws {
        let benchmark = MultiRowQuantizedMatvecBenchmarkTests()
        for format in qwen38Formats {
            for projection in qwen38StepProjections {
                let layers = benchmark.makeLayers(projection, format)
                let layer = try #require(layers.first)
                for rows in Self.measuredRows {
                    let x = MLXRandom.normal([rows, projection.inputs]).asType(.bfloat16)
                    #expect(
                        MultiRowQuantizedMatvec.canRun(x, layer)
                            == Self.kernelRows.contains(rows))
                    benchmark.expectRowsMatchTheirOwnMatvec(rows: rows, layer)
                    let mlx = benchmark.medianStepMilliseconds(layers, rows: rows) { $1($0) }
                    let kernel = benchmark.medianStepMilliseconds(layers, rows: rows) {
                        MultiRowQuantizedMatvec.quantizedMatmulRows($0, $1)
                    }
                    #expect(mlx.isFinite && mlx > 0)
                    #expect(kernel.isFinite && kernel > 0)
                    benchmark.report(
                        String(
                            format:
                                "[MULTIROW-SHAPE] %@ %dx%d x%d M=%d: MLX %.3f ms, kernel %.3f ms",
                            format.name, projection.outputs, projection.inputs, projection.count,
                            rows, mlx, kernel))
                }
            }
        }
    }
}
