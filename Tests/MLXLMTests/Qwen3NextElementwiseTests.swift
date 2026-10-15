// Pins the compiled elementwise steps of Qwen3-Next and Qwen3.5 bit for bit
// against the plain MLX ops they replace. Prefill and decode both run these
// steps, thus a mismatch would change the output of the model.

import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen3NextElementwiseTests: XCTestCase {

    /// The dtypes that the models run in.
    private static let dtypes: [DType] = [.float16, .bfloat16, .float32]

    /// The half-precision dtypes, where a float32 step rounds differently
    /// from a step in the dtype of the operands.
    private static let halfDtypes: [DType] = [.float16, .bfloat16]

    /// A small `(batch, tokens, heads, width)` operand shape of a prefill block.
    private static let operandShape = [1, 64, 4, 32]

    /// The width of the operand, which is the width of the norm weight.
    private static var operandWidth: Int { operandShape[operandShape.count - 1] }

    /// The standard deviation of the random operands: a wide range, so that
    /// the sigmoid and the SiLU see their saturated tails as well.
    private static let operandScale: Float = 4

    /// The epsilon of the gated RMSNorm in the Qwen3.5 configuration.
    private static let normEps: Float = 1e-6

    /// The seed of the random operands.
    private static let seed: UInt64 = 20_261_015

    /// Three random operands of `operandShape` in `dtype`.
    private func operands(_ dtype: DType) -> (MLXArray, MLXArray, MLXArray) {
        withRandomState(MLXRandom.RandomState(seed: Self.seed)) {
            let shape = Self.operandShape
            let scale = Self.operandScale
            return (
                (MLXRandom.normal(shape) * scale).asType(dtype),
                (MLXRandom.normal(shape) * scale).asType(dtype),
                (MLXRandom.normal(shape) * scale).asType(dtype)
            )
        }
    }

    private func assertSameBits(
        _ got: MLXArray, _ want: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.dtype, want.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(got.shape, want.shape, "\(label): shape", file: file, line: line)
        let mismatches = BitwiseComparison.mismatchCount(got, want)
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(got.size) numbers differ", file: file,
            line: line)
    }

    /// The plain ops of the precise SwiGLU of the gated RMSNorm.
    private func plainPreciseSwiGLU(
        hiddenStates: MLXArray, gate: MLXArray, normed: MLXArray
    ) -> MLXArray {
        (silu(gate.asType(.float32)) * normed.asType(.float32)).asType(hiddenStates.dtype)
    }

    func testSigmoidMultiplyMatchesPlainOps() {
        for dtype in Self.dtypes {
            let (x, gate, _) = operands(dtype)
            assertSameBits(
                Qwen3NextElementwise.sigmoidMultiply(x, gate), x * sigmoid(gate), "\(dtype)")
        }
    }

    func testSiluProductMatchesPlainOps() {
        for dtype in Self.dtypes {
            let (gate, up, _) = operands(dtype)
            assertSameBits(compiledSiluProduct(gate, up), silu(gate) * up, "\(dtype)")
        }
    }

    func testPreciseSwiGLUMatchesPlainOps() {
        for dtype in Self.dtypes {
            let (hiddenStates, gate, normed) = operands(dtype)
            assertSameBits(
                Qwen3NextElementwise.preciseSwiGLU(hiddenStates, gate, normed),
                plainPreciseSwiGLU(hiddenStates: hiddenStates, gate: gate, normed: normed),
                "\(dtype)")
        }
    }

    /// The gated RMSNorm module, with a weight in the dtype of the operands
    /// and with the float32 weight a new module holds.
    func testGatedRMSNormMatchesPlainOps() {
        for dtype in Self.dtypes {
            let (hiddenStates, gate, weightSource) = operands(dtype)
            let weights = [MLXArray.ones([Self.operandWidth]), weightSource[0, 0, 0]]
            for weight in weights {
                let norm = Qwen3NextRMSNormGated(dimensions: Self.operandWidth, eps: Self.normEps)
                norm.update(parameters: ModuleParameters.unflattened(["weight": weight]))
                let normed = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: Self.normEps)
                assertSameBits(
                    norm(hiddenStates, gate: gate),
                    plainPreciseSwiGLU(hiddenStates: hiddenStates, gate: gate, normed: normed),
                    "\(dtype), weight \(weight.dtype)")
            }
        }
    }

    /// Positive control: the comparison sees one rounding. The precise SwiGLU
    /// rounds once, in float32; the same ops in a half dtype round after each
    /// op. A comparison that cannot fail would report the same bits here too.
    func testComparisonSeesHalfPrecisionRounding() {
        for dtype in Self.halfDtypes {
            let (hiddenStates, gate, normed) = operands(dtype)
            let precise = Qwen3NextElementwise.preciseSwiGLU(hiddenStates, gate, normed)
            XCTAssertGreaterThan(
                BitwiseComparison.mismatchCount(precise, silu(gate) * normed), 0, "\(dtype)")
        }
    }
}
