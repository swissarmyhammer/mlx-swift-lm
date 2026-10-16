import Foundation
import MLX
import MLXLMCommon
import XCTest

final class Gemma4FusionTests: XCTestCase {
    private func assertIdentical(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self).map(\.bitPattern),
            expected.asType(.float32).asArray(Float.self).map(\.bitPattern),
            file: file, line: line)
    }

    func testSoftcapPreservesScalarPromotionAndRuntimeCaps() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let x = MLX.linspace(Float(-100), Float(100), count: 4096).reshaped(1, 4096)
                .asType(dtype)
            for value: Float in [30, 7.25, 1] {
                for cap in [MLXArray(value), value.asMLXArray(dtype: dtype)] {
                    assertIdentical(gemma4LogitSoftcap(x, cap), tanh(x / cap) * cap)
                    let strided = x[.ellipsis, .stride(by: 2)]
                    assertIdentical(
                        gemma4LogitSoftcap(strided, cap), tanh(strided / cap) * cap)
                }
            }
        }
    }

    func testGradientsMatchOriginal() {
        let cap = MLXArray(Float(30))
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-3), Float(3), count: 64)
                .reshaped(1, 1, 64).asType(dtype)
            assertIdentical(
                grad({ sum(gemma4LogitSoftcap($0, cap)) })(logits),
                grad({ sum(tanh($0 / cap) * cap) })(logits))
        }
    }

    func testScalarSoftcapPreservesDtypeAndRuntimeCaps() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-100), Float(100), count: 4096)
                .reshaped(2, 2, 1024).asType(dtype)
            for cap: Float in [30, 7.25, 30.1, 1, -30] {
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
                let strided = logits[.ellipsis, .stride(by: 2)]
                assertIdentical(gemma4LogitSoftcap(strided, cap), tanh(strided / cap) * cap)
            }
        }
    }

    func testScalarSoftcapPreservesEveryFiniteLowPrecisionValue() {
        for dtype in [DType.float16, .bfloat16] {
            let exponentMask: UInt16 = dtype == .float16 ? 0x7c00 : 0x7f80
            let bits = (0 ... 65535).map { UInt16($0) }.filter {
                $0 & exponentMask != exponentMask
            }
            let logits = MLXArray(bits).view(dtype: dtype)
            for cap: Float in [30, 7.25, 30.1, 1, -30] {
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
            }
        }
    }

    func testScalarSoftcapGradientsMatchOriginal() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-3), Float(3), count: 64)
                .reshaped(1, 1, 64).asType(dtype)
            let cap: Float = 30
            assertIdentical(
                grad({ sum(gemma4LogitSoftcap($0, cap)) })(logits),
                grad({ sum(tanh($0 / cap) * cap) })(logits))
        }
    }

    func testScalarSoftcapPreservesSpecialValues() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLXArray([-Float.infinity, -100, -0.0, 0.0, 100, .infinity, .nan])
                .asType(dtype)
            for cap: Float in [30, -30, 0] {
                let actual = gemma4LogitSoftcap(logits, cap)
                let expected = tanh(logits / cap) * cap
                XCTAssertEqual(actual.dtype, expected.dtype)
                for (a, b) in zip(
                    actual.asType(.float32).asArray(Float.self),
                    expected.asType(.float32).asArray(Float.self))
                {
                    if b.isNaN {
                        XCTAssertTrue(a.isNaN)
                    } else {
                        XCTAssertEqual(a.bitPattern, b.bitPattern)
                    }
                }
            }
        }
    }

    func testScalarSoftcapOnCPU() {
        Stream.withNewDefaultStream(device: .cpu) {
            for dtype in [DType.float16, .bfloat16, .float32] {
                let logits = MLX.linspace(Float(-100), Float(100), count: 128).asType(dtype)
                let cap: Float = 30
                assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
                let tensorCap = MLXArray(cap)
                assertIdentical(
                    gemma4LogitSoftcap(logits, tensorCap), tanh(logits / tensorCap) * tensorCap)
            }
        }
        for dtype in [DType.float16, .bfloat16, .float32] {
            let logits = MLX.linspace(Float(-100), Float(100), count: 128).asType(dtype)
            let cap: Float = 30
            assertIdentical(gemma4LogitSoftcap(logits, cap), tanh(logits / cap) * cap)
            let tensorCap = MLXArray(cap)
            assertIdentical(
                gemma4LogitSoftcap(logits, tensorCap), tanh(logits / tensorCap) * tensorCap)
        }
    }
}
