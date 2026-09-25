import MLX
import MLXTestPrecision
import Testing

/// The float32 tests of this bundle compare two computation paths with float32
/// tolerances (for example the cold check of `ExecutorPromptCacheQwenFileTests`).
/// On a GPU with neural accelerators (for example the Apple M5), MLX computes
/// float32 matrix operations in TF32 for some shapes only, and the two paths
/// then differ by several TF32 units of roundoff (2^-11).
///
/// The `MLXTestPrecision` target sets `MLX_ENABLE_TF32=0` when the bundle
/// loads. This suite makes sure that the setting is in effect. A failure here
/// explains the tolerance failures in the model tests.
@Suite("Float32 precision of the test process")
struct Float32PrecisionTests {

    /// The number of rows of the left matrix and the columns of the right matrix.
    /// MLX sends this shape to the NAX kernel on a GPU with neural accelerators.
    private static let outerSize = 48

    /// The shared dimension of the two matrices.
    private static let innerSize = 32

    /// The seed of the random matrices.
    private static let seed: UInt64 = 0

    /// The largest difference between the GPU product and the CPU product. Two
    /// float32 products agree to approximately 1e-6. A TF32 product differs by
    /// approximately 1e-3.
    private static let float32Tolerance: Float = 1e-4

    @Test("MLX computes float32 matrix operations in float32, not TF32")
    func tf32IsDisabledForTheTestProcess() {
        #expect(mlx_test_precision_tf32_is_disabled())
    }

    @Test("a float32 matmul on the GPU agrees with the same matmul on the CPU")
    func gpuMatmulAgreesWithCPUMatmul() {
        let key = MLXRandom.key(Self.seed)
        let keys = MLXRandom.split(key: key)
        let lhs = MLXRandom.uniform(
            0 ..< 1, [Self.outerSize, Self.innerSize], key: keys.0)
        let rhs = MLXRandom.uniform(
            0 ..< 1, [Self.innerSize, Self.outerSize], key: keys.1)

        let gpuProduct = matmul(lhs, rhs, stream: .gpu)
        let cpuProduct = matmul(lhs, rhs, stream: .cpu)

        #expect(gpuProduct.dtype == .float32)
        #expect(abs(gpuProduct - cpuProduct).max().item(Float.self) <= Self.float32Tolerance)
    }
}
