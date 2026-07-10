// Copyright © 2026 Apple Inc.
//
// Proves the `swift test` metallib PATH RESOLUTION fix (kanban 23ff1zx,
// memory note `swiftpm-test-gpu-metallib-limit`): mlx-swift's
// `load_default_library` (`Cmlx`, `mlx/backend/metal/device.cpp`) probes a
// fixed list of locations for `default.metallib`, none of which the running
// binary satisfies under plain `swift test` (see `MetalLibraryTestBootstrap`
// for the full explanation). Before that bootstrap runs, any MLXArray eval
// on the GPU device aborts the whole test process with "Failed to load the
// default metallib" -- this suite is the first thing in the target that
// forces a real GPU eval, so it is where that crash would surface.

import Foundation
import MLX
import Testing

@Suite("Metal library bootstrap for swift test")
struct MetalLibraryBootstrapTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    @Test("slice, concatenate, and compare MLXArrays on the default (GPU) device")
    func gpuTensorEvalSucceeds() {
        #expect(Device.defaultDevice().deviceType == .gpu)

        let a = MLXArray(0..<12, [3, 4])
        let b = MLXArray(12..<24, [3, 4])

        // a[1...], cols 0..<2 -> [[4, 5], [8, 9]]
        let aSlice = a[1..., 0..<2]
        // b[0..<2], cols 2... -> [[14, 15], [18, 19]]
        let bSlice = b[0..<2, 2...]

        let combined = concatenated([aSlice, bSlice], axis: 0)
        eval(combined)

        #expect(combined.shape == [4, 2])
        let expected = MLXArray([4, 5, 8, 9, 14, 15, 18, 19], [4, 2])
        #expect((combined .== expected).all().item())
    }
}
