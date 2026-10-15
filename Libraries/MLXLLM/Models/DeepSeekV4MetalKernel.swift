// Copyright © 2026 Apple Inc.
//
// The values and the checks that the DeepSeek-V4 Metal kernels share.

import Foundation
import MLX

/// The values and the checks that every DeepSeek-V4 Metal kernel shares.
///
/// Each kernel file declares only its own layout. The facts of the GPU, of
/// MLX and of the residual stream stand here one time, thus a change to one
/// of them needs one edit.
enum DeepSeekV4MetalKernel {

    /// The number of lanes in one Apple GPU SIMD group.
    static let simdGroupWidth = 32

    /// The number of values one vector load reads: the width of a `float4`.
    static let vectorWidth = 4

    /// The smallest input MLX gives a kernel in the `device` address space.
    ///
    /// MLX gives an input of fewer values in the `constant` address space,
    /// and a `device` pointer to such an input does not compile. Thus a
    /// kernel that can read a smaller input indexes it and keeps no pointer
    /// into it, or takes the pointer type as a template argument, and a
    /// kernel that needs a `device` pointer runs only on an input of this
    /// many values or more.
    static let smallestDeviceInput = 8

    /// The axis of the residual stream that holds the copies.
    static let copyAxis = -2

    /// The last axis of a tensor: the width of one copy of the residual
    /// stream, or the width of one head.
    static let widthAxis = -1

    /// The number of trailing axes of the residual stream that one token
    /// holds: ``copyAxis`` and ``widthAxis``.
    static let tokenAxisCount = 2

    /// Tells if a Metal kernel can run: the default device is the GPU.
    static var deviceIsGPU: Bool {
        Device.defaultDevice().deviceType == .gpu
    }

    /// The number of groups of `groupSize` items that cover `count` items.
    ///
    /// - Parameters:
    ///   - count: The number of items, 0 or more.
    ///   - groupSize: The number of items in one group, 1 or more.
    /// - Returns: The fewest groups that hold every item.
    static func groupCount(covering count: Int, groupSize: Int) -> Int {
        (count + groupSize - 1) / groupSize
    }

    /// The token axes of a residual stream: its shape without the copy axis
    /// and the width axis.
    ///
    /// - Parameter stream: The residual stream, shape `(..., copies, width)`.
    /// - Returns: The leading axes, one for each token axis.
    static func tokenShape(of stream: MLXArray) -> [Int] {
        Array(stream.shape.dropLast(tokenAxisCount))
    }
}
