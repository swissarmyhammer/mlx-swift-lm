// Copyright © 2026 Apple Inc.
//
// The relative root-mean-square error, the one accuracy metric that the
// numeric tests of this target measure an output with.

import Foundation
import MLX

/// The smallest power of a reference, which keeps an all-zero reference from
/// dividing by zero.
private let referencePowerFloor: Float = 1e-10

/// The relative RMS error of an output against a reference: the root mean
/// square of their gap, as a part of the root mean square of the reference,
/// `sqrt(mean((reference - actual)²) / mean(reference²))`, in float32.
///
/// - Parameters:
///   - reference: The expected output, in any float dtype.
///   - actual: The output under test, in any float dtype.
/// - Returns: The relative error.
func relativeRMSError(_ reference: MLXArray, _ actual: MLXArray) -> Float {
    let reference = reference.asType(.float32)
    let gap = reference - actual.asType(.float32)
    let gapPower = mean(gap * gap).item(Float.self)
    let referencePower = mean(reference * reference).item(Float.self)
    return sqrt(gapPower / max(referencePower, referencePowerFloor))
}
