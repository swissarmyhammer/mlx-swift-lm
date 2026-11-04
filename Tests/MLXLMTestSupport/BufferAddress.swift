// Copyright © 2026 Apple Inc.

import MLX

/// The buffer of an array, for tests that check whether two arrays share one.
public enum BufferAddress {

    /// The address of the first byte of the buffer of `array`, or `nil` for an empty buffer.
    ///
    /// - Parameter array: The array; it is evaluated.
    /// - Returns: The address.
    public static func of(_ array: MLXArray) -> Int? {
        array.asData(access: .noCopy).data.withUnsafeBytes { bytes in
            bytes.baseAddress.map { Int(bitPattern: $0) }
        }
    }
}
