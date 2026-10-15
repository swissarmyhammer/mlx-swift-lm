// Copyright © 2026 Apple Inc.

/// The number of lanes of one SIMD group of an Apple GPU. The Metal kernels of
/// this module write this value into their source as `SIMD_WIDTH`.
let metalSimdWidth = 32

/// The Metal declaration of `SIMD_WIDTH` as ``metalSimdWidth``, one line for
/// the header of each Metal kernel of this module.
let metalSimdWidthDeclaration = """
    constexpr constant int SIMD_WIDTH = \(metalSimdWidth);

    """
