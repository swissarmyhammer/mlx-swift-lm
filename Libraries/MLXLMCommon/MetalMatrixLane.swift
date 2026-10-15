// Copyright © 2026 Apple Inc.

/// The Metal declaration of the lane layout of an 8x8 SIMD-group matrix, for
/// the header of each Metal kernel that fills or reads such a matrix one lane
/// at a time.
///
/// `lane_coordinate` gives the row and the first column of the two
/// neighbouring elements one lane holds, the layout MLX reads in
/// `BaseMMAFrag::get_coord`. `matrixSide` is the side of the matrix and
/// `elementsPerLane` the number of elements of one lane.
package let metalMatrixLaneDeclaration = """
    constexpr constant short matrixSide = 8;
    constexpr constant short elementsPerLane = 2;
    constexpr constant short lanesPerQuad = 4;
    constexpr constant short quadRowBit = 4;
    constexpr constant short quadColumnBit = 2;

    inline short2 lane_coordinate(ushort lane) {
        short quad = lane / lanesPerQuad;
        short row = (quad & quadRowBit) + (lane / elementsPerLane) % lanesPerQuad;
        short column = (quad & quadColumnBit) * elementsPerLane + (lane % elementsPerLane) * elementsPerLane;
        return short2(column, row);
    }

    """
