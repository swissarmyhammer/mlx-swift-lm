// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

/// Helpers that read MLX arrays back into Swift values for the DeepSeek-V4
/// test suites.
///
/// More than one suite compares array values against Swift literals or
/// against a second array. Each of these suites reads the values through
/// these shared helpers, so the reading stays the same in every suite.
///
/// A caller needs no `eval` and no `reshaped(-1)` before a read.
/// `MLXArray.asArray(_:)` evaluates the array itself, and it copies the
/// values out in row-major order for every memory layout: a transposed, a
/// sliced or a broadcast array gives the same list as its contiguous copy.
///
/// More helpers read the graph of an array, not its values. They count the
/// gather nodes in it, and they check the counter against a control graph
/// that holds one known gather.
enum DeepSeekV4TestArrays {

    /// Reads an array back as a flat list of `Float`, in row-major order.
    ///
    /// - Parameter array: The array to read. Its dtype changes to float32
    ///   before the read.
    /// - Returns: Every value of the array, in float32.
    static func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// Reads an array back as a flat list of `Int32`, in row-major order.
    ///
    /// - Parameter array: The array to read. Its dtype changes to int32
    ///   before the read.
    /// - Returns: Every value of the array, in int32.
    static func integers(_ array: MLXArray) -> [Int32] {
        array.asType(.int32).asArray(Int32.self)
    }

    /// The label `exportToDot(_:)` writes for a node of the general MLX
    /// gather. The quotation marks keep `GatherAxis` and `GatherQMM` out of
    /// the count, because those are different primitives.
    private static let gatherNodeLabel = "\"Gather\""

    /// Counts the general gather nodes in the graph that makes `arrays`.
    ///
    /// Each gather node is one more kernel dispatch on each evaluation, thus
    /// a test can hold a path to the gathers it needs.
    ///
    /// - Parameter arrays: The outputs whose graph to read. The function does
    ///   not evaluate them.
    /// - Returns: The number of `Gather` nodes in the graph.
    static func gatherNodeCount(_ arrays: [MLXArray]) -> Int {
        exportToDot(arrays).components(separatedBy: gatherNodeLabel).count - 1
    }

    /// The length of the vector that the control graph reads.
    private static let controlVectorLength = 4

    /// The element that the control graph reads with an `Int` subscript.
    private static let controlElementIndex = 1

    /// The number of general gather nodes in the control graph.
    private static let controlGatherCount = 1

    /// Builds the control graph of the gather counter: a graph that holds
    /// exactly one general gather node.
    ///
    /// An `Int` subscript on a vector calls `take`, and `take` makes one
    /// `Gather` node. The function does not evaluate the graph.
    ///
    /// - Returns: The one element that the subscript reads.
    private static func controlGatherGraph() -> MLXArray {
        MLXArray.zeros([controlVectorLength])[controlElementIndex]
    }

    /// Checks that `gatherNodeCount(_:)` finds the one gather of the control
    /// graph.
    ///
    /// If the dot label changes, or the export gives no match, the counter
    /// returns 0 for each graph. This check then fails, thus a zero count
    /// from the counter is a real result.
    ///
    /// - Parameter sourceLocation: The location that a failure reports.
    static func expectCounterSeesTheControlGather(
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(
            gatherNodeCount([controlGatherGraph()]) == controlGatherCount,
            "the gather counter finds the one gather of the control graph",
            sourceLocation: sourceLocation)
    }

    /// Checks that the graph that makes `arrays` holds no general gather node.
    ///
    /// The check first runs the positive control of the counter, thus a
    /// counter that cannot see a gather fails this check and does not pass it.
    ///
    /// - Parameters:
    ///   - arrays: The outputs whose graph to read. The function does not
    ///     evaluate them.
    ///   - sourceLocation: The location that a failure reports.
    static func expectNoGather(
        _ arrays: [MLXArray], sourceLocation: SourceLocation = #_sourceLocation
    ) {
        expectCounterSeesTheControlGather(sourceLocation: sourceLocation)
        #expect(
            gatherNodeCount(arrays) == 0, "the graph holds no gather node",
            sourceLocation: sourceLocation)
    }
}
