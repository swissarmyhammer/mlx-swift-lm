// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A model that can split its layers across the processes of a distributed group.
///
/// This is the counterpart of `model.shard(group)` in Python mlx-lm. Each process loads the
/// whole model, then calls ``shard(group:)`` with the same group. Afterwards each process holds
/// its share of every sharded layer, and the layers sum their partial results across the group.
/// Every process then computes the same logits.
public protocol TensorParallelShardable {

    /// Replaces the weights of this process with its share of them.
    ///
    /// A group of one process leaves the model unchanged. A model that cannot split evenly
    /// across the group throws before it changes anything.
    ///
    /// - Parameter group: The group to split across. Every process passes the same group.
    /// - Throws: ``TensorParallelShardingError`` when the model cannot split across the group,
    ///   or `ShardingError` of MLXNN when a quantized projection cannot.
    func shard(group: MLXDistributed.Group) throws
}

/// A model cannot split across a distributed group.
public enum TensorParallelShardingError: Error, Equatable, CustomStringConvertible {

    /// The group size does not divide a count, such as the key and value heads.
    case indivisible(quantity: String, count: Int, groupSize: Int)

    /// Sharding does not handle a part of the model yet.
    case unsupported(feature: String)

    public var description: String {
        switch self {
        case .indivisible(let quantity, let count, let groupSize):
            "\(quantity) \(count) not divisible by \(groupSize)"
        case .unsupported(let feature):
            "tensor parallel sharding does not support \(feature) yet"
        }
    }
}

/// Building blocks for ``TensorParallelShardable`` conformances.
public enum TensorParallel {

    /// The share of `count` that each process of a group keeps.
    ///
    /// - Parameters:
    ///   - count: The count to split, such as the attention heads.
    ///   - quantity: The name of the count, for the error.
    ///   - groupSize: The number of processes.
    /// - Returns: `count / groupSize`.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``.
    public static func share(
        of count: Int, named quantity: String, across groupSize: Int
    ) throws -> Int {
        guard groupSize > 0, count % groupSize == 0 else {
            throw TensorParallelShardingError.indivisible(
                quantity: quantity, count: count, groupSize: groupSize)
        }
        return count / groupSize
    }

    /// Throws unless `layer` is a stock `Linear` or `QuantizedLinear`.
    ///
    /// An adapter or a rotated layer is a subclass whose weights do not split by rows or columns.
    ///
    /// - Parameters:
    ///   - layer: The projection to check.
    ///   - path: The path of the projection, for the error.
    public static func requireStockProjection(_ layer: Linear, at path: String) throws {
        let type = ObjectIdentifier(Swift.type(of: layer))
        guard
            type == ObjectIdentifier(Linear.self) || type == ObjectIdentifier(QuantizedLinear.self)
        else {
            throw TensorParallelShardingError.unsupported(
                feature: "\(Swift.type(of: layer)) at \(path)")
        }
    }

    /// A copy of `layer` whose inputs split across `group`, which sums the partial outputs.
    ///
    /// - Parameters:
    ///   - layer: A stock projection. It does not change.
    ///   - path: The path of the projection, for the error.
    ///   - group: The group to split across.
    /// - Returns: A `ShardedToAllLinear` or `QuantizedShardedToAllLinear`.
    public static func makeShardedToAll(
        _ layer: Linear, at path: String, group: MLXDistributed.Group
    ) throws -> Linear {
        try requireStockProjection(layer, at: path)
        return try shardLinear(layer, sharding: .shardedToAll, group: group)
    }

    /// A copy of the depthwise `conv` that keeps the channels of this process.
    ///
    /// A depthwise conv has one group per channel, so the copy has fewer groups too.
    ///
    /// - Parameters:
    ///   - conv: A depthwise conv. It does not change.
    ///   - segments: The segments of the unsharded channels, each of which splits on its own.
    ///   - group: The group to split across.
    /// - Returns: The conv of this process.
    public static func makeShardedDepthwiseConv(
        _ conv: Conv1d, segments: Segments, group: MLXDistributed.Group
    ) throws -> Conv1d {
        let channels = conv.weight.dim(0)
        guard conv.groups == channels else {
            throw TensorParallelShardingError.unsupported(
                feature: "a conv of \(conv.groups) groups over \(channels) channels")
        }
        let shardChannels = try share(of: channels, named: "conv channels", across: group.size)
        let sharded = Conv1d(
            inputChannels: shardChannels, outputChannels: shardChannels,
            kernelSize: conv.weight.dim(1), stride: conv.stride, padding: conv.padding,
            dilation: conv.dilation, groups: shardChannels, bias: conv.bias != nil)
        sharded.update(parameters: conv.parameters())
        try shardLeadingAxis(of: sharded, ["weight": segments, "bias": segments], group: group)
        return sharded
    }

    /// Splits the leading axis of the named parameters of `owner` across `group`, in place.
    ///
    /// The update goes through `owner`, so an owner that fuses its projections sees it and drops
    /// the fusion. A key names a parameter or a child module, whose parameters all split. Each
    /// value gives the segments of the unsharded axis, each of which splits on its own.
    ///
    /// - Parameters:
    ///   - owner: The module that holds the parameters.
    ///   - segments: The segments of each key's leading axis.
    ///   - group: The group to split across.
    public static func shardLeadingAxis(
        of owner: Module, _ segments: [String: Segments], group: MLXDistributed.Group
    ) throws {
        try shardInPlace(
            owner,
            predicate: { path, _ in
                segments.first { key, _ in path == key || path.hasPrefix("\(key).") }
                    .map { (axis: 0, segments: $0.value) }
            },
            group: group)
    }
}
