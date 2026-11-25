// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A model that can split its layers across the processes of a distributed group.
///
/// This is the counterpart of `model.shard(group)` in Python mlx-lm. Each process loads the
/// model, then calls ``shard(group:)`` with the same group. Afterwards each process holds its
/// share of every sharded layer, and the layers sum their partial results across the group.
/// Every process then computes the same logits.
///
/// A load that keeps its weights lazy splits layer by layer instead, through
/// ``shardLayers(group:hooks:)``, so no process holds more than one layer's full weights; see
/// ``loadShardedModelContext(from:using:group:willShardLayer:)``.
public protocol TensorParallelShardable {

    /// Replaces the weights of this process with its share of them, one layer at a time, and
    /// calls `hooks` around the split of each layer.
    ///
    /// Every layer's shares are checked before any layer changes, so a model that cannot split
    /// evenly across the group throws before it changes anything, and before any hook runs. A
    /// group of one process leaves the model unchanged and calls no hook.
    ///
    /// - Parameters:
    ///   - group: The group to split across. Every process passes the same group.
    ///   - hooks: The calls around the split of each layer.
    /// - Throws: ``TensorParallelShardingError`` when the model cannot split across the group,
    ///   `ShardingError` of MLXNN when a quantized projection cannot, or the error of a hook.
    func shardLayers(group: MLXDistributed.Group, hooks: TensorParallelLayerHooks) throws
}

extension TensorParallelShardable {

    /// Replaces the weights of this process with its share of them.
    ///
    /// A group of one process leaves the model unchanged. A model that cannot split evenly
    /// across the group throws before it changes anything.
    ///
    /// - Parameter group: The group to split across. Every process passes the same group.
    /// - Throws: ``TensorParallelShardingError`` when the model cannot split across the group,
    ///   or `ShardingError` of MLXNN when a quantized projection cannot.
    public func shard(group: MLXDistributed.Group) throws {
        try shardLayers(group: group, hooks: .none)
    }
}

/// The calls around the split of each layer in ``TensorParallelShardable/shardLayers(group:hooks:)``.
///
/// Each call takes the index of the layer and the layer module, whose parameters are the full
/// weights in `willShard` and the shares of this process in `didShard`. A sharded load evaluates
/// the layer in each, so the full weights of one layer at most stand beside the shards.
public struct TensorParallelLayerHooks {

    /// A call on one layer: its index, and its module.
    public typealias Hook = (_ index: Int, _ layer: Module) throws -> Void

    /// Runs before the layer splits.
    public let willShard: Hook

    /// Runs after the layer split.
    public let didShard: Hook

    /// - Parameters:
    ///   - willShard: Runs before each layer splits.
    ///   - didShard: Runs after each layer split.
    public init(willShard: @escaping Hook, didShard: @escaping Hook) {
        self.willShard = willShard
        self.didShard = didShard
    }

    /// Hooks that do nothing, for a split of a model already in memory.
    public static var none: Self {
        Self(willShard: { _, _ in }, didShard: { _, _ in })
    }
}

/// A model cannot split across a distributed group.
public enum TensorParallelShardingError: Error, Equatable, CustomStringConvertible {

    /// The group size does not divide a count, such as the key and value heads.
    case indivisible(quantity: String, count: Int, groupSize: Int)

    /// Sharding does not handle a part of the model yet.
    case unsupported(feature: String)

    /// The model, of the named type, does not conform to ``TensorParallelShardable``.
    case notShardable(modelType: String)

    /// The reason in words, naming the count, the feature or the model type.
    public var description: String {
        switch self {
        case .notShardable(let modelType):
            "\(modelType) does not conform to TensorParallelShardable"
        case .indivisible(let quantity, let count, let groupSize):
            "\(quantity) \(count) not divisible by \(groupSize)"
        case .unsupported(let feature):
            "tensor parallel sharding does not support \(feature) yet"
        }
    }
}

/// Building blocks for ``TensorParallelShardable`` conformances.
public enum TensorParallel {

    /// Splits `layers` one at a time, after every layer's shards are made.
    ///
    /// `makeShards` builds the replacements of a layer without changing it, and throws when the
    /// layer cannot split, so a model that cannot split throws before any layer changes. Each
    /// layer then splits through `shard`, between the two calls of `hooks`.
    ///
    /// - Parameters:
    ///   - layers: The layers, in order.
    ///   - hooks: The calls around the split of each layer.
    ///   - makeShards: Builds and checks the replacements of one layer.
    ///   - shard: Splits one layer with its replacements.
    public static func shardLayers<Layer: Module, Shards>(
        _ layers: [Layer], hooks: TensorParallelLayerHooks,
        makeShards: (Layer) throws -> Shards, shard: (Layer, Shards) throws -> Void
    ) throws {
        let shards = try layers.map(makeShards)
        for (index, (layer, layerShards)) in zip(layers, shards).enumerated() {
            try hooks.willShard(index, layer)
            try shard(layer, layerShards)
            try hooks.didShard(index, layer)
        }
    }

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

    /// The quantization of a projection, whose inputs split only at the edge of a group.
    public struct QuantizationGroups: Equatable, Sendable {
        /// The quantization mode, such as affine, mxfp4 or nvfp4.
        public let mode: QuantizationMode
        /// The bits of each quantized value.
        public let bits: Int
        /// The inputs of each group.
        public let groupSize: Int

        /// - Parameters:
        ///   - mode: The quantization mode.
        ///   - bits: The bits of each quantized value.
        ///   - groupSize: The inputs of each group.
        public init(mode: QuantizationMode, bits: Int, groupSize: Int) {
            self.mode = mode
            self.bits = bits
            self.groupSize = groupSize
        }

        /// The quantization of `layer`, or nil for a float projection.
        init?(of layer: Linear) {
            guard let quantized = layer as? QuantizedLinear else { return nil }
            self.init(mode: quantized.mode, bits: quantized.bits, groupSize: quantized.groupSize)
        }
    }

    /// Throws unless every process keeps whole rows of each segment of `rows`.
    ///
    /// The scales and biases of a quantized projection hold one row per output, so whole rows
    /// keep whole groups.
    ///
    /// - Parameters:
    ///   - rows: The rows of the unsharded projection.
    ///   - segments: The segments of the rows, each of which splits on its own.
    ///   - path: The path of the projection, for the error.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``.
    public static func requireWholeRows(
        _ rows: Int, segments: Segments = .count(1), at path: String, across groupSize: Int
    ) throws {
        for segment in try segmentLengths(of: rows, segments, at: path) {
            _ = try share(of: segment, named: "\(path) rows", across: groupSize)
        }
    }

    /// Throws unless every process keeps whole rows of each segment of `layer`.
    ///
    /// - Parameters:
    ///   - layer: The projection whose rows split.
    ///   - segments: The segments of the rows, each of which splits on its own.
    ///   - path: The path of the projection, for the error.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``.
    public static func requireWholeRows(
        of layer: Linear, segments: Segments = .count(1), at path: String,
        across groupSize: Int
    ) throws {
        try requireWholeRows(layer.shape.0, segments: segments, at: path, across: groupSize)
    }

    /// Throws unless every process keeps an equal run of `inputs` that holds whole groups of
    /// `quantization`.
    ///
    /// - Parameters:
    ///   - inputs: The inputs of the unsharded projection.
    ///   - quantization: The quantization of the projection, or nil for a float one.
    ///   - path: The path of the projection, for the error.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``.
    public static func requireWholeInputs(
        _ inputs: Int, quantization: QuantizationGroups?, at path: String, across groupSize: Int
    ) throws {
        guard let quantization else {
            _ = try share(of: inputs, named: "\(path) inputs", across: groupSize)
            return
        }
        let groups = inputs / quantization.groupSize
        guard groups * quantization.groupSize == inputs, groups % groupSize == 0 else {
            throw TensorParallelShardingError.indivisible(
                quantity: """
                    \(path) quantization groups (\(quantization.mode.rawValue), \
                    \(quantization.bits)-bit, \(quantization.groupSize) inputs each)
                    """,
                count: groups, groupSize: groupSize)
        }
    }

    /// Throws unless every process keeps an equal run of the inputs of `layer` that holds whole
    /// quantization groups.
    ///
    /// - Parameters:
    ///   - layer: The projection whose inputs split.
    ///   - path: The path of the projection, for the error.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``.
    public static func requireWholeInputs(
        of layer: Linear, at path: String, across groupSize: Int
    ) throws {
        try requireWholeInputs(
            layer.shape.1, quantization: QuantizationGroups(of: layer), at: path,
            across: groupSize)
    }

    /// The projections of a layer that split, by the axis they split on.
    public struct SplitProjections {
        /// The projections whose rows split, with the segments of their rows.
        public let rows: [(path: String, projection: Linear, segments: Segments)]
        /// The projections whose inputs split.
        public let inputs: [(path: String, projection: Linear)]

        /// - Parameters:
        ///   - rows: The projections whose rows split, by path, with their segments.
        ///   - inputs: The projections whose inputs split, by path.
        public init(
            rows: [(path: String, projection: Linear, segments: Segments)],
            inputs: [(path: String, projection: Linear)]
        ) {
            self.rows = rows
            self.inputs = inputs
        }

        /// Throws unless every projection whose rows split is a stock projection. A projection
        /// whose inputs split is checked as ``TensorParallel/makeShardedToAll(_:at:group:)``
        /// copies it.
        public func requireStockRowProjections() throws {
            for row in rows {
                try TensorParallel.requireStockProjection(row.projection, at: row.path)
            }
        }

        /// Throws unless each process keeps whole rows of every projection whose rows split,
        /// and whole quantization groups of every projection whose inputs split.
        ///
        /// - Parameter groupSize: The number of processes.
        /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)``
        ///   naming the first projection that does not split.
        public func requireWholeShares(across groupSize: Int) throws {
            for row in rows {
                try TensorParallel.requireWholeRows(
                    of: row.projection, segments: row.segments, at: row.path, across: groupSize)
            }
            for input in inputs {
                try TensorParallel.requireWholeInputs(
                    of: input.projection, at: input.path, across: groupSize)
            }
        }
    }

    /// The lengths of the segments of an axis of length `dimension`.
    private static func segmentLengths(
        of dimension: Int, _ segments: Segments, at path: String
    ) throws -> [Int] {
        let boundaries: [Int]
        switch segments {
        case .count(let count):
            let length = try share(of: dimension, named: "\(path) segments", across: count)
            boundaries = (1 ..< count).map { $0 * length }
        case .indices(let indices):
            boundaries = indices
        case .fractions(let fractions):
            boundaries = fractions.map { Int($0 * Double(dimension)) }
        }
        let edges = [0] + boundaries + [dimension]
        return zip(edges, edges.dropFirst()).map { $1 - $0 }
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
        try shardAxes(
            of: owner, segments.mapValues { Split(axis: 0, segments: $0) }, group: group)
    }

    /// The axis a parameter splits on, and the segments of that axis.
    public struct Split: Equatable, Sendable {
        /// The axis to split.
        public let axis: Int
        /// The segments of the unsharded axis, each of which splits on its own.
        public let segments: Segments

        /// - Parameters:
        ///   - axis: The axis to split.
        ///   - segments: The segments of the unsharded axis.
        public init(axis: Int, segments: Segments) {
            self.axis = axis
            self.segments = segments
        }
    }

    /// The split of the parameter at `path`, or nil when it stays whole.
    ///
    /// A key names a parameter, or a child module whose parameters all split. When several
    /// keys name the parameter, as `conv` and `conv.weight` do, the longest one decides.
    ///
    /// - Parameters:
    ///   - path: The path of the parameter in the owner, such as `q_proj.weight`.
    ///   - splits: The split of each key.
    public static func split(of path: String, in splits: [String: Split]) -> Split? {
        splits.filter { key, _ in path == key || path.hasPrefix("\(key).") }
            .max { $0.key.count < $1.key.count }?.value
    }

    /// Splits the named parameters of `owner` across `group` on their own axes, in place.
    ///
    /// The update goes through `owner`, as in ``shardLeadingAxis(of:_:group:)``.
    ///
    /// - Parameters:
    ///   - owner: The module that holds the parameters.
    ///   - splits: The split of each key; see ``split(of:in:)``.
    ///   - group: The group to split across.
    public static func shardAxes(
        of owner: Module, _ splits: [String: Split], group: MLXDistributed.Group
    ) throws {
        try shardInPlace(
            owner,
            predicate: { path, _ in
                split(of: path, in: splits).map { (axis: $0.axis, segments: $0.segments) }
            },
            group: group)
    }
}
