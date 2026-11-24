// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The share of an Inkling 2 text stack that each process of a group keeps.
///
/// Each process keeps whole attention heads, and a slice of the inner width of every expert.
/// Query head `h` reads key head `h / (heads / kvHeads)`, so a contiguous run of key heads keeps
/// the query heads that read them. The router, the norms, the loop gates and the depth
/// attention act on the whole residual stream and stay whole.
struct Inkling2TensorParallelPlan: Equatable {

    /// The number of processes.
    let groupSize: Int

    /// The full-attention query heads of each process.
    let attentionHeads: Int

    /// The full-attention key and value heads of each process.
    let kvHeads: Int

    /// The sliding-window query heads of each process.
    let slidingAttentionHeads: Int

    /// The sliding-window key and value heads of each process.
    let slidingKVHeads: Int

    /// The inner width of each expert on each process.
    let expertWidth: Int

    /// - Parameters:
    ///   - configuration: The configuration of the unsharded text stack.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)`` when the
    ///   group does not divide a count.
    init(configuration: Inkling2TextConfiguration, groupSize: Int) throws {
        func share(_ count: Int, _ quantity: String) throws -> Int {
            try TensorParallel.share(of: count, named: quantity, across: groupSize)
        }
        self.groupSize = groupSize
        attentionHeads = try share(configuration.attentionHeads, "attentionHeads")
        kvHeads = try share(configuration.kvHeads, "kvHeads")
        slidingAttentionHeads = try share(
            configuration.slidingAttentionHeads, "slidingAttentionHeads")
        slidingKVHeads = try share(configuration.slidingKVHeads, "slidingKVHeads")
        expertWidth = try share(configuration.intermediateSize, "intermediateSize")
    }

    /// The query and key-value heads of each process in a layer of either kind.
    func heads(sliding: Bool) -> (heads: Int, kvHeads: Int) {
        sliding ? (slidingAttentionHeads, slidingKVHeads) : (attentionHeads, kvHeads)
    }

    /// One whole run per process.
    private static let byRows = TensorParallel.Split(axis: 0, segments: .count(1))

    /// The splits of the attention parameters of a layer of either kind, which hold whole heads
    /// per process. `o_proj` becomes a sharded-to-all layer instead, and the relative logit
    /// table, which every head reads, stays whole.
    func attentionSplits(sliding: Bool) -> [String: TensorParallel.Split] {
        var splits = Dictionary(
            uniqueKeysWithValues: [
                "q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj", "q_l2_clip",
                "rel_l2_clip",
            ].map { ($0, Self.byRows) })
        for convolution in ["k_sconv", "v_sconv"] {
            if sliding {
                splits["\(convolution).conv1d"] = Self.byRows
            } else {
                // The static taps are `[taps, channels]`; the tap projection is head-major.
                splits["\(convolution).static_weight"] = TensorParallel.Split(
                    axis: 1, segments: .count(1))
                splits["\(convolution).weight_proj"] = Self.byRows
            }
        }
        return splits
    }

    /// The axis of the gate and up rows, `[experts, 2 * width, inputs]`, which alternate per
    /// channel, so one contiguous run keeps each channel's pair.
    private static let gateUpRowAxis = 1

    /// The axis of the packed inputs of the down projection, `[experts, outputs, inputs]`.
    private static let downInputAxis = 2

    /// The splits of the parameters of the experts. The global scale of each expert stays whole.
    static let expertSplits: [String: TensorParallel.Split] = Dictionary(
        uniqueKeysWithValues: ["weight", "scales", "biases"].flatMap { suffix in
            [
                (
                    "gate_up_proj.\(suffix)",
                    TensorParallel.Split(axis: gateUpRowAxis, segments: .count(1))
                ),
                (
                    "down_proj.\(suffix)",
                    TensorParallel.Split(axis: downInputAxis, segments: .count(1))
                ),
            ]
        })

    /// Throws unless the inputs of the down projection split into whole quantization groups.
    ///
    /// - Parameter expertGroupSize: The quantization group size of the down projection.
    func requireWholeQuantizationGroups(expertGroupSize: Int) throws {
        let wholeWidth = expertWidth * groupSize
        _ = try TensorParallel.share(
            of: wholeWidth / expertGroupSize, named: "expert quantization groups",
            across: groupSize)
    }
}

/// The modules that replace parts of one decoder layer, made before anything changes.
struct Inkling2LayerShards {

    /// The output projection of the attention.
    let output: Linear
}

extension Inkling2DecoderLayer {

    /// The replacements of this layer for one process of `group`.
    ///
    /// It also checks every projection and expert that ``shard(_:with:group:)`` splits in place,
    /// so a layer that cannot split throws before any layer changes.
    ///
    /// - Parameters:
    ///   - plan: The plan of the model.
    ///   - group: The group to split across.
    func makeShards(
        _ plan: Inkling2TensorParallelPlan, group: MLXDistributed.Group
    ) throws -> Inkling2LayerShards {
        var projections = [
            ("q_proj", attention.queryProj), ("k_proj", attention.keyProj),
            ("v_proj", attention.valueProj), ("r_proj", attention.relativeProj),
            ("gate_proj", attention.gateProj), ("sink_proj", attention.sinkProj),
        ]
        for (path, convolution) in [
            ("k_sconv", attention.keyConvolution), ("v_sconv", attention.valueConvolution),
        ] {
            if let dynamic = convolution as? Inkling2DynamicShortConvolution {
                projections.append(("\(path).weight_proj", dynamic.weightProj))
            }
        }
        for (path, projection) in projections {
            try TensorParallel.requireStockProjection(projection, at: "self_attn.\(path)")
        }
        // The gate and up rows split whole; only the down inputs cut across groups.
        try plan.requireWholeQuantizationGroups(
            expertGroupSize: mlp.experts.downProj.format.groupSize)
        let output = try TensorParallel.makeShardedToAll(
            attention.outputProj, at: "self_attn.o_proj", group: group)
        return Inkling2LayerShards(output: output)
    }

    /// Keeps the share of this process of every attention head and of every expert.
    ///
    /// - Parameters:
    ///   - plan: The plan of the model.
    ///   - shards: The replacements of ``makeShards(_:group:)``.
    ///   - group: The group to split across.
    func shard(
        _ plan: Inkling2TensorParallelPlan, with shards: Inkling2LayerShards,
        group: MLXDistributed.Group
    ) throws {
        try TensorParallel.shardAxes(
            of: attention, plan.attentionSplits(sliding: isSliding), group: group)
        let heads = plan.heads(sliding: isSliding)
        attention.adoptShardedHeads(heads: heads.heads, kvHeads: heads.kvHeads)
        try attention.update(
            modules: ModuleChildren(values: ["o_proj": .value(shards.output)]), verify: [])

        try TensorParallel.shardAxes(
            of: mlp.experts, Inkling2TensorParallelPlan.expertSplits, group: group)
        mlp.adoptShardedExperts(group: group)
    }
}
