// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The share of a Qwen 3.5 model that each process of a group keeps.
///
/// Each process keeps whole heads. The query projection of an attention layer holds the query
/// and the gate of each head side by side, so a split of its rows keeps both halves together.
struct Qwen35TensorParallelPlan: Equatable {

    /// The number of processes.
    let groupSize: Int

    /// The full-attention query heads of each process.
    let attentionHeads: Int

    /// The full-attention key and value heads of each process.
    let kvHeads: Int

    /// The gated delta key heads of each process.
    let linearKeyHeads: Int

    /// The gated delta value heads of each process.
    let linearValueHeads: Int

    /// The q, k and v parts of the fused gated delta input projection and of its conv, each of
    /// which splits by head on its own.
    let gatedDeltaInputSegments: Segments

    /// - Parameters:
    ///   - configuration: The configuration of the unsharded model.
    ///   - groupSize: The number of processes.
    /// - Throws: ``TensorParallelShardingError`` when a count does not split, or for a
    ///   mixture-of-experts model.
    init(configuration: Qwen35TextConfiguration, groupSize: Int) throws {
        guard configuration.numExperts == 0 else {
            throw TensorParallelShardingError.unsupported(feature: "mixture-of-experts layers")
        }
        self.groupSize = groupSize
        attentionHeads = try TensorParallel.share(
            of: configuration.attentionHeads, named: "attentionHeads", across: groupSize)
        kvHeads = try TensorParallel.share(
            of: configuration.kvHeads, named: "kvHeads", across: groupSize)
        linearKeyHeads = try TensorParallel.share(
            of: configuration.linearNumKeyHeads, named: "linearNumKeyHeads", across: groupSize)
        linearValueHeads = try TensorParallel.share(
            of: configuration.linearNumValueHeads, named: "linearNumValueHeads",
            across: groupSize)
        _ = try TensorParallel.share(
            of: configuration.intermediateSize, named: "intermediateSize", across: groupSize)

        let keyDim = configuration.linearNumKeyHeads * configuration.linearKeyHeadDim
        gatedDeltaInputSegments = .indices([keyDim, Qwen35GatedDeltaNet.queryKeyParts * keyDim])
    }
}

/// The modules that replace parts of one decoder layer, made before anything changes.
struct Qwen35LayerShards {

    /// The children of the attention or of the gated delta net to replace, by key: the output
    /// projection, and the conv of a gated delta net.
    let mixer: [String: Module]

    /// The down projection of the MLP.
    let down: Linear
}

extension Qwen35DecoderLayer {

    /// The replacements of this layer for one process of `group`.
    ///
    /// It also checks every projection that ``shard(_:with:group:)`` splits in place, so a layer
    /// that cannot split throws before any layer changes.
    ///
    /// - Parameters:
    ///   - plan: The plan of the model.
    ///   - group: The group to split across.
    func makeShards(
        _ plan: Qwen35TensorParallelPlan, group: MLXDistributed.Group
    ) throws -> Qwen35LayerShards {
        try splitProjections(plan).requireStockRowProjections()
        try requireWholeShares(plan)
        let down = try TensorParallel.makeShardedToAll(
            try denseMLP().downProj, at: "mlp.down_proj", group: group)

        if let attention = selfAttn {
            let output = try TensorParallel.makeShardedToAll(
                attention.oProj, at: "self_attn.o_proj", group: group)
            return Qwen35LayerShards(mixer: ["o_proj": output], down: down)
        }

        let gatedDelta = gatedDeltaNet
        let output = try TensorParallel.makeShardedToAll(
            gatedDelta.outProj, at: "linear_attn.out_proj", group: group)
        let conv = try TensorParallel.makeShardedDepthwiseConv(
            gatedDelta.conv1d, segments: plan.gatedDeltaInputSegments, group: group)
        return Qwen35LayerShards(mixer: ["out_proj": output, "conv1d": conv], down: down)
    }

    /// Throws unless each process keeps whole rows of every projection whose rows split, and
    /// whole quantization groups of every projection whose inputs split.
    ///
    /// - Parameter plan: The plan of the model.
    /// - Throws: ``TensorParallelShardingError/indivisible(quantity:count:groupSize:)`` naming
    ///   the first projection that does not split.
    func requireWholeShares(_ plan: Qwen35TensorParallelPlan) throws {
        try splitProjections(plan).requireWholeShares(across: plan.groupSize)
    }

    /// The projections of this layer that split, by the axis they split on.
    private func splitProjections(_ plan: Qwen35TensorParallelPlan) throws
        -> TensorParallel.SplitProjections
    {
        let mlp = try denseMLP()
        let rows: [(path: String, projection: Linear, segments: Segments)] = [
            ("mlp.gate_proj", mlp.gateProj, .count(1)), ("mlp.up_proj", mlp.upProj, .count(1)),
        ]
        let down = (path: "mlp.down_proj", projection: mlp.downProj)
        if let attention = selfAttn {
            return TensorParallel.SplitProjections(
                rows: rows + [
                    ("self_attn.q_proj", attention.qProj, .count(1)),
                    ("self_attn.k_proj", attention.kProj, .count(1)),
                    ("self_attn.v_proj", attention.vProj, .count(1)),
                ],
                inputs: [down, ("self_attn.o_proj", attention.oProj)])
        }
        let gatedDelta = gatedDeltaNet
        return TensorParallel.SplitProjections(
            rows: rows + [
                ("linear_attn.in_proj_qkv", gatedDelta.inProjQKV, plan.gatedDeltaInputSegments),
                ("linear_attn.in_proj_z", gatedDelta.inProjZ, .count(1)),
                ("linear_attn.in_proj_b", gatedDelta.inProjB, .count(1)),
                ("linear_attn.in_proj_a", gatedDelta.inProjA, .count(1)),
            ],
            inputs: [down, ("linear_attn.out_proj", gatedDelta.outProj)])
    }

    /// Keeps the share of this process of every projection, conv and per-head parameter.
    ///
    /// - Parameters:
    ///   - plan: The plan of the model.
    ///   - shards: The replacements of ``makeShards(_:group:)``.
    ///   - group: The group to split across.
    func shard(
        _ plan: Qwen35TensorParallelPlan, with shards: Qwen35LayerShards,
        group: MLXDistributed.Group
    ) throws {
        let mixer: Module
        if let attention = selfAttn {
            try TensorParallel.shardLeadingAxis(
                of: attention, ["q_proj": .count(1), "k_proj": .count(1), "v_proj": .count(1)],
                group: group)
            attention.adoptShardedHeads(from: plan)
            mixer = attention
        } else {
            let gatedDelta = gatedDeltaNet
            try TensorParallel.shardLeadingAxis(
                of: gatedDelta,
                [
                    "in_proj_qkv": plan.gatedDeltaInputSegments, "in_proj_z": .count(1),
                    "in_proj_b": .count(1), "in_proj_a": .count(1), "dt_bias": .count(1),
                    "A_log": .count(1),
                ],
                group: group)
            gatedDelta.adoptShardedHeads(from: plan)
            mixer = gatedDelta
        }
        mixer.update(modules: ModuleChildren(values: shards.mixer.mapValues { .value($0) }))

        let mlp = try denseMLP()
        try TensorParallel.shardLeadingAxis(
            of: mlp, ["gate_proj": .count(1), "up_proj": .count(1)], group: group)
        mlp.update(modules: ModuleChildren(values: ["down_proj": .value(shards.down)]))
    }

    /// The dense MLP of this layer.
    private func denseMLP() throws -> Qwen3NextMLP {
        guard let mlp = mlp as? Qwen3NextMLP else {
            throw TensorParallelShardingError.unsupported(feature: "mixture-of-experts layers")
        }
        return mlp
    }
}
