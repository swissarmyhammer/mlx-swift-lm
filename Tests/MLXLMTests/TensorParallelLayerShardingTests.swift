// Copyright © 2026 Apple Inc.

// The per-layer loop of tensor parallel sharding: every layer's shards are made before any
// layer changes, and the hooks of a sharded load run around each layer's split.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("TensorParallel.shardLayers")
struct TensorParallelLayerShardingTests {

    /// The failure of a layer that cannot split.
    private enum ExpectedFailure: Error { case indivisible }

    /// The number of layers of each loop.
    private static let layerCount = 3

    /// Layers with no parameters: the loop reads only their order.
    private static func makeLayers() -> [Module] {
        (0 ..< layerCount).map { _ in Module() }
    }

    /// The events of one loop, in order.
    private final class Events {
        var log: [String] = []
    }

    @Test("a layer that cannot split stops the loop before any layer splits")
    func validatesEveryLayerFirst() {
        let layers = Self.makeLayers()
        let events = Events()
        #expect(throws: ExpectedFailure.self) {
            try TensorParallel.shardLayers(
                layers, hooks: .none,
                makeShards: { layer -> Int in
                    guard layer !== layers.last else { throw ExpectedFailure.indivisible }
                    return 0
                },
                shard: { _, _ in events.log.append("shard") })
        }
        #expect(events.log.isEmpty)
    }

    @Test("the hooks run around the split of each layer, in layer order")
    func runsHooksAroundEachLayer() throws {
        let layers = Self.makeLayers()
        let events = Events()
        let hooks = TensorParallelLayerHooks(
            willShard: { index, _ in events.log.append("will \(index)") },
            didShard: { index, _ in events.log.append("did \(index)") })
        try TensorParallel.shardLayers(
            layers, hooks: hooks,
            makeShards: { layer in layers.firstIndex { $0 === layer } ?? -1 },
            shard: { _, index in events.log.append("shard \(index)") })
        #expect(
            events.log == [
                "will 0", "shard 0", "did 0", "will 1", "shard 1", "did 1", "will 2", "shard 2",
                "did 2",
            ])
    }
}
