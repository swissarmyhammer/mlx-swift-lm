// Copyright © 2026 Apple Inc.
//
// The ops fallback of the gated delta rule on a process that runs in a
// distributed group, where the worker uses exactly one GPU stream: under
// `MLX_METAL_FAST_SYNCH` work that crosses two GPU streams hangs (exo#2391).
// mlx core's `custom_function` calls `stop_gradient` on the calling thread's
// C++ default stream, and that stream is made per thread on first use, so a
// `CustomFunction` makes a second GPU stream on every thread that calls it.

import Cmlx
import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct GatedDeltaGPUStreamTests {

    /// The batch of each fixture.
    private static let batch = 1

    /// The steps of each fixture: three recompute chunks, the last one short.
    private static let steps = 40

    /// The key heads of each fixture.
    private static let keyHeads = 2

    /// The key head dim of each fixture: a multiple of the simd width, so the
    /// fused kernel would take it, but every call here turns the kernel off.
    private static let keyHeadDim = 32

    /// The value heads of each fixture.
    private static let valueHeads = 4

    /// The value head dim of each fixture.
    private static let valueHeadDim = 16

    /// The seed of the fixture's random inputs.
    private static let seed: UInt64 = 42

    /// The scale of the fixture's `aLog`, which keeps the decay near one.
    private static let aLogScale: Float = 0.1

    /// The inputs of one call to `gatedDeltaUpdate`.
    private struct Inputs {
        let q, k, v, a, b, aLog, dtBias: MLXArray
    }

    /// Random bf16 inputs from a fixed key. The key is explicit, so the
    /// global random state that parallel tests share stays untouched.
    private static func makeInputs() -> Inputs {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 7)
        func normal(_ shape: [Int], _ index: Int) -> MLXArray {
            MLXRandom.normal(shape, key: keys[index]).asType(.bfloat16)
        }
        return Inputs(
            q: normal([batch, steps, keyHeads, keyHeadDim], 0),
            k: normal([batch, steps, keyHeads, keyHeadDim], 1),
            v: normal([batch, steps, valueHeads, valueHeadDim], 2),
            a: normal([batch, steps, valueHeads], 3),
            b: normal([batch, steps, valueHeads], 4),
            aLog: normal([valueHeads], 5) * MLXArray(aLogScale),
            dtBias: normal([valueHeads], 6))
    }

    /// The registry index of every GPU stream MLX made in this process.
    ///
    /// MLX's stream registry (`mlx_get_streams`) holds every stream any
    /// thread made, and no stream leaves it. Reading it makes no stream.
    private static func gpuStreams() -> Set<Int32> {
        var streams = mlx_vector_stream_new()
        defer { mlx_vector_stream_free(streams) }
        mlx_get_streams(&streams)
        let indices = (0 ..< mlx_vector_stream_size(streams)).compactMap { position in
            var stream = mlx_stream_new()
            defer { mlx_stream_free(stream) }
            mlx_vector_stream_get(&stream, streams, position)
            return isGPU(stream) ? index(of: stream) : nil
        }
        return Set(indices)
    }

    /// Whether `stream` runs on the GPU.
    private static func isGPU(_ stream: mlx_stream) -> Bool {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        mlx_stream_get_device(&device, stream)
        var type = MLX_CPU
        mlx_device_get_type(&type, device)
        return type == MLX_GPU
    }

    /// The registry index of `stream`.
    private static func index(of stream: mlx_stream) -> Int32 {
        var index: Int32 = 0
        mlx_stream_get_index(&index, stream)
        return index
    }

    /// The GPU streams that `forward` adds, run on a new thread, which has
    /// no C++ default stream yet.
    ///
    /// The caller evaluates every input of `forward` first, so the new thread
    /// runs nothing but the forward: any work before it could make the
    /// thread's default stream outside the measured span and hide the one
    /// the forward makes. The forward's outputs are not evaluated yet when
    /// `eval` runs, so `eval` takes their stream and never the thread's
    /// default stream.
    ///
    /// - Parameter forward: The forward to measure, which returns its
    ///   outputs unevaluated.
    /// - Returns: The registry index of each GPU stream the forward made.
    private static func gpuStreamsAdded(by forward: @escaping () -> [MLXArray]) async -> Set<Int32>
    {
        let box = SendableBox(forward)
        return await withCheckedContinuation { continuation in
            let thread = Thread {
                let forward = box.consume()
                let before = gpuStreams()
                eval(forward())
                continuation.resume(returning: gpuStreams().subtracting(before))
            }
            thread.start()
        }
    }

    /// The inputs of the fixture, evaluated.
    private static func makeEvaluatedInputs() -> Inputs {
        let inputs = makeInputs()
        eval(inputs.q, inputs.k, inputs.v, inputs.a, inputs.b, inputs.aLog, inputs.dtBias)
        return inputs
    }

    /// The gradients of a loss over the ops fallback's output and state, with
    /// respect to q, k, v, a and b.
    ///
    /// - Parameters:
    ///   - inputs: The inputs of the call.
    ///   - recomputeChunks: Whether the backward recomputes each chunk.
    /// - Returns: The gradients with respect to q, k, v, a and b.
    private static func opsGradients(of inputs: Inputs, recomputeChunks: Bool) -> [MLXArray] {
        let loss: ([MLXArray]) -> [MLXArray] = { arrays in
            let (y, state) = gatedDeltaUpdate(
                q: arrays[0], k: arrays[1], v: arrays[2], a: arrays[3], b: arrays[4],
                aLog: inputs.aLog, dtBias: inputs.dtBias, useKernel: false,
                recomputeChunks: recomputeChunks)
            return [y.asType(.float32).square().sum() + state.sum()]
        }
        let (_, gradients) = valueAndGrad(loss, argumentNumbers: [0, 1, 2, 3, 4])(
            [inputs.q, inputs.k, inputs.v, inputs.a, inputs.b])
        eval(gradients)
        return gradients
    }

    /// The largest difference between `lhs` and `rhs`, in float32.
    private static func largestDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
    }

    @Test("a forward through the ops fallback without chunk recompute makes no GPU stream")
    func opsForwardMakesNoGPUStream() async {
        let inputs = Self.makeEvaluatedInputs()
        let added = await Self.gpuStreamsAdded {
            let (y, state) = gatedDeltaUpdate(
                q: inputs.q, k: inputs.k, v: inputs.v, a: inputs.a, b: inputs.b,
                aLog: inputs.aLog, dtBias: inputs.dtBias, useKernel: false,
                recomputeChunks: false)
            return [y, state]
        }
        #expect(added.isEmpty)
    }

    @Test("the ops fallback without chunk recompute has the gradients of the recomputing one")
    func opsGradientsWithoutRecomputeMatch() {
        let inputs = Self.makeEvaluatedInputs()
        let recomputed = Self.opsGradients(of: inputs, recomputeChunks: true)
        let plain = Self.opsGradients(of: inputs, recomputeChunks: false)
        let differences = zip(recomputed, plain).map { Self.largestDifference($0, $1) }
        #expect(differences == Array(repeating: 0, count: recomputed.count))
    }

    @Test("a gated delta layer that adopts its sharded heads makes no GPU stream in its forward")
    func shardedLayerForwardMakesNoGPUStream() async throws {
        // The fixture's key head dim of 16 is not a multiple of the simd width,
        // so the layer's inference forward takes the ops fallback. A plan of one
        // process keeps the layer's weights whole beside its adopted heads.
        let configuration = try Qwen35TensorParallelPlanTests.makeConfiguration()
        let layer = Qwen35GatedDeltaNet(configuration)
        layer.adoptShardedHeads(
            from: try Qwen35TensorParallelPlan(configuration: configuration, groupSize: 1))
        let x = MLXRandom.normal(
            [Self.batch, Self.steps, configuration.hiddenSize], key: MLXRandom.key(Self.seed))
        let convState = layer.zeroStates(batch: Self.batch, dtype: x.dtype).conv
        eval(layer, x, convState)
        let added = await Self.gpuStreamsAdded {
            let out = layer.forward(x, convState: convState, recState: nil, mask: nil)
            return [out.output, out.recurrentState]
        }
        #expect(added.isEmpty)
    }
}
