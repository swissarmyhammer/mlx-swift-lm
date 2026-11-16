// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Default-on rollback switch for the Qwen 3.5/3.6 four-projection GDN fusion.
package let qwen35FourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw != "0" && raw != "false" && raw != "off"
}()

/// A fused quantized projection and checkpoint-shaped views into its storage.
///
/// The views let a model keep its public/checkpoint module topology without
/// retaining a second physical copy of the quantized weights.
package struct FusedQuantizedLinearProjection {
    package let fused: QuantizedLinear
    package let sourceViews: [QuantizedLinear]
}

/// A fused projection could not replace its source modules atomically.
///
/// `rollbackError` is non-nil only when restoring the original source modules
/// also failed; callers can then surface both failures while retaining the
/// conservative unfused execution path.
package struct FusedQuantizedLinearPreparationError: Error, CustomStringConvertible {
    package let installationError: any Error
    package let rollbackError: (any Error)?

    package var description: String {
        if let rollbackError {
            return "unable to install fused projection views (\(installationError)); "
                + "restoring the original projections also failed (\(rollbackError))"
        }
        return "unable to install fused projection views (\(installationError)); "
            + "the original projections were restored"
    }
}

/// Lifecycle state for a physical projection derived from several registered
/// quantized linears.
///
/// The cache is intentionally not synchronized. Its only mutating operations
/// run during model loading or an explicit model update, both of which require
/// exclusive access. Inference only reads ``fused`` after the model has been
/// published, avoiding a lock in the token-generation hot path.
package final class FusedQuantizedLinearProjectionCache {
    private enum State {
        case unprepared
        case preparing
        case ready
        case ineligible
    }

    private var state = State.unprepared
    package private(set) var fused: QuantizedLinear?

    package init() {}

    package var isPrepared: Bool {
        state == .ready && fused != nil
    }

    /// Drop derived state after any source-module or source-parameter update.
    /// Invalidations caused by installing the cache's own storage-sharing views
    /// are ignored; the preparation transaction publishes `fused` only after
    /// every view has been installed successfully.
    package func invalidate() {
        guard state != .preparing else { return }
        fused = nil
        state = .unprepared
    }

    /// Build and publish the fused projection once for the current source
    /// topology. A failed or ineligible attempt is terminal until a subsequent
    /// source update calls ``invalidate()``.
    @discardableResult
    package func prepare(
        enabled: Bool,
        linears: [Linear],
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> Bool {
        guard enabled else { return false }

        switch state {
        case .ready:
            return fused != nil
        case .preparing, .ineligible:
            return false
        case .unprepared:
            break
        }

        state = .preparing
        guard let projection = fuseQuantizedLinearProjections(linears),
            projection.sourceViews.count == linears.count
        else {
            state = .ineligible
            return false
        }

        do {
            try installSourceModules(projection.sourceViews)
        } catch let installationError {
            let rollbackError: (any Error)?
            do {
                try installSourceModules(linears)
                rollbackError = nil
            } catch let error {
                rollbackError = error
            }
            fused = nil
            state = .ineligible
            throw FusedQuantizedLinearPreparationError(
                installationError: installationError,
                rollbackError: rollbackError)
        }

        fused = projection.fused
        state = .ready
        return true
    }
}

/// Coalesce compatible quantized linears along their output dimension.
///
/// This is intentionally stricter than merely casting to `QuantizedLinear`:
/// custom subclasses may add behavior, adapters, or state that cannot be
/// represented by one stock quantized matmul. Incompatible inputs return
/// `nil` and callers keep their original projections.
package func fuseQuantizedLinearProjections(
    _ linears: [Linear]
) -> FusedQuantizedLinearProjection? {
    guard linears.count > 1 else { return nil }

    let projections = linears.compactMap { $0 as? QuantizedLinear }
    guard projections.count == linears.count,
        zip(linears, projections).allSatisfy({ linear, projection in
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                && linear === projection
        }),
        let first = projections.first,
        first.bias == nil,
        first.globalScale == nil,
        first.weight.ndim == 2,
        first.weight.dim(0) == first.shape.0,
        first.scales.ndim == 2,
        first.scales.dim(0) == first.shape.0,
        first.biases == nil || first.biases?.shape == first.scales.shape
    else {
        return nil
    }

    let hasQuantizationBiases = first.biases != nil
    guard
        projections.allSatisfy({ projection in
            projection.bias == nil
                && projection.globalScale == nil
                && projection.bits == first.bits
                && projection.groupSize == first.groupSize
                && projection.mode == first.mode
                && projection.shape.1 == first.shape.1
                && projection.weight.ndim == 2
                && projection.weight.dim(0) == projection.shape.0
                && projection.weight.dim(1) == first.weight.dim(1)
                && projection.weight.dtype == first.weight.dtype
                && projection.scales.ndim == 2
                && projection.scales.dim(0) == projection.shape.0
                && projection.scales.dim(1) == first.scales.dim(1)
                && projection.scales.dtype == first.scales.dtype
                && (projection.biases != nil) == hasQuantizationBiases
                && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                && projection.biases?.dtype == first.biases?.dtype
        })
    else {
        return nil
    }

    let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
    let fusedScales = concatenated(projections.map(\.scales), axis: 0)
    let fusedBiases =
        hasQuantizationBiases
        ? concatenated(projections.compactMap(\.biases), axis: 0)
        : nil

    // Realize the concatenations once. The fused projection and the named
    // source views below then share this materialized backing storage.
    eval(fusedWeight, fusedScales)
    if let fusedBiases {
        eval(fusedBiases)
    }

    let fused = QuantizedLinear(
        weight: fusedWeight,
        bias: nil,
        scales: fusedScales,
        biases: fusedBiases,
        groupSize: first.groupSize,
        bits: first.bits,
        mode: first.mode)
    fused.freeze()

    var start = 0
    let sourceViews = projections.map { projection in
        let end = start + projection.shape.0
        defer { start = end }

        let rows = start ..< end
        let view = QuantizedLinear(
            weight: fusedWeight[rows],
            bias: nil,
            scales: fusedScales[rows],
            biases: fusedBiases.map { $0[rows] },
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode)
        view.freeze()
        return view
    }

    return FusedQuantizedLinearProjection(fused: fused, sourceViews: sourceViews)
}

/// Sibling quantized projections of one module that read the same input,
/// computed by one fused matmul once the module prepares them at load.
///
/// The module keeps each projection under its checkpoint key, as a view into
/// the fused storage, so checkpoints, adapters and parameter updates see the
/// separate projections, and the fusion costs no second copy of the weights.
/// An update that replaces one of the projections, or that writes one of their
/// parameters, turns the fusion off and drops every compiled trace that read
/// it, until the next preparation.
///
/// By default (``FusedRows/vectorCalls``) the fused matmul serves the calls of
/// fewer than ``fewestMatrixRows`` input rows: a decode step and a short
/// speculative verify. There MLX computes each output from its own weights,
/// scales and biases in an order the output width does not change (`qmv`,
/// `qmv_wide`, or ``MultiRowQuantizedMatvec``), thus each output of the fused
/// matmul has the bits of the separate projection. A longer call, a prefill,
/// keeps the separate matmuls: MLX gives it to its split-K matrix kernel, whose
/// split of the inner dimension depends on the output width, so a wider fused
/// matmul would sum in another order. A group of ``FusedRows/everyCall`` runs
/// the fused matmul for a prefill too.
package final class FusedQuantizedProjectionGroup {

    /// The calls whose outputs the fused matmul computes.
    package enum FusedRows {
        /// The calls of fewer than ``FusedQuantizedProjectionGroup/fewestMatrixRows``
        /// input rows. Each output has the bits of its separate projection.
        case vectorCalls
        /// Every call, a prefill too. A prefill may sum in another order than
        /// the separate projections.
        case everyCall
    }

    /// The last axis of a projection input or output, which holds the
    /// numbers of one row.
    private static let rowAxis = -1

    /// The fewest input rows MLX gives to its matrix-matrix kernels on any GPU
    /// (`get_qmv_batch_limit` of the Metal backend). Calls of fewer rows run
    /// the fused matmul.
    package static let fewestMatrixRows = 6

    /// A compiled trace that read the fused projection, held weakly: the
    /// trace belongs to a module that the group must not keep alive.
    private struct ReadingTrace {
        weak var value: (any CompiledTraceInvalidating)?
    }

    /// The checkpoint keys of the projections under their module, in the
    /// order of the fused rows.
    package let keys: [String]

    /// The calls whose outputs the fused matmul computes.
    package let fusedRows: FusedRows

    /// Whether the group fuses its projections. A disabled group prepares
    /// nothing, and computes every call with the separate matmuls, even after
    /// a preparation. A change drops every compiled trace that read the
    /// fusion, and keeps the preparation.
    package var isEnabled: Bool {
        didSet { isEnabledDidChange(from: oldValue) }
    }

    private let projection = FusedQuantizedLinearProjectionCache()

    // Trace setup runs under the lock of each trace, which an invalidation
    // must not take while it holds this one.
    private let readingTracesLock = NSLock()
    private var readingTraces: [ObjectIdentifier: ReadingTrace] = [:]

    /// - Parameters:
    ///   - keys: The checkpoint keys of the projections under their module,
    ///     in the order the module gives the projections.
    ///   - fusedRows: The calls whose outputs the fused matmul computes.
    ///   - isEnabled: Whether the group fuses its projections.
    package init(keys: [String], fusedRows: FusedRows = .vectorCalls, isEnabled: Bool = true) {
        self.keys = keys
        self.fusedRows = fusedRows
        self.isEnabled = isEnabled
    }

    /// Whether the fused projection computes the outputs.
    package var isPrepared: Bool { projection.isPrepared }

    /// Fuses `projections` and installs, in `owner`, a view of each one under
    /// its key. Projections that differ in quantization, carry a bias, or are
    /// not plain ``QuantizedLinear`` layers (an adapter, for example) keep
    /// their own matmuls, as all projections of a disabled group do.
    ///
    /// - Parameters:
    ///   - projections: The projections of ``keys``, in that order.
    ///   - owner: The module that holds the projections.
    /// - Returns: Whether the fused projection computes the outputs.
    @discardableResult
    package func prepare(_ projections: [Linear], in owner: Module) throws -> Bool {
        let keys = keys
        return try projection.prepare(enabled: isEnabled, linears: projections) { views in
            try owner.update(
                modules: ModuleChildren(
                    values: Dictionary(
                        uniqueKeysWithValues: zip(keys, views.map { .value($0) }))),
                verify: [])
        }
    }

    /// The outputs of `projections` for `x`, in order: the parts of the fused
    /// matmul once prepared, while enabled, for the calls of ``fusedRows``,
    /// else one matmul for each projection. Each matmul runs through
    /// ``MultiRowQuantizedMatvec/projection(_:by:)``.
    ///
    /// - Parameters:
    ///   - x: The input, shape `(batch, tokens, inputWidth)`.
    ///   - projections: The projections of ``keys``, in that order.
    /// - Returns: The output of each projection, shape `(batch, tokens, rows)`.
    package func callAsFunction(_ x: MLXArray, _ projections: [Linear]) -> [MLXArray] {
        guard isEnabled, let fused = projection.fused, fuses(x) else {
            return projections.map { MultiRowQuantizedMatvec.projection(x, by: $0) }
        }
        let rows = projections.dropLast().map(\.shape.0)
        let ends = rows.indices.map { rows[...$0].reduce(0, +) }
        return split(
            MultiRowQuantizedMatvec.projection(x, by: fused), indices: ends,
            axis: Self.rowAxis)
    }

    /// Whether the fused matmul computes the outputs for `x`, by
    /// ``fusedRows``.
    private func fuses(_ x: MLXArray) -> Bool {
        switch fusedRows {
        case .vectorCalls:
            x.size / x.dim(Self.rowAxis) < Self.fewestMatrixRows
        case .everyCall:
            true
        }
    }

    /// The fused projection as compile state of `trace`, which the group
    /// drops when the fusion turns off. It is not a registered child of its
    /// module, so a trace would otherwise read its arrays as tape constants,
    /// which MLX keeps alive after the trace is erased.
    ///
    /// - Parameter trace: The trace that reads the state.
    /// - Returns: The fused projection, or nothing before the preparation.
    package func traceState(of trace: any CompiledTraceInvalidating) -> [Module] {
        readingTracesLock.withLock {
            readingTraces[ObjectIdentifier(trace)] = ReadingTrace(value: trace)
        }
        return projection.fused.map { [$0] } ?? []
    }

    /// Runs `update`, a parameter update of the module, then turns the fusion
    /// off when the update reaches one of the projections, also when it
    /// throws after it wrote an earlier tensor.
    ///
    /// - Parameters:
    ///   - parameters: The parameters of the update, under the module.
    ///   - update: The update.
    /// - Returns: The result of `update`.
    package func invalidating<Result>(
        afterUpdating parameters: ModuleParameters, _ update: () throws -> Result
    ) rethrows -> Result {
        let keys = keys
        let reachesProjection = parameters.flattened().contains { name, _ in
            keys.contains { name.hasPrefix("\($0).") }
        }
        defer {
            if reachesProjection {
                invalidate()
            }
        }
        return try update()
    }

    /// Runs `update`, the replacement of the child module under `key`, then
    /// turns the fusion off when `key` is one of the projections, also when
    /// the module refuses the replacement.
    ///
    /// - Parameters:
    ///   - key: The key of the child module.
    ///   - update: The replacement.
    /// - Returns: The result of `update`.
    package func invalidating<Result>(
        afterReplacing key: String, _ update: () throws -> Result
    ) rethrows -> Result {
        defer {
            if keys.contains(key) {
                invalidate()
            }
        }
        return try update()
    }

    /// Drops every trace that read the fusion when ``isEnabled`` changed: a
    /// trace captured under the old setting would keep computing the outputs
    /// the old way.
    ///
    /// - Parameter oldValue: The setting before the change.
    private func isEnabledDidChange(from oldValue: Bool) {
        guard isEnabled != oldValue else { return }
        dropReadingTraces()
    }

    /// Turns the fusion off and drops every trace that read it. The views the
    /// preparation installs land here too, and drop the traces that ran before
    /// it.
    private func invalidate() {
        projection.invalidate()
        dropReadingTraces()
    }

    /// Drops every trace that read the fusion, and forgets it.
    private func dropReadingTraces() {
        let traces = readingTracesLock.withLock {
            let traces = readingTraces.values.compactMap(\.value)
            readingTraces.removeAll()
            return traces
        }
        for trace in traces {
            trace.invalidate()
        }
    }
}

/// The four input projections of a Qwen 3.5/3.6 gated delta net, which one
/// ``FusedQuantizedProjectionGroup`` computes for every call: their checkpoint
/// keys, their order among the fused rows, and the split of the outputs. The
/// text model and the vision-language model both read them here.
package enum Qwen35GatedDeltaInputProjection {

    /// The four projections, in the order of the fused rows. The raw value is
    /// the place of the output of each one among the outputs of the group.
    private enum Part: Int, CaseIterable {
        /// `in_proj_qkv`: the mixed query, key and value.
        case qkv
        /// `in_proj_z`: the gate of the output norm.
        case z
        /// `in_proj_b`: the write strength of each value head.
        case b
        /// `in_proj_a`: the decay of each value head.
        case a

        /// The checkpoint key of the projection.
        var key: String {
            switch self {
            case .qkv: "in_proj_qkv"
            case .z: "in_proj_z"
            case .b: "in_proj_b"
            case .a: "in_proj_a"
            }
        }
    }

    /// The checkpoint keys of the four projections under the gated delta
    /// net, in the order of the fused rows.
    package static let keys = Part.allCases.map(\.key)

    /// The group that fuses the four projections for every call, a prefill
    /// too, unless `MLX_QWEN_FOUR_GDN=0` turns the fusion off.
    ///
    /// - Returns: A new group, enabled by ``qwen35FourGDNEnabled``.
    package static func makeGroup() -> FusedQuantizedProjectionGroup {
        FusedQuantizedProjectionGroup(
            keys: keys, fusedRows: .everyCall, isEnabled: qwen35FourGDNEnabled)
    }

    /// The four projections in the order of ``keys``, as the group takes them.
    ///
    /// - Parameters:
    ///   - qkv: `in_proj_qkv`.
    ///   - z: `in_proj_z`.
    ///   - b: `in_proj_b`.
    ///   - a: `in_proj_a`.
    /// - Returns: The projections, in the order of the fused rows.
    package static func ordered(qkv: Linear, z: Linear, b: Linear, a: Linear) -> [Linear] {
        [qkv, z, b, a]
    }

    /// The four projections of `inputs` through `group`, with `z` split into
    /// heads.
    ///
    /// - Parameters:
    ///   - inputs: The layer input, shape `(batch, sequence, hiddenSize)`.
    ///   - group: The group of ``makeGroup()`` that the gated delta net holds.
    ///   - projections: The projections of ``ordered(qkv:z:b:a:)``.
    ///   - batch: The batch size of `inputs`.
    ///   - sequence: The tokens of `inputs`.
    ///   - heads: The value heads that `z` splits into.
    ///   - headWidth: The width of each value head.
    /// - Returns: The mixed qkv, `z` of shape `(batch, sequence, heads,
    ///   headWidth)`, `b` and `a`.
    package static func project(
        _ inputs: MLXArray, by group: FusedQuantizedProjectionGroup, _ projections: [Linear],
        batch: Int, sequence: Int, heads: Int, headWidth: Int
    ) -> (qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray) {
        let projected = group(inputs, projections)
        return (
            projected[Part.qkv.rawValue],
            projected[Part.z.rawValue].reshaped(batch, sequence, heads, headWidth),
            projected[Part.b.rawValue],
            projected[Part.a.rawValue]
        )
    }
}
