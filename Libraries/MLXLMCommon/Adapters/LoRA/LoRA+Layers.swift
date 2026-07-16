// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Parameter key under which the low-rank `A` matrix of a LoRA adapter layer
/// is registered and serialized.
private let loraAKey = "lora_a"

/// Parameter key under which the low-rank `B` matrix of a LoRA adapter layer
/// is registered and serialized.
private let loraBKey = "lora_b"

/// Common declarations of the LoRA `Linear` adapter layers (``LoRALinear``
/// and ``QLoRALinear``).
///
/// The two adapter classes differ only in the base layer they extend
/// (`Linear` vs `QuantizedLinear`), so the shared adapter logic --
/// construction, initialization, freezing, evaluation, and weight fusing --
/// lives once in the protocol extension. Swift cannot declare stored
/// properties in a protocol extension and the two adapter classes extend
/// different superclasses, so each class supplies the storage for these
/// requirements itself.
protocol LoRAAdapterLayer: LoRALayer {

    /// The base `Linear` subclass the adapter wraps.
    associatedtype Base: Linear

    /// Scale applied to the low-rank update.
    var scale: Float { get }

    /// The low-rank `A` matrix, registered as the `lora_a` parameter.
    var loraA: MLXArray { get set }

    /// The low-rank `B` matrix, registered as the `lora_b` parameter.
    var loraB: MLXArray { get set }

    /// The `DType` inputs are converted to before the base layer evaluates
    /// them -- the dtype of the base layer's weight representation.
    var inputDType: DType { get }

    /// Create an adapter over `linear` -- see the concrete initializers of
    /// ``LoRALinear`` and ``QLoRALinear`` for the parameter documentation.
    init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int, bias: Bool, scale: Float,
        linear: Base)
}

extension LoRAAdapterLayer {

    /// The parameter keys to freeze: `keys` when given, otherwise all of the
    /// layer's local parameters, always omitting the LoRA adapter keys so the
    /// adapter stays trainable.
    ///
    /// This is the single shared implementation behind the
    /// `freeze(recursive:keys:strict:)` overrides of ``LoRALinear`` and
    /// ``QLoRALinear`` -- a protocol extension cannot override a class method
    /// or call `super`, so each class keeps a minimal override that passes
    /// this result to `super.freeze`.
    ///
    /// - Parameter keys: explicit keys to freeze, or `nil` to freeze all
    ///   local parameters
    /// - Returns: the realized keys with `lora_a` and `lora_b` removed
    func frozenParameterKeys(from keys: [String]?) -> [String] {
        (keys ?? self.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 })
            .filter {
                $0 != loraAKey && $0 != loraBKey
            }
    }

    /// Shared implementation of the `from(linear:rank:scale:)` factories:
    /// reads the dimensions from `linear` and constructs an adapter of type
    /// `Self` over it.
    ///
    /// - Parameters:
    ///   - linear: the base layer to adapt
    ///   - rank: rank of the low-rank update matrices
    ///   - scale: scale applied to the low-rank update
    /// - Returns: the adapter layer wrapping `linear`
    static func adapting(linear: Base, rank: Int, scale: Float) -> LoRALayer {
        let (outputDimensions, inputDimensions) = linear.shape
        return Self(
            inputDimensions, outputDimensions, rank: rank, bias: false, scale: scale,
            linear: linear)
    }

    /// Shared second phase of the concrete initializers, called after the
    /// base layer is initialized: gives the low-rank `A` matrix small random
    /// values, the `B` matrix zeros (so the adapter initially contributes
    /// nothing), and freezes every parameter except the LoRA parameters.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input features
    ///   - outputDimensions: number of output features
    ///   - rank: rank of the low-rank update matrices
    func initializeLoRA(inputDimensions: Int, outputDimensions: Int, rank: Int) {
        let loraScale = 1 / sqrt(Float(inputDimensions))
        loraA = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        loraB = MLXArray.zeros([rank, outputDimensions])

        freeze()
    }

    /// Shared implementation of `callAsFunction(_:)`: the base layer output
    /// for `x` plus, when ``LoRALayer/loraEnabled`` is `true`, the scaled
    /// low-rank LoRA update `scale * (x @ loraA @ loraB)`.
    ///
    /// - Parameters:
    ///   - x: the input array
    ///   - base: the base layer evaluation, i.e. `super.callAsFunction`
    /// - Returns: the adapted layer output
    func adapted(_ x: MLXArray, base: (MLXArray) -> MLXArray) -> MLXArray {
        let y = base(x.asType(inputDType))
        if !loraEnabled { return y }
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }

    /// `weight` with the scaled low-rank update fused in, computed in
    /// `weight`'s dtype. Shared implementation behind the `fused()` methods.
    ///
    /// - Parameter weight: the (dequantized) base layer weight
    /// - Returns: the fused weight `weight + scale * (loraA @ loraB).T`
    func fusedWeight(base weight: MLXArray) -> MLXArray {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return weight + matmul(loraB, loraA)
    }
}

/// Implementation of LoRA `Linear` replacement layer.
///
/// This layer implements the LoRA capabilities for `Linear` layers, specifically:
///
/// - converting `Linear` or `QuantizedLinear` layers to ``LoRALinear`` / ``QLoRALinear``
/// - converting ``LoRALinear`` back to `Linear` or `QuantizedLinear` via ``LoRALinear/fused()``
/// - implementing the LoRA evaluation
///
/// ``QLoRALinear`` is the equivalent class for `QuantizedLinear`.
///
/// This is not typically used directly -- `LoRATrain.convert(model:layers:)` is used to
/// add the adapter layers to a given model.
///
/// ### See Also
/// - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
/// - [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
/// - ``QLoRALinear``
public class LoRALinear: Linear, LoRAAdapterLayer {

    // These are `public` (rather than the private constants above) because
    // Swift only allows public declarations in the default argument values of
    // the public initializers and `from(linear:rank:scale:)` factories.

    /// Default rank of the low-rank update matrices.
    public static let defaultRank = 8

    /// Default scale applied to the low-rank update.
    public static let defaultScale: Float = 20.0

    /// Scale applied to the low-rank update.
    let scale: Float

    /// When `false` the low-rank update is skipped and the layer evaluates as
    /// the plain base `Linear` layer -- see ``LoRALayer/loraEnabled``.
    public var loraEnabled: Bool = true

    /// The low-rank `A` matrix.
    @ParameterInfo(key: loraAKey) var loraA: MLXArray

    /// The low-rank `B` matrix.
    @ParameterInfo(key: loraBKey) var loraB: MLXArray

    /// The dtype of the base `weight` -- inputs are converted to this dtype
    /// before the base layer evaluates them.
    var inputDType: DType { weight.dtype }

    /// Create a ``LoRALinear`` layer that adapts `linear`.
    ///
    /// The low-rank `A` matrix starts with small random values, the `B`
    /// matrix starts at zero (so the adapter initially contributes nothing),
    /// and every parameter except the LoRA parameters is frozen.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input features
    ///   - outputDimensions: number of output features
    ///   - rank: rank of the low-rank update matrices
    ///   - bias: ignored -- the bias is taken from `linear`
    ///   - scale: scale applied to the low-rank update
    ///   - linear: the `Linear` layer to adapt
    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = LoRALinear.defaultRank,
        bias: Bool = false, scale: Float = LoRALinear.defaultScale, linear: Linear
    ) {
        self.scale = scale

        super.init(weight: linear.weight, bias: linear.bias)

        initializeLoRA(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions, rank: rank)
    }

    /// Freeze all parameters except the lora parameters.
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        try super.freeze(
            recursive: recursive, keys: frozenParameterKeys(from: keys), strict: strict)
    }

    /// Convert a `Linear` or `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via `LoRATrain.convert(model:layers:)`.
    ///
    /// ### See Also
    /// - ``QLoRALinear/from(linear:rank:scale:)``
    public static func from(
        linear: Linear, rank: Int = LoRALinear.defaultRank, scale: Float = LoRALinear.defaultScale
    ) -> LoRALayer {
        if let linear = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: linear, rank: rank, scale: scale)
        }
        return adapting(linear: linear, rank: rank, scale: scale)
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// ### See Also
    /// - ``QLoRALinear/fused()``
    public func fused() -> Module {
        Linear(weight: fusedWeight(base: weight), bias: bias)
    }

    /// Compute the base `Linear` output for `x` plus, when ``loraEnabled`` is
    /// `true`, the scaled low-rank LoRA update `scale * (x @ loraA @ loraB)`.
    ///
    /// - Parameter x: the input array
    /// - Returns: the adapted layer output
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        adapted(x) { super.callAsFunction($0) }
    }
}

/// Implementation of LoRA `QuantizedLinear` replacement layer.
///
/// See ``LoRALinear`` (equivalent class for `Linear` layers) for more information.
public class QLoRALinear: QuantizedLinear, LoRAAdapterLayer {

    /// Scale applied to the low-rank update.
    let scale: Float

    /// When `false` the low-rank update is skipped and the layer evaluates as
    /// the plain base `QuantizedLinear` layer -- see ``LoRALayer/loraEnabled``.
    public var loraEnabled: Bool = true

    /// The low-rank `A` matrix.
    @ParameterInfo(key: loraAKey) var loraA: MLXArray

    /// The low-rank `B` matrix.
    @ParameterInfo(key: loraBKey) var loraB: MLXArray

    /// The dtype of the quantization `scales` (which the dequantized weight
    /// carries) -- inputs are converted to this dtype before the base layer
    /// evaluates them.
    var inputDType: DType { scales.dtype }

    /// Create a ``QLoRALinear`` layer that adapts `linear`.
    ///
    /// The low-rank `A` matrix starts with small random values, the `B`
    /// matrix starts at zero (so the adapter initially contributes nothing),
    /// and every parameter except the LoRA parameters is frozen.
    ///
    /// - Parameters:
    ///   - inputDimensions: number of input features
    ///   - outputDimensions: number of output features
    ///   - rank: rank of the low-rank update matrices
    ///   - bias: ignored -- the bias is taken from `linear`
    ///   - scale: scale applied to the low-rank update
    ///   - linear: the `QuantizedLinear` layer to adapt
    required public init(
        _ inputDimensions: Int, _ outputDimensions: Int, rank: Int = LoRALinear.defaultRank,
        bias: Bool = false, scale: Float = LoRALinear.defaultScale, linear: QuantizedLinear
    ) {
        self.scale = scale

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        initializeLoRA(
            inputDimensions: inputDimensions, outputDimensions: outputDimensions, rank: rank)
    }

    /// Freeze all parameters except the lora parameters.
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        try super.freeze(
            recursive: recursive, keys: frozenParameterKeys(from: keys), strict: strict)
    }

    /// Convert a `QuantizedLinear` layer into a new `Linear` layer
    /// that implements the `LoRA` adapter.
    ///
    /// This is typically called via `LoRATrain.convert(model:layers:)`.
    ///
    /// ### See Also
    /// - ``LoRALinear/from(linear:rank:scale:)``
    public static func from(
        linear: QuantizedLinear, rank: Int = LoRALinear.defaultRank,
        scale: Float = LoRALinear.defaultScale
    )
        -> LoRALayer
    {
        adapting(linear: linear, rank: rank, scale: scale)
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// ### See Also
    /// - ``LoRALinear/fused()``
    public func fused() -> Module {
        QuantizedLinear(
            weight: fusedWeight(base: dequantizedWeight), bias: bias, groupSize: groupSize,
            bits: bits)
    }

    /// Compute the base `QuantizedLinear` output for `x` plus, when
    /// ``loraEnabled`` is `true`, the scaled low-rank LoRA update
    /// `scale * (x @ loraA @ loraB)`.
    ///
    /// - Parameter x: the input array
    /// - Returns: the adapted layer output
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        adapted(x) { super.callAsFunction($0) }
    }
}
