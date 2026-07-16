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
/// Declaring the shared properties once here lets the shared freezing logic
/// live in a single protocol extension. Swift cannot declare stored properties
/// in a protocol extension, and the two adapter classes extend different
/// superclasses (`Linear` and `QuantizedLinear`), so each class supplies the
/// storage for these requirements itself.
protocol LoRAAdapterLayer: LoRALayer {

    /// Scale applied to the low-rank update.
    var scale: Float { get }

    /// The low-rank `A` matrix, registered as the `lora_a` parameter.
    var loraA: MLXArray { get }

    /// The low-rank `B` matrix, registered as the `lora_b` parameter.
    var loraB: MLXArray { get }
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
        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(weight: linear.weight, bias: linear.bias)

        freeze()
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
        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Convert back into a fused `Linear` layer.
    ///
    /// ### See Also
    /// - ``QLoRALinear/fused()``
    public func fused() -> Module {
        let dtype = weight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return Linear(weight: weight + matmul(loraB, loraA), bias: bias)
    }

    /// Compute the base `Linear` output for `x` plus, when ``loraEnabled`` is
    /// `true`, the scaled low-rank LoRA update `scale * (x @ loraA @ loraB)`.
    ///
    /// - Parameter x: the input array
    /// - Returns: the adapted layer output
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        if !loraEnabled { return y }
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
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

        // Scale for low-rank update
        self.scale = scale

        // Low rank lora weights
        let loraScale = 1 / sqrt(Float(inputDimensions))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: -loraScale, high: loraScale, [inputDimensions, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)

        // start frozen except for the lora keys
        freeze()
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
        let (outputDimensions, inputDimensions) = linear.shape
        return QLoRALinear(
            inputDimensions, outputDimensions, rank: rank, scale: scale, linear: linear)
    }

    /// Convert back into a fused `QuantizedLinear` layer.
    ///
    /// ### See Also
    /// - ``LoRALinear/fused()``
    public func fused() -> Module {
        let weight = dequantizedWeight
        let dtype = dequantizedWeight.dtype
        let loraB = (scale * loraB.T).asType(dtype)
        let loraA = loraA.T.asType(dtype)
        return QuantizedLinear(
            weight: weight + matmul(loraB, loraA),
            bias: bias,
            groupSize: groupSize,
            bits: bits
        )
    }

    /// Compute the base `QuantizedLinear` output for `x` plus, when
    /// ``loraEnabled`` is `true`, the scaled low-rank LoRA update
    /// `scale * (x @ loraA @ loraB)`.
    ///
    /// - Parameter x: the input array
    /// - Returns: the adapted layer output
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        if !loraEnabled { return y }
        let z = matmul(matmul(x, self.loraA), self.loraB)
        return y + scale * z
    }
}
