import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

/// Bound for float32 values compared at full precision: on machines whose float32 matmuls run
/// in IEEE float32, and for values every run reaches through matmuls of the same shapes.
let ieeeFloat32Tolerance = 1e-5

/// Bound for float32 values that the same inputs reach through matmuls of different shapes on
/// machines whose float32 matmuls run as TF32; see ``MatmulPrecision``.
let tf32ReshapedMatmulTolerance = 2e-3

/// `count` token ids below `vocabularySize`, shape `[count]`, drawn from `seed`: the same
/// tokens for the same seed on every run.
func seededTokens(_ count: Int, vocabularySize: Int, seed: UInt64) -> MLXArray {
    MLXRandom.randInt(Int32(0) ..< Int32(vocabularySize), [count], key: MLXRandom.key(seed))
}

/// The hidden rows `target` emits for `tokens` when the MTP iterator asks it to, evaluated.
///
/// - Parameters:
///   - target: The target model.
///   - tokens: The tokens, shape `[1, length]`.
///   - cache: The cache of the target, or `nil` for a forward without one.
/// - Throws: When the output state of the target holds no hidden rows.
func targetHiddenStates(
    _ target: any LanguageModel, tokens: MLXArray, cache: [KVCache]? = nil
) throws -> MLXArray {
    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    let output = target(LMInput.Text(tokens: tokens), cache: cache, state: state)
    let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
    eval(hidden)
    return hidden
}
