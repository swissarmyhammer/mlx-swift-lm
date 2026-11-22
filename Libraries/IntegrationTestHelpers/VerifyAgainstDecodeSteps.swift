// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// The largest difference between the logits of a speculative verify pass over `verify` and
/// those of decode steps over the same tokens, one at a time, both after `prefix`.
///
/// Each side gets a new cache, takes the prefill `prefix[0]` in one call and each later chunk
/// of `prefix` one call at a time, so both reach `verify` through the same writes.
///
/// - Parameters:
///   - model: the model.
///   - prefix: the chunks before the verify, `[1, L]` token arrays: the prefill, then the
///     decode steps.
///   - verify: the tokens of the verify pass, `[1, L]`.
/// - Returns: the largest absolute difference of the two sides' logits; 0 when the verify
///   computes the logits of the decode steps bit for bit.
/// - Throws: the error of a new cache.
public func verifyDifferenceFromDecodeSteps(
    _ model: any LanguageModel, prefix: [MLXArray], verify: MLXArray
) throws -> Float {
    /// A new cache after `prefix`.
    func advanced() throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        for chunk in prefix {
            eval(model(chunk, cache: cache))
        }
        return cache
    }
    let stepped = try advanced()
    let steps = (0 ..< verify.dim(1)).map { index in
        let step = model(verify[0..., index ..< (index + 1)], cache: stepped)
        eval(step)
        return step
    }
    let verified = model(verify, cache: try advanced())
    return maxAbsoluteDifference(verified, concatenated(steps, axis: 1))
}

/// The largest absolute difference of `a` and `b`, in float32.
///
/// - Parameters:
///   - a: one array.
///   - b: the other array, of a shape that broadcasts with `a`.
/// - Returns: the largest `|a - b|`; 0 when the two are equal.
public func maxAbsoluteDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}
