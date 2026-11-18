// Copyright © 2026 Apple Inc.
//
// The prompt prefill of the Qwen 3.5 verify suites, of the text and of the
// vision-language model. Each suite fills its caches with the prompt through
// this one helper, then runs its verify over them, so no suite can leave a
// prompt unevaluated in its caches.

import MLX

/// The prompt prefill of the Qwen 3.5 verify suites.
enum Qwen35VerifyPrefill {
    /// Runs `prompt` through `model` over `caches`, and evaluates the output of the prompt.
    ///
    /// - Parameters:
    ///   - model: The model of the suite.
    ///   - prompt: The prompt tokens, shape `(1, length)`.
    ///   - caches: The new caches of the suite, which the prompt fills.
    ///   - forward: The prompt forward of the suite: the model, the prompt and the caches give
    ///     the output of the prompt.
    ///   - evaluated: The array of the output that the prefill evaluates.
    /// - Returns: The filled caches and the output of the prompt.
    static func prefilled<Model, Cache, Output>(
        _ model: Model,
        prompt: MLXArray,
        caches: Cache,
        forward: (Model, MLXArray, Cache) -> Output,
        evaluating evaluated: KeyPath<Output, MLXArray>
    ) -> (caches: Cache, output: Output) {
        let output = forward(model, prompt, caches)
        eval(output[keyPath: evaluated])
        return (caches, output)
    }
}
