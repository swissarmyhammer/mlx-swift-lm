// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN

/// Records the prefill step its `prepare` resolves, the way a model resolves it: with its own
/// default step when it has one, and with the generic default otherwise. Its logits over
/// ``GuidedByteTokenizer`` prefer EOS, so the grammar decides the output.
public final class PrefillStepRecordingModel: Module, LanguageModel {

    /// The logit of the EOS token, high enough to win every decision the grammar leaves open.
    private static let eosLogit: Float = 20

    /// The step the model prefills with when the caller sets none, or `nil` for a model without
    /// a default of its own.
    private let ownDefaultStepSize: Int?

    /// The step the last `prepare` resolved, or `nil` before the first `prepare`.
    public private(set) var resolvedStepSize: Int?

    /// Makes a model whose own default step is `ownDefaultStepSize`.
    ///
    /// - Parameter ownDefaultStepSize: The model's own default step, or `nil` for none.
    public init(ownDefaultStepSize: Int?) {
        self.ownDefaultStepSize = ownDefaultStepSize
        super.init()
    }

    /// Records the step `prefill` resolves to, and leaves every token of `input` to the
    /// generation.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        resolvedStepSize =
            ownDefaultStepSize.map { prefill.resolvedStepSize(defaultStepSize: $0) }
            ?? prefill.resolvedStepSize()
        return .tokens(input.text)
    }

    /// No cache: the model keeps no state between calls.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    /// One EOS-preferring logit row for each position of `input`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.size
        var row = [Float](repeating: 0, count: GuidedByteTokenizer.vocabularySize)
        row[GuidedByteTokenizer.eosTokenID] = Self.eosLogit
        let rows = Array(repeating: row, count: positions).flatMap { $0 }
        return LMOutput(
            logits: MLXArray(rows).reshaped(1, positions, GuidedByteTokenizer.vocabularySize))
    }
}
