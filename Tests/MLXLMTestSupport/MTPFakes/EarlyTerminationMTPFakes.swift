// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLMCommon
import MLXNN

/// Attention-only target whose logits accept the next token already present
/// in a verification block. The last position always predicts ordinary token
/// 4, which is the prepare-time and all-accepted bonus.
public final class EarlyTerminationMTPTarget: Module, LanguageModel, KVCacheDimensionProvider {
    /// The number of tokens of the fake vocabulary.
    public static let vocabularySize = 16

    public var kvHeads: [Int] { [1] }

    public func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        output(for: inputs, cache: cache, emitMTPState: false).logits
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        output(for: input.tokens, cache: cache, emitMTPState: state?[mtpEmitFlagKey] == true)
    }

    private func output(
        for tokens: MLXArray,
        cache: [KVCache]?,
        emitMTPState: Bool
    ) -> LMOutput {
        let positions = tokens.dim(-1)
        let newKV = MLXArray.zeros([1, 1, positions, 1])
        let sharedKV = cache?.first?.update(keys: newKV, values: newKV) ?? (newKV, newKV)

        let tokenValues = tokens.asArray(Int.self)
        var values = [Float](
            repeating: SyntheticMTPValues.rejectedLogit, count: positions * Self.vocabularySize)
        for position in 0 ..< positions {
            let nextToken =
                position + 1 < tokenValues.count
                ? tokenValues[position + 1] : SyntheticMTPValues.ordinaryToken
            values[position * Self.vocabularySize + nextToken] = SyntheticMTPValues.selectedLogit
        }
        let logits = MLXArray(values, [1, positions, Self.vocabularySize])

        guard emitMTPState else {
            return LMOutput(logits: logits)
        }

        var outputState = LMOutput.State()
        outputState[mtpLastHiddenStatesKey] = MLXArray.zeros([
            1, positions, SyntheticMTPValues.rowWidth,
        ])
        outputState[mtpSharedKVStatesKey] = ["full_attention": sharedKV]
        outputState[mtpSharedKVSourceIndicesKey] = ["full_attention": 0]
        outputState[mtpSharedKVOffsetsKey] = [
            "full_attention": cache?.first?.offset ?? positions
        ]
        return LMOutput(logits: logits, state: outputState)
    }
}

/// The first four-token prompt proposes EOS for every available draft position;
/// later full-transcript prompts propose ordinary token 4. Query offset makes
/// this behavior per-stream and deterministic without mutable model state.
public final class EarlyTerminationMTPDrafter: Module, ResumableMTPDrafterModel {
    /// The tokens of each round: three drafted tokens and the bonus token.
    public static let blockSize = 4

    public var maximumBlockSize: Int? { Self.blockSize }
    public var requiresSharedTargetKV: Bool { false }
    public var requiresPromptPrefill: Bool { true }
    public var requiresGreedySampling: Bool { true }

    public func validateCompatibility(with _: any LanguageModel) throws {}

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [])
    }

    public func finalizeDrafterState(
        target _: any LanguageModel,
        targetBoundaryHidden _: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens _: Int,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool {
        state.nextPosition = targetProcessedTokenCount
        state.seedToken = nil
        state.seedHidden = nil
        state.proposalAppended = 0
        return true
    }

    public func resumeDrafterState(
        target _: any LanguageModel,
        suffixTokens _: MLXArray,
        suffixTargetHidden _: MLXArray,
        targetBoundaryHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> Bool {
        true
    }

    public func appendDrafterState(
        target _: any LanguageModel,
        tokens: MLXArray,
        targetHidden _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> Bool {
        state.nextPosition += tokens.dim(-1)
        return true
    }

    public func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(
            batchSize: lastToken.dim(0), queryOffset: queryOffset, blockSize: blockSize)
    }

    public func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(
            batchSize: lastToken.dim(0), queryOffset: queryOffset, blockSize: blockSize)
    }

    private func draftedTokens(
        batchSize: Int,
        queryOffset: Int,
        blockSize: Int
    ) -> MLXArray {
        let token =
            queryOffset == 1
            ? SyntheticMTPValues.endOfSequenceToken : SyntheticMTPValues.ordinaryToken
        return SyntheticMTPValues.makeRepeatedDraft(
            of: Int32(token), batchSize: batchSize, blockSize: blockSize)
    }
}
