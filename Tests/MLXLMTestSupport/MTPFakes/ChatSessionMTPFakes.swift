// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLMCommon
import MLXNN

/// Deterministic attention-only target that publishes the same MTP state shape
/// used by Qwen text models, while keeping the ChatSession tests lightweight.
public final class MTPStateEmittingTarget: Module, LanguageModel, KVCacheDimensionProvider {
    /// The number of tokens of the fake vocabulary.
    public static let vocabularySize = 16

    /// The value the target writes under ``stagingTargetMarkerKey`` when it runs with a probe.
    public static let stagingTargetMarker = 42

    private let probe: MTPStagingProbe?
    public var kvHeads: [Int] { [1] }

    public init(probe: MTPStagingProbe? = nil) {
        self.probe = probe
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        guard let probe else { return try [makeAttentionKVCache(parameters: parameters)] }
        probe.cacheCreationCount += 1
        return [probe.cache]
    }

    public func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        let cache: [KVCache] =
            probe == nil
            ? try newCache(parameters: parameters)
            : [SpeculativeStagingCache()]
        return KVCacheStatus(
            cache: cache, plan: try parameters?.kvCachePlan() ?? .disabled, phase: .planned)
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        if let probe, let cache = cache.first as? SpeculativeStagingCache {
            probe.preparations.append(
                .init(
                    cacheID: ObjectIdentifier(cache), cachedTokens: cache.cachedTokens,
                    tokens: input.text.tokens.asArray(Int.self),
                    emitsMTPState: state?[mtpEmitFlagKey] == true,
                    targetMarker: state?[stagingTargetMarkerKey]))
            if probe.failAfterPreparationWrite {
                let values = input.text.tokens.asType(.float32).reshaped([1, 1, -1, 1])
                _ = cache.update(keys: values, values: values)
                throw KVCacheError(message: "Target preparation failed after a cache write")
            }
        }
        return .tokens(input.text)
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
        let newKV =
            probe == nil
            ? MLXArray.zeros([1, 1, positions, 1])
            : tokens.asType(.float32).reshaped([1, 1, positions, 1])
        let sharedKV = cache?.first?.update(keys: newKV, values: newKV) ?? (newKV, newKV)

        var values = [Float](
            repeating: SyntheticMTPValues.rejectedLogit, count: positions * Self.vocabularySize)
        for position in 0 ..< positions {
            values[position * Self.vocabularySize + SyntheticMTPValues.ordinaryToken] =
                SyntheticMTPValues.selectedLogit
        }
        let logits = MLXArray(values, [1, positions, Self.vocabularySize])

        var outputState = LMOutput.State()
        if probe != nil {
            outputState[stagingTargetMarkerKey] = Self.stagingTargetMarker
        }
        guard emitMTPState else {
            return LMOutput(logits: logits, state: probe == nil ? nil : outputState)
        }
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

/// Qwen-like contract: private per-stream state, prompt prefill, greedy-only,
/// no dependency on target-shared K/V, and one drafted token per round.
public final class QwenStyleMTPDrafter: Module, ResumableMTPDrafterModel {
    private let probe: MTPStagingProbe?

    public init(probe: MTPStagingProbe? = nil) {
        self.probe = probe
    }

    public var maximumBlockSize: Int? { SyntheticMTPValues.qwenBlockSize }
    public var requiresSharedTargetKV: Bool { false }
    public let requiresPromptPrefill = true
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
        probe?.finalizedPositions.append(targetProcessedTokenCount)
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

    public func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(batchSize: lastToken.dim(0), blockSize: blockSize)
    }

    public func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) -> MLXArray {
        draftedTokens(batchSize: lastToken.dim(0), blockSize: blockSize)
    }

    private func draftedTokens(batchSize: Int, blockSize: Int) -> MLXArray {
        if let probe { probe.draftCount += 1 }
        return MLXArray(
            Array(
                repeating: Int32(SyntheticMTPValues.ordinaryToken),
                count: batchSize * (blockSize - 1)),
            [batchSize, blockSize - 1]
        )
    }
}

/// A model-state key that `MTPStateEmittingTarget` writes when it runs with a probe, so a test
/// can see whether the target state reached the next call.
public let stagingTargetMarkerKey = LMOutput.Key<Int>("staging-test-target-marker")

/// A record of what `MTPStateEmittingTarget` and `QwenStyleMTPDrafter` saw, and a switch that
/// makes the target fail after it wrote into its cache.
public final class MTPStagingProbe {

    /// One call of the target's `prepare`.
    public struct Preparation {

        /// The identity of the cache that the call prepared into.
        public let cacheID: ObjectIdentifier

        /// The keys that the cache held when the call began.
        public let cachedTokens: [Float]

        /// The tokens of the call's input.
        public let tokens: [Int]

        /// Whether the call's state asked the target to emit drafter state.
        public let emitsMTPState: Bool

        /// The value of ``stagingTargetMarkerKey`` in the call's state.
        public let targetMarker: Int?
    }

    /// The one cache that the target hands out for every `newCache` call.
    public let cache = SpeculativeStagingCache()

    /// The number of `newCache` calls.
    public var cacheCreationCount = 0

    /// Each `prepare` call of the target, in order.
    public var preparations: [Preparation] = []

    /// The number of blocks the drafter drafted.
    public var draftCount = 0

    /// The processed token count of each drafter finalization, in order.
    public var finalizedPositions: [Int] = []

    /// Whether the target throws from `prepare` after it wrote the input into its cache.
    public var failAfterPreparationWrite = false

    /// Makes a probe with nothing recorded.
    public init() {}
}

/// A `KVCacheSimple` that can refuse to stage a speculative round wider than one token, and
/// that records each width it refused.
public final class SpeculativeStagingCache: KVCache {
    private let storage: KVCacheSimple
    public var refusesSpeculativeStaging = false
    public var refusedWidths: [Int] = []

    public init(storage: KVCacheSimple = KVCacheSimple()) {
        self.storage = storage
    }

    public var offset: Int { storage.offset }
    public var maxSize: Int? { storage.maxSize }
    public var isTrimmable: Bool { storage.isTrimmable }
    public var cachedTokens: [Float] { storage.state.first?.asArray(Float.self) ?? [] }
    public var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }
    public var metaState: [String] {
        get { storage.metaState }
        set { storage.metaState = newValue }
    }

    public func isTrimmable(after positions: Int) -> Bool {
        if refusesSpeculativeStaging && positions > 1 {
            refusedWidths.append(positions)
            return false
        }
        return storage.isTrimmable(after: positions)
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int { storage.trim(n) }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func innerState() -> [MLXArray] { storage.innerState() }

    public func copy() -> any KVCache {
        let copy = SpeculativeStagingCache(storage: storage.copy() as! KVCacheSimple)
        copy.refusesSpeculativeStaging = refusesSpeculativeStaging
        return copy
    }
}

/// Gemma-style MTP fixture: all conditioning state comes from the target, so
/// no iterator-owned continuation is needed between turns.
public class StatelessMTPDrafter: Module, MTPDrafterModel {
    public var maximumBlockSize: Int? { SyntheticMTPValues.qwenBlockSize }
    public var requiresSharedTargetKV: Bool { true }
    public var requiresPromptPrefill: Bool { false }

    public func validateCompatibility(with _: any LanguageModel) throws {}

    public func draftBlock(
        target _: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        MLXArray(
            Array(
                repeating: Int32(SyntheticMTPValues.ordinaryToken),
                count: lastToken.dim(0) * (blockSize - 1)),
            [lastToken.dim(0), blockSize - 1])
    }
}

/// Stateful but deliberately lacking `ResumableMTPDrafterModel`. It proves
/// that the capability is additive and absence degrades only speculation.
public final class NonResumableMTPDrafter: StatelessMTPDrafter,
    StatefulMTPDrafterModel
{
    public override var requiresSharedTargetKV: Bool { false }
    public override var requiresPromptPrefill: Bool { true }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [])
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state _: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden,
            sharedKV: sharedKV, positionDeltas: positionDeltas,
            queryOffset: queryOffset, blockSize: blockSize, sampler: sampler)
    }
}
