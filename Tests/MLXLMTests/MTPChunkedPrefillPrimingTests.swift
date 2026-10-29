// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon

/// The prompt tokens the target's prefill hands the drafter as one chunk.
private let prefillChunkLength = 4

/// A prompt longer than one prefill chunk, so the target's final forward evaluates the rest.
private let chunkedPrompt: [Int32] = [1, 2, 3, 4, 5, 6]

/// The tokens each stream generates: enough for the first speculative round to draft.
private let generatedTokenCount = 4

/// A target whose prefill evaluates the first ``prefillChunkLength`` prompt tokens itself and
/// hands their hidden rows to the drafter's chunk handler, as a chunked prefill does, leaving the
/// rest of the prompt to the iterator's final forward.
private final class ChunkFeedingTarget: Module, LanguageModel, KVCacheDimensionProvider {
    private let inner: MTPStateEmittingTarget

    var kvHeads: [Int] { inner.kvHeads }

    /// Makes a target that evaluates every forward with `inner`.
    ///
    /// - Parameter inner: The target that evaluates the chunk and every later forward.
    init(inner: MTPStateEmittingTarget) {
        self.inner = inner
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        guard let chunkHandler = state?[mtpPrefillChunkHandlerKey],
            tokens.dim(-1) > prefillChunkLength
        else { return .tokens(input.text) }
        let chunk = tokens[..<prefillChunkLength]
        let chunkOutput = inner(.init(tokens: chunk[.newAxis]), cache: cache, state: state)
        let chunkHidden = try #require(chunkOutput.state?[mtpLastHiddenStatesKey])
        chunkHandler(chunk, chunkHidden, nil)
        return .tokens(.init(tokens: tokens[prefillChunkLength...]))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inner(inputs, cache: cache)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        inner(input, cache: cache, state: state)
    }
}

/// A stream over ``chunkedPrompt`` whose drafter takes the prompt chunk by chunk.
private struct ChunkedPrefillRun {

    /// What the drafter and the target saw.
    let probe: MTPStagingProbe

    /// The tokens the stream generated.
    let tokens: [Int]

    /// The reason the stream decoded without speculation, or `nil` when it speculated.
    let passthroughReason: String?

    /// Generates ``generatedTokenCount`` tokens greedily, after `configure` set up the probe.
    ///
    /// - Parameter configure: Sets the probe's switches before the stream starts.
    init(configure: (MTPStagingProbe) -> Void = { _ in }) throws {
        let probe = MTPStagingProbe()
        configure(probe)
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(chunkedPrompt)),
            mainModel: ChunkFeedingTarget(inner: MTPStateEmittingTarget(probe: probe)),
            drafter: QwenStyleMTPDrafter(probe: probe),
            parameters: GenerateParameters(maxTokens: generatedTokenCount, temperature: 0),
            blockSize: SyntheticMTPValues.qwenBlockSize)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        self.probe = probe
        self.tokens = tokens
        self.passthroughReason = iterator.passthroughReason
    }
}

@Suite("MTP drafter priming after a chunked prefill")
struct MTPChunkedPrefillPrimingTests {

    @Test("the final prompt segment resumes the drafter from the boundary the chunks left")
    func finalSegmentResumesFromChunkBoundary() throws {
        let run = try ChunkedPrefillRun()

        #expect(run.tokens.count == generatedTokenCount)
        #expect(run.passthroughReason == nil)
        #expect(run.probe.coldStartPositions.isEmpty)
        #expect(run.probe.draftCount > 0)
    }

    @Test("a refused resume cold-starts the drafter at the position after the chunks")
    func refusedResumeColdStartsAfterChunks() throws {
        let run = try ChunkedPrefillRun { $0.refusesResumption = true }

        #expect(run.passthroughReason == nil)
        #expect(run.probe.coldStartPositions == [prefillChunkLength])
        #expect(run.probe.draftCount > 0)
    }

    @Test("a drafter that can neither resume nor cold-start leaves the stream to the target")
    func refusedColdStartDecodesPlainly() throws {
        let run = try ChunkedPrefillRun {
            $0.refusesResumption = true
            $0.refusesColdStart = true
        }

        #expect(run.tokens.count == generatedTokenCount)
        #expect(run.passthroughReason == MTPSpeculativeTokenIterator.coldStartFailedReason)
        #expect(run.probe.draftCount == 0)
    }
}
