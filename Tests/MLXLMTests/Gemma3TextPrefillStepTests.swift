// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXScriptedLM
import XCTest

@testable import MLXLLM

/// The prefill step of Gemma 3 text when the caller sets none: the tuned
/// 128-token step, capped at the sliding window, in the prepare and in the
/// checkpoint feeds of a prompt cache, on tiny random-weight models.
final class Gemma3TextPrefillStepTests: XCTestCase {

    /// The prefill step that `Gemma3TextModel.prepare` is tuned to on Apple
    /// Silicon.
    private let tunedStep = 128

    /// A sliding window wider than the tuned step, as in the Gemma 3
    /// checkpoints (512 or 1024 tokens).
    private let wideWindow = 512

    /// A sliding window narrower than the tuned step, which caps every step.
    private let narrowWindow = 16

    /// A step that a caller sets, wider than either window.
    private let wideCallerStep = 1024

    /// The vocabulary of the test models.
    private let vocabularySize = 97

    /// The seed of the random weights of the test models.
    private let weightSeed: UInt64 = 7

    /// A Gemma 3 text model with a sliding-window layer and a global layer,
    /// thus a rotating cache and a standard cache.
    private func model(slidingWindow: Int) -> Gemma3TextModel {
        MLXRandom.seed(weightSeed)
        let model = Gemma3TextModel(
            Gemma3TextConfiguration(
                modelType: "gemma3_text", hiddenSize: 32, hiddenLayers: 2,
                intermediateSize: 64, attentionHeads: 4, headDim: 8, rmsNormEps: 1e-6,
                vocabularySize: vocabularySize, kvHeads: 2, ropeTheta: 1_000_000,
                ropeLocalBaseFreq: 10_000, ropeTraditional: false, queryPreAttnScalar: 8,
                slidingWindow: slidingWindow, slidingWindowPattern: 2,
                maxPositionEmbeddings: 4096))
        eval(model)
        return model
    }

    /// The counts of `count` positions read in steps of `step`.
    private func steps(of step: Int, through count: Int) -> [Int] {
        Array(stride(from: step, through: count, by: step))
    }

    // MARK: - The step and the cap

    /// The model takes the tuned step when the caller sets none.
    func testDefaultPrefillStepIsTheTunedStep() {
        XCTAssertEqual(model(slidingWindow: wideWindow).defaultPrefillStepSize, tunedStep)
    }

    /// The model caps every prefill step at its sliding window.
    func testMaximumPrefillStepIsTheSlidingWindow() {
        XCTAssertEqual(model(slidingWindow: narrowWindow).maximumPrefillStepSize, narrowWindow)
    }

    /// A model that forwards to Gemma 3 text keeps its step and its cap.
    func testForwardingModelKeepsTheStepAndTheCap() {
        let model = ForwardingLanguageModel(wrapping: model(slidingWindow: narrowWindow))
        XCTAssertEqual(model.defaultPrefillStepSize, tunedStep)
        XCTAssertEqual(model.maximumPrefillStepSize, narrowWindow)
    }

    // MARK: - Checkpoint feeds

    /// A checkpoint feed with no step from the caller reads in the tuned
    /// step.
    func testCheckpointFeedReadsInTheTunedStep() throws {
        let count = 2 * tunedStep
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                model(slidingWindow: wideWindow), tokenCount: count),
            steps(of: tunedStep, through: count))
    }

    /// A checkpoint feed with no step from the caller caps the tuned step at
    /// a narrower window.
    func testCheckpointFeedCapsTheTunedStepAtTheWindow() throws {
        let count = 3 * narrowWindow
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                model(slidingWindow: narrowWindow), tokenCount: count),
            steps(of: narrowWindow, through: count))
    }

    /// A checkpoint feed caps the step a caller sets at the window.
    func testCheckpointFeedCapsTheStepOfTheCallerAtTheWindow() throws {
        let count = 3 * narrowWindow
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                model(slidingWindow: narrowWindow), tokenCount: count, stepSize: wideCallerStep),
            steps(of: narrowWindow, through: count))
    }

    /// A checkpoint feed of a model that forwards to Gemma 3 text keeps the
    /// cap.
    func testForwardingModelCheckpointFeedCapsTheStepAtTheWindow() throws {
        let count = 3 * narrowWindow
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                ForwardingLanguageModel(wrapping: model(slidingWindow: narrowWindow)),
                tokenCount: count, stepSize: wideCallerStep),
            steps(of: narrowWindow, through: count))
    }

    /// A checkpoint feed in steps capped at the window, followed by one
    /// forward of the last token, gives the logits and the greedy token of
    /// the whole prompt in one forward.
    func testCappedCheckpointFeedMatchesTheUnchunkedReference() throws {
        let model = model(slidingWindow: narrowWindow)
        let tokens = (0 ..< 4 * narrowWindow + 3).map { ($0 * 31 + 5) % vocabularySize }

        let referenceLogits = model(
            MLXArray(tokens.map(Int32.init))[.newAxis], cache: try model.newCache(parameters: nil)
        )[0, -1]

        let caches = try model.newCache(parameters: nil)
        _ = try PromptCacheCheckpoint.prefill(
            Array(tokens.dropLast()), into: caches, state: nil, model: model,
            prefill: PrefillParameters(stepSize: wideCallerStep))
        let fedLogits = model(MLXArray([Int32(tokens.last!)])[.newAxis], cache: caches)[0, -1]

        XCTAssertEqual(fedLogits.argMax().item(Int.self), referenceLogits.argMax().item(Int.self))
        XCTAssertTrue(
            allClose(fedLogits, referenceLogits, rtol: 1e-4, atol: 1e-4).item(Bool.self),
            "max difference \(abs(fedLogits - referenceLogits).max().item(Float.self))")
    }

    // MARK: - Prepare

    /// The prepare caps the step a caller sets at the window, and leaves the
    /// last token for the iterator.
    func testPrepareCapsTheStepOfTheCallerAtTheWindow() throws {
        let model = model(slidingWindow: narrowWindow)
        let count = 3 * narrowWindow + 1
        let log = PrefillProgressLog()
        _ = try model.prepare(
            LMInput(tokens: MLXArray((0 ..< count).map { Int32($0 % vocabularySize) })),
            cache: try model.newCache(parameters: nil), state: nil,
            prefill: log.recordingPrefill(stepSize: wideCallerStep))
        XCTAssertEqual(log.processed, steps(of: narrowWindow, through: count - 1))
    }
}
