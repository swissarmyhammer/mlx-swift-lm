// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// Real-weights image Q&A integration test for MiniMax-M3's vision tower
/// (kanban ^9a2aw98): loads `mlx-community/MiniMax-M3-4bit` (or the
/// `MLX_MINIMAX_M3_CHECKPOINT` override) through `VLMModelFactory` and asserts
/// the model names the color of a synthetic solid-color test image -- the
/// same color-naming pattern `VisionIntegrationTests.namesImageColor(color:)`
/// uses for Qwen3-VL, adapted to this repository's lower-level
/// `ChatSession`/`vlmContainer` path (matching `MiniMaxM3CoherenceIntegrationTests`'s
/// own text-only coherence test in this same file group, rather than the
/// FoundationModels adapter `VisionIntegrationTests` uses).
///
/// This is an `IntegrationTesting`-only suite -- `IntegrationTesting/` is an
/// Xcode project run via `xcodebuild`, not part of `swift test` -- and skips
/// gracefully (passes trivially) rather than failing when the machine lacks
/// enough memory, the checkpoint override path doesn't exist, or the
/// checkpoint fails to download/load (treated the same as "absent"), mirroring
/// `MiniMaxM3CoherenceIntegrationTests`'s own gating helpers
/// (`minimaxM3RequiredMemoryBytes`, `resolveMiniMaxM3Configuration`,
/// `checkpointIsAvailable`) verbatim. Bandwidth-limited in every prior task of
/// this chain (^xgvth41, ^wz8y8qq, ^b90razv) and in this task (^9a2aw98) --
/// the real checkpoint is 120-214 GB -- so this test is written but has not
/// been exercised against real weights; see the kanban comments on ^9a2aw98.
/// Run explicitly via:
/// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/MiniMaxM3ImageQAIntegrationTests`
@Suite(.serialized, .timeLimit(.minutes(240)))
struct MiniMaxM3ImageQAIntegrationTests {

    /// Colors exercised by ``namesImageColor(color:)`` -- mirrors
    /// `VisionIntegrationTests.TestColor`: two colors give an implicit
    /// negative control (a model that always answers "red" fails the blue
    /// case), and `Sendable` with a plain `String` raw value keeps this a
    /// valid parameterized-test argument (`CIColor` is not `Sendable`).
    enum TestColor: String, CaseIterable, Sendable {
        case red, blue

        var ciColor: CIColor { self == .red ? .red : .blue }
    }

    @Test(arguments: TestColor.allCases)
    func namesImageColor(color: TestColor) async throws {
        guard ProcessInfo.processInfo.physicalMemory >= minimaxM3RequiredMemoryBytes else {
            print(
                "Skipping MiniMax-M3 image Q&A test: physical memory "
                    + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                    + "\(minimaxM3RequiredMemoryBytes) bytes")
            return
        }

        let configuration = resolveMiniMaxM3Configuration()
        guard checkpointIsAvailable(configuration) else {
            print("Skipping MiniMax-M3 image Q&A test: checkpoint not found at \(configuration.name)")
            return
        }

        let container: LLModelContainer
        do {
            container = try await models.vlmContainer(for: configuration)
        } catch {
            print(
                "Skipping MiniMax-M3 image Q&A test: failed to load checkpoint "
                    + "\(configuration.name): \(error)")
            return
        }

        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 50, temperature: 0))
        let image = VisionTestImages.solidColor(color.ciColor)

        var result = ""
        for try await token in session.streamResponse(
            to: "What color is this image? Reply with just the color name.",
            images: [.ciImage(image)]
        ) {
            result += token
        }

        // Whole-word match: split on non-letters so trailing punctuation
        // ("red.") still counts, while "colored"/"coloured" cannot satisfy a
        // color name.
        let words = Set(
            result.lowercased()
                .split(whereSeparator: { !$0.isLetter })
                .map(String.init))
        #expect(
            words.contains(color.rawValue),
            "expected the model to name the color \(color.rawValue); got: \(result)")
    }
}
