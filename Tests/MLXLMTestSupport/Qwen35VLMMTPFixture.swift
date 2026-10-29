// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// A tiny Qwen 3.5 vision-language target with its MTP drafter, on seeded random weights.
///
/// The language model has the hybrid layout of ``Qwen35MTPFixture``: one Gated DeltaNet
/// (linear, recurrent) layer, then one full-attention layer. The VLM target routes a long or
/// warm prompt through its windowed continuation forward, which the text target does not have,
/// so a test of the MTP prefill reaches that path with this fixture. Both models take their
/// weights from a seed, thus every run of a test decodes the same tokens.
public struct Qwen35VLMMTPFixture {

    /// The id of the last token of the vocabulary.
    private static let lastTokenID = Qwen35MTPFixture.vocabularySize - 1

    /// The image tokens of a fixture that reads images: the two tokens at the top of the
    /// vocabulary.
    public static let imageTokens = Qwen35FixtureJSON.ImageTokens(
        visionStart: lastTokenID - 1, image: lastTokenID)

    /// The patches along each side of the image ``makeImage()`` gives.
    private static let patchesPerSide = 2

    /// The patch grid of the image ``makeImage()`` gives: one frame of ``patchesPerSide`` by
    /// ``patchesPerSide`` patches.
    public static let imageGrid = THW(1, patchesPerSide, patchesPerSide)

    /// The color channels of each pixel of an image.
    private static let colorChannelCount = 3

    /// The pixels along each side of one patch: the `patch_size` of
    /// ``Qwen35FixtureJSON/visionLanguage(textConfigJSON:visionOutputSize:imageTokens:)``,
    /// whose `temporal_patch_size` makes each patch one frame.
    private static let patchSize = 2

    /// The values of one patch: the color channels of one frame of ``patchSize`` by
    /// ``patchSize`` pixels.
    public static let patchValueCount = colorChannelCount * patchSize * patchSize

    /// The configuration of the model, as a loaded model would carry it.
    public let modelConfiguration = ModelConfiguration(id: "tiny-qwen35-vlm-mtp")

    /// The target model.
    public let target: Qwen35

    /// The MTP drafter of the target.
    public let drafter: Qwen35VLMNextNDraftModel

    /// The tokenizer of the target.
    public let tokenizer = ScalarTokenizer(vocabularySize: Qwen35MTPFixture.vocabularySize)

    /// Makes the target and the drafter from `seed`.
    ///
    /// - Parameters:
    ///   - seed: The seed of the weights.
    ///   - readsImages: Whether the target reads images: its vision tower then gives features of
    ///     the width of its text model, and ``imageTokens`` mark the images of a prompt. The
    ///     vision tower then has other weights, which changes the weights the seed gives the
    ///     language model, so a target that reads images decodes other tokens than one that
    ///     does not.
    /// - Throws: The error of the configuration decoder.
    public init(seed: UInt64 = Qwen35MTPFixture.defaultSeed, readsImages: Bool = false) throws {
        let textConfigJSON = Qwen35FixtureJSON.text(
            mtpLayers: Qwen35MTPFixture.mtpLayers,
            hiddenLayers: Qwen35MTPFixture.hiddenLayers,
            fullAttentionInterval: Qwen35MTPFixture.fullAttentionInterval,
            hiddenSize: Qwen35MTPFixture.hiddenSize,
            vocabularySize: Qwen35MTPFixture.vocabularySize,
            tieWordEmbeddings: false)
        let json =
            readsImages
            ? Qwen35FixtureJSON.visionLanguage(
                textConfigJSON: textConfigJSON, visionOutputSize: Qwen35MTPFixture.hiddenSize,
                imageTokens: Self.imageTokens)
            : Qwen35FixtureJSON.visionLanguage(textConfigJSON: textConfigJSON)
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35(configuration)
        }
        let drafter = withRandomState(
            MLXRandom.RandomState(seed: seed &+ Qwen35MTPFixture.drafterSeedOffset)
        ) {
            Qwen35VLMNextNDraftModel(configuration)
        }
        eval(target, drafter)
        self.target = target
        self.drafter = drafter
    }

    /// One image of ``imageGrid``, with fixed pixel values.
    ///
    /// - Returns: The processed image.
    public static func makeImage() -> LMInput.ProcessedImage {
        let patchCount = imageGrid.t * imageGrid.h * imageGrid.w
        let valueCount = patchCount * patchValueCount
        let pixels = MLXArray(0 ..< valueCount).asType(.float32) / Float(valueCount)
        return LMInput.ProcessedImage(
            pixels: pixels.reshaped(patchCount, patchValueCount), frames: [imageGrid])
    }

    /// A model container over the target, with the tokenizer of the fixture and an input
    /// processor that marks each image of a turn with ``imageTokens`` and gives it the pixels
    /// of ``makeImage()``.
    ///
    /// - Returns: The container.
    public func makeModelContainer() -> ModelContainer {
        ModelContainer(
            context: ModelContext(
                configuration: modelConfiguration, model: target,
                processor: ScalarImageInputProcessor(
                    tokenizer: tokenizer, imageTokens: Self.imageTokens, imageGrid: Self.imageGrid,
                    makeImage: Self.makeImage),
                tokenizer: tokenizer))
    }
}
