// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// An input processor for a vision-language model on random weights: it renders the messages of
/// a `UserInput` with a ``ScalarTokenizer``, and marks each image of the input with image tokens
/// and fixed pixels.
///
/// The tokens of each image follow the text of the turn: the token that opens an image, then
/// one placeholder token for each patch of the image grid. The tokens take the batched form
/// `[1, n]` that a vision-language `prepare` requires.
public struct ScalarImageInputProcessor: UserInputProcessor {

    /// The tokenizer that renders the messages.
    public let tokenizer: ScalarTokenizer

    /// The tokens that mark an image.
    public let imageTokens: Qwen35FixtureJSON.ImageTokens

    /// The patch grid of each image.
    public let imageGrid: THW

    /// Gives the pixels of one image.
    public let makeImage: @Sendable () -> LMInput.ProcessedImage

    /// Makes a processor that renders with `tokenizer` and marks images with `imageTokens`.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer that renders the messages.
    ///   - imageTokens: The tokens that mark an image.
    ///   - imageGrid: The patch grid of each image, which sets the placeholder count.
    ///   - makeImage: Gives the pixels of one image, of `imageGrid`.
    public init(
        tokenizer: ScalarTokenizer, imageTokens: Qwen35FixtureJSON.ImageTokens, imageGrid: THW,
        makeImage: @escaping @Sendable () -> LMInput.ProcessedImage
    ) {
        self.tokenizer = tokenizer
        self.imageTokens = imageTokens
        self.imageGrid = imageGrid
        self.makeImage = makeImage
    }

    /// The tokens of the messages of `input`, then the tokens of each of its images, and the
    /// pixels of those images.
    ///
    /// - Parameter input: The input of one turn.
    /// - Returns: The rendered tokens, with the images when `input` has any.
    /// - Throws: The error of the chat template.
    public func prepare(input: UserInput) async throws -> LMInput {
        let text = try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(from: input), tools: input.tools,
            additionalContext: input.additionalContext)
        let placeholders = Array(
            repeating: imageTokens.image, count: imageGrid.t * imageGrid.h * imageGrid.w)
        let imageMarks = input.images.flatMap { _ in [imageTokens.visionStart] + placeholders }
        let tokens = MLXArray((text + imageMarks).map(Int32.init))[.newAxis, 0...]
        let images = input.images.map { _ in makeImage() }
        guard !images.isEmpty else { return LMInput(text: .init(tokens: tokens)) }
        return LMInput(
            text: .init(tokens: tokens),
            image: LMInput.ProcessedImage(
                pixels: concatenated(images.map(\.pixels)),
                frames: images.flatMap { $0.frames ?? [] }))
    }
}
