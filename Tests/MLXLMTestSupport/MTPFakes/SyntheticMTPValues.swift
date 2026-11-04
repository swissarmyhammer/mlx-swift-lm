// Copyright © 2026 Apple Inc.

import MLX

/// The values that the synthetic MTP targets and drafters write, named once for every fake.
public enum SyntheticMTPValues {

    /// The logit of the token a fake selects. Any value with a clear gap to the other logits
    /// wins `argmax`.
    public static let selectedLogit: Float = 100

    /// The logit of every token a chat-session fake does not select.
    public static let rejectedLogit: Float = -100

    /// The width of each hidden row and of each key and value row that a fake emits.
    public static let rowWidth = 4

    /// The sequence axis of a key or value array: batch, heads, sequence, width.
    public static let sequenceAxis = -2

    /// The ordinary token that the chat-session fakes predict and draft.
    public static let ordinaryToken = 4

    /// The end-of-sequence token of the tokenizers that the chat-session fakes run under.
    public static let endOfSequenceToken = 15

    /// The token that the iterator mock drafters propose when a test names none.
    public static let draftedToken: Int32 = 7

    /// The block size of a Qwen-style drafter: one drafted token and the bonus token.
    public static let qwenBlockSize = 2

    /// Makes the draft block a fake drafter returns when it proposes one token at every
    /// drafted position.
    ///
    /// - Parameters:
    ///   - token: The token proposed at every drafted position.
    ///   - batchSize: The number of rows of the block.
    ///   - blockSize: The verification block size: the drafted tokens plus the bonus token.
    /// - Returns: A `[batchSize, blockSize - 1]` array of `token`.
    public static func makeRepeatedDraft(
        of token: Int32, batchSize: Int, blockSize: Int
    ) -> MLXArray {
        let draftedCount = blockSize - 1
        return MLXArray(
            Array(repeating: token, count: batchSize * draftedCount), [batchSize, draftedCount])
    }
}
