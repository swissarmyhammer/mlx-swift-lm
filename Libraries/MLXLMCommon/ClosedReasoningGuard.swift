// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Processor that masks the end delimiter of a reasoning block once a block
/// has closed in the generation.
///
/// A model whose chat template writes one reasoning block for each assistant
/// message has no use for a second end delimiter in a pass. Flint is such a
/// model: its `</think>` is a plain token, and after a status line it can put
/// a fifth to a half of its probability on a second `</think>` in place of the
/// `<|user|>` that ends its turn. After that token it writes the next user turn
/// as plain text and does not stop. The mask takes the token away, so the
/// model ends the turn with its next choice.
///
/// The processor keeps the closed state on the GPU, so it reads no sampled
/// token back to the CPU and the decode stays pipelined. It depends on the
/// tokens it samples, thus a speculative verifier treats it as a stateful
/// processor and feeds it each accepted token in order.
public struct ClosedReasoningEndProcessor: LogitProcessor {

    /// The token ID of the end delimiter.
    public let endTokenId: Int

    /// Whether a reasoning block has closed: a `[1]` boolean array.
    private var closed: MLXArray

    /// Creates a processor that masks `endTokenId` once it has been sampled.
    ///
    /// - Parameter endTokenId: the token ID of the end delimiter.
    public init(endTokenId: Int) {
        self.endTokenId = endTokenId
        self.closed = MLXArray([false])
    }

    /// Starts a generation. A prompt that ends with the end delimiter (a
    /// closed, empty block that turns thinking off) has closed its block.
    public mutating func prompt(_ prompt: MLXArray) {
        let tokens = prompt.flattened()
        closed =
            tokens.size > 0
            ? (tokens[(tokens.size - 1)...] .== MLXArray(Int32(endTokenId)))
            : MLXArray([false])
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard endTokenId < logits.dim(-1) else { return logits }
        let indices = MLXArray([UInt32(endTokenId)])[.newAxis, 0...]
        let current = takeAlong(logits, indices, axis: -1)
        let masked = MLX.where(closed, MLXArray(-Float.infinity).asType(logits.dtype), current)
        return putAlong(logits, indices, values: masked, axis: -1)
    }

    public mutating func didSample(token: MLXArray) {
        closed = closed .|| (token.flattened()[0 ..< 1] .== MLXArray(Int32(endTokenId)))
    }
}

/// Side table of the end delimiter each model masks after its reasoning
/// closes. Keys are held weakly, so entries disappear with their model.
private final class ClosedReasoningEndRegistry: @unchecked Sendable {

    static let shared = ClosedReasoningEndRegistry()

    private let lock = NSLock()
    private let table = NSMapTable<AnyObject, NSNumber>.weakToStrongObjects()

    func set(_ tokenId: Int?, for model: AnyObject) {
        lock.lock()
        defer { lock.unlock() }
        if let tokenId {
            table.setObject(NSNumber(value: tokenId), forKey: model)
        } else {
            table.removeObject(forKey: model)
        }
    }

    func tokenId(for model: AnyObject) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return table.object(forKey: model)?.intValue
    }
}

/// The token ID that `reasoning` masks after its block closes, or `nil`.
///
/// It is `nil` when the protocol does not set
/// ``ReasoningConfig/masksEndDelimiterAfterClose``, and when its end delimiter
/// is not exactly one token of `tokenizer`.
///
/// - Parameters:
///   - reasoning: the reasoning protocol of the model.
///   - tokenizer: the tokenizer of the model.
/// - Returns: the token ID of the end delimiter, or `nil`.
public func closedReasoningEndTokenId(
    reasoning: ReasoningConfig?, tokenizer: any Tokenizer
) -> Int? {
    guard let reasoning, reasoning.masksEndDelimiterAfterClose, reasoning.channel == nil
    else { return nil }
    let tokens = tokenizer.encode(text: reasoning.endDelimiter, addSpecialTokens: false)
    return tokens.count == 1 ? tokens[0] : nil
}

/// Records the end delimiter that `model` masks after its reasoning closes.
///
/// The model factories call this after they load the model and its tokenizer.
/// Every token iterator then chains a ``ClosedReasoningEndProcessor`` for the
/// model. A protocol that does not ask for the mask clears the record.
///
/// - Parameters:
///   - model: the loaded model.
///   - reasoning: the reasoning protocol of the model.
///   - tokenizer: the tokenizer of the model.
public func registerClosedReasoningEndGuard(
    model: any LanguageModel, reasoning: ReasoningConfig?, tokenizer: any Tokenizer
) {
    ClosedReasoningEndRegistry.shared.set(
        closedReasoningEndTokenId(reasoning: reasoning, tokenizer: tokenizer), for: model)
}

/// The ``ClosedReasoningEndProcessor`` of a model, or `nil` when the model
/// masks no end delimiter.
func makeClosedReasoningEndProcessor(model: any LanguageModel) -> ClosedReasoningEndProcessor? {
    ClosedReasoningEndRegistry.shared.tokenId(for: model).map {
        ClosedReasoningEndProcessor(endTokenId: $0)
    }
}
