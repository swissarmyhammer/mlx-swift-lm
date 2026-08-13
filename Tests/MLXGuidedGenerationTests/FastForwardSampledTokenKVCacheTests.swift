// Copyright © 2026 Apple Inc.
//
// Regression test for kanban task `w7nvvg8`: the sampled token that
// triggers a non-empty fast-forward (FF) batch must still get its own
// forward pass through the model, exactly like the non-FF path
// (`advanceSingleSampledToken`) does. `CommitResult.tokens` never echoes
// the sampled token back (see `GrammarConstraint.commitToken`'s doc
// comment), so before the fix, `processFastForwardTokens` fed only the FF
// tokens through the model and silently skipped the sampled token's own
// KV-cache entry -- `onTokenCommitted` (documented as firing for "every
// token actually fed through the model ... sampled tokens and
// fast-forward tokens alike") would then omit it too.
//
// Uses a byte-fallback vocab (same shape as `ConstraintCachingTests`'
// `makeByteFallbackTokenizer()`) and a literal EBNF grammar so the FF path
// is exercised deterministically without any real model weights: a
// forced multi-byte literal guarantees `FindJumpForwardString` returns a
// non-empty, multi-byte suffix after the first commit, and
// `emitFastForwardLocked`'s boundary-safety trim (the last FF-encoded
// token always closes the boundary and is left to the sampler) leaves at
// least one byte for a final, separately-sampled commit -- exercising the
// FF-batch path this test targets. The grammar's own EOS token (reserved
// id 255) is registered as the host tokenizer's `eosToken` so `run()`'s
// existing premature-EOS check (unrelated to this fix) stops generation
// cleanly once the literal is complete, instead of trying to commit/emit
// the grammar's internal stop token as ordinary text.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Minimal byte-per-character tokenizer: token id `n` (0..<255) is exactly
/// UTF-8 byte value `n`; id 255 is reserved as the special EOS token (never
/// a real byte value emitted by the grammar's literal). Serves as both
/// `GrammarConstraint`'s `hostTokenizer` and `context.tokenizer` -- its
/// `encode`/`decode` round-trip is all `emitFastForwardLocked`'s boundary
/// trim and `NaiveStreamingDetokenizer` need, and registering the EOS id
/// lets `run()`'s stop-token set (`buildStopTokenIDs`) recognize it so the
/// loop stops before ever trying to decode it as a raw byte.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    static let eosTokenID = 255
    private static let eosTokenString = "<eos>"

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == Self.eosTokenString { return Self.eosTokenID }
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eosTokenID { return Self.eosTokenString }
        guard id >= 0, id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { Self.eosTokenString }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// `UserInputProcessor` stand-in. `GuidedGenerationLoop.run` never touches
/// `context.processor`, so this is only present to satisfy `ModelContext`'s
/// initializer and traps if that assumption ever changes.
private struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("GuidedGenerationLoop.run must not use context.processor")
    }
}

/// Records every token id actually fed through `callAsFunction` (prompt
/// prefill and every subsequent single-token forward pass alike), in call
/// order. Always returns all-zero logits over `vocabSize` -- irrelevant to
/// which token gets sampled, since the literal grammar's mask forces
/// exactly one allowed bit at every step regardless of the model's raw
/// logit values.
private final class RecordingProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    static let vocabSize = 256

    var kvHeads: [Int] { [1] }
    private(set) var fedTokenIDs: [Int] = []

    func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fedTokenIDs.append(contentsOf: inputs.asArray(Int32.self).map(Int.init))
        return MLXArray.zeros([1, inputs.size, Self.vocabSize])
    }
}

@Suite
struct FastForwardSampledTokenKVCacheTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    private func makeByteFallbackGrammarTokenizer() throws -> GrammarTokenizer {
        let vocab: [String] = (0 ..< RecordingProbeModel.vocabSize).map { byte in
            String(format: "<0x%02X>", byte)
        }
        return try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            eosTokenId: Int32(ByteTokenizer.eosTokenID)
        )
    }

    /// - Throws: Rethrows any error from building the grammar tokenizer or
    ///   constraint, or from `GuidedGenerationLoop.run`, surfacing setup or
    ///   generation failures as a test failure rather than swallowing them.
    @Test(
        """
        The sampled token that triggers a non-empty FF batch gets its own model \
        forward pass, matching the non-FF path and onTokenCommitted's documented contract
        """
    )
    func sampledTokenIsFedThroughModelWhenFastForwardFires() throws {
        let hostTokenizer = ByteTokenizer()
        let grammarTokenizer = try makeByteFallbackGrammarTokenizer()

        // Literal 4-byte grammar: after committing 'A' (byte 65), the
        // remaining "BCD" is fully forced. The boundary-safety trim in
        // `emitFastForwardLocked` drops the last encoded FF token (it
        // straddles the forced/unforced boundary), so with 3 remaining
        // bytes each mapping to its own token, exactly 2 ("B", "C") come
        // back as a non-empty FF batch and "D" is left for a separate,
        // ordinary (non-FF) commit -- covering the FF-batch path this test
        // targets. Once "D" is committed the grammar's only remaining
        // allowed token is its own EOS (id 255, registered as this
        // tokenizer's `eosToken`), which `run()`'s existing premature-EOS
        // check recognizes and stops on before ever committing/emitting it.
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer,
            grammar: "root ::= \"ABCD\"\n",
            fastForward: true,
            hostTokenizer: hostTokenizer
        )

        let model = RecordingProbeModel()
        let context = ModelContext(
            configuration: ModelConfiguration(id: "fast-forward-kv-cache-test"),
            model: model,
            processor: UnusedProcessor(),
            tokenizer: hostTokenizer
        )

        // Single arbitrary prompt token (byte 0); its own accounting is
        // irrelevant to this test, which only inspects the generation-phase
        // callback below.
        let input = LMInput(tokens: MLXArray([Int32(0)]))

        var committedTokenIDs: [Int] = []
        var emittedText = ""

        let result = try GuidedGenerationLoop.run(
            input: input,
            context: context,
            constraint: constraint,
            maxTokens: 10,
            vocabSize: RecordingProbeModel.vocabSize,
            onTokenCommitted: { committedTokenIDs.append($0) },
            emit: { text in
                emittedText += text
                return true
            }
        )

        #expect(emittedText == "ABCD", "all four grammar-forced bytes must still reach the caller")
        #expect(result.tokenCount == 4)

        // The bug: without the fix, `committedTokenIDs` is only [66, 67, 68]
        // ("B", "C", "D") -- the sampled token 'A' (65) that triggered the
        // FF batch is silently never fed through the model, even though it
        // was emitted and accepted by the grammar. 'D' (68) IS correctly
        // present even before the fix (it takes the ordinary non-FF path,
        // `advanceSingleSampledToken`, since its own commit produces an
        // empty FF batch); the grammar's actual EOS token (255) is what's
        // correctly absent, stopped on before ever being committed/emitted.
        #expect(
            committedTokenIDs == [65, 66, 67, 68],
            "sampled token 'A' (65) must be fed through the model alongside its FF batch ['B'=66, 'C'=67] -- got \(committedTokenIDs)"
        )

        // Cross-check against the model's own call log: every generation
        // token reported via `onTokenCommitted` must correspond to an
        // actual forward pass, and the prompt token (0) must precede them.
        #expect(model.fedTokenIDs == [0, 65, 66, 67, 68])
    }
}
