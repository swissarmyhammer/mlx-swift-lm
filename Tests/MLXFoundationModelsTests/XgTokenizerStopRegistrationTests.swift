// Copyright © 2026 Apple Inc.
//
// Regression coverage for kanban task `y4s0w2j`: the grammar tokenizer
// must register the model's FULL stop-token set with xgrammar — the same
// set `GuidedGenerationLoop.buildStopTokenIDs` stops generation on.
// GLM-4.7-Flash's `config.json` declares `eos_token_id: [154820, 154827,
// 154829]` (`<|endoftext|>`, `<|user|>`, `<|observation|>`); with only
// `tokenizer.eosTokenId` registered, the unregistered ids stayed
// ordinary vocab entries the grammar was free to admit as string
// content, and live constrained decode sampled one mid-`response`,
// truncating the envelope.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Byte-per-token tokenizer with an EOS at 255, mirroring the guided-
/// generation test tokenizers.
private struct StopRegistrationTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == "</s>" { return 255 }
        if token == "<extra_stop>" { return 254 }
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == 255 { return "</s>" }
        if id == 254 { return "<extra_stop>" }
        guard id >= 0, id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Grammar tokenizer stop-token registration")
struct XgTokenizerStopRegistrationTests {

    @Test("makeXgTokenizer registers every host-side stop token with xgrammar")
    func registersFullStopSet() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let tokenizer = StopRegistrationTokenizer()
        // The GLM shape: config.json's eos_token_id array carries
        // secondary turn-enders beyond the tokenizer's own EOS, and a
        // registry entry adds an extra stop token by string.
        let configuration = ModelConfiguration(
            id: "org/stop-registration-\(UUID().uuidString)",
            extraEOSTokens: ["<extra_stop>"],
            eosTokenIds: [200, 201]
        )

        let xg = try await MLXLanguageModel.makeXgTokenizer(
            modelID: configuration.name, tokenizer: tokenizer, configuration: configuration)

        let registered = Set(xg.stopTokenIDs.map(Int.init))
        #expect(registered.isSuperset(of: [200, 201]), "configuration.eosTokenIds must register")
        #expect(registered.contains(255), "the tokenizer's own EOS must register")
        #expect(registered.contains(254), "extraEOSTokens must register")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
