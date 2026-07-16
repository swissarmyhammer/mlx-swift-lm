// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct CoherenceIntegrationTests {

    @Test func bitnet_b1_58_2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.bitnetB1582b4t4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func exaone_4_0_1_2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.exaone4012b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func gemma3_1B_qat() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.gemma31bQat4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func gemma3n_E2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.gemma3nE2bItLm4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func gemma4_e2b() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.gemma4E2bIt4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func glm4_9B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.glm49b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func granite3_3_2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.granite332b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func granite4_0_H_tiny() async throws {
        let container = try await models.llmContainer(
            for: LLMRegistry.granite40hTiny4bitDwq)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func jamba_3B_4bit() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.jamba3b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func lfm2_1_2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.lfm212b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func llama3_2_1B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.llama321b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func mistral_7B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.mistral7b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func olmo2_7B() async throws {
        let container = try await models.llmContainer(
            for: LLMRegistry.olmo211247bInstruct4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test(
        .disabled(
            """
            OLMoE-1B-7B-0125-Instruct's tokenizer_class is "GPTNeoXTokenizer", \
            which swift-transformers 1.3.3 (the `Tokenizers` package backing \
            `#huggingFaceTokenizerLoader()`) does not register in \
            `TokenizerModel.knownTokenizers` -- confirmed by reading \
            Sources/Tokenizers/Tokenizer.swift in the resolved swift-transformers \
            checkout, where an unregistered class throws \
            `TokenizerError.unsupportedTokenizer(name)` under the default \
            `strict: true` path rather than silently falling back to BPE. This is \
            a genuine upstream library gap, not a bug in this project's tokenizer \
            loading code: there is no per-model override point between \
            `#huggingFaceTokenizerLoader()` and `AutoTokenizer.from` to register \
            an additional tokenizer class. Explicitly gated here rather than \
            picking a different OLMoE-representative model, since OLMoE-1B-7B is \
            the only MoE-family model in this coherence sweep and dropping it \
            would silently lose that coverage axis. Re-enable once \
            swift-transformers registers GPTNeoXTokenizer (it is a byte-level BPE \
            tokenizer, so `BPETokenizer` would very likely handle it correctly \
            -- re-verify decode/encode parity against the Python tokenizer before \
            trusting it, though) or once this project grows its own \
            tokenizer-class-to-implementation bridge.
            """
        )
    )
    func olmoe_1B_7B() async throws {
        let container = try await models.llmContainer(
            for: LLMRegistry.olmoe1b7b0125Instruct4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func phi3_5() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.phi354bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func qwen3_1_7B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.qwen317b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func qwen3_5_2B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.qwen352b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }

    @Test func smollm3_3B() async throws {
        let container = try await models.llmContainer(for: LLMRegistry.smolLm33b4bit)
        try await ChatSessionTests.planetsCoherence(container: container)
    }
}
