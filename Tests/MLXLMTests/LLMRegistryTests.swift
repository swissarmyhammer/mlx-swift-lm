// Copyright © 2026 Apple Inc.

import MLXLLM
import XCTest

final class LLMRegistryTests: XCTestCase {

    func testDeepSeekV4FlashModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "mlx-community/DeepSeek-V4-Flash-4bit"))

        let configuration = LLMRegistry.deepseekV4Flash4bit
        XCTAssertEqual(configuration.name, "mlx-community/DeepSeek-V4-Flash-4bit")
        XCTAssertEqual(configuration.defaultPrompt, "Why is the sky blue?")
    }

    func testFalconH1RModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "tiiuae/Falcon-H1R-7B"))

        let configuration = LLMRegistry.falconH1r7b
        XCTAssertEqual(configuration.name, "tiiuae/Falcon-H1R-7B")
    }

    /// Characterization: pins one representative configuration per shared default
    /// prompt so the literal → named-constant extraction cannot silently change
    /// any serialized prompt value.
    func testRegistryDefaultPromptsRemainStable() {
        XCTAssertEqual(
            LLMRegistry.smollm135m4bit.defaultPrompt, "Tell me about the history of Spain.")
        XCTAssertEqual(
            LLMRegistry.deepseekR14bit.defaultPrompt, "Tell me about the history of Spain.")
        XCTAssertEqual(LLMRegistry.mistralNemo4bit.defaultPrompt, "Explain quaternions.")
        XCTAssertEqual(
            LLMRegistry.nemotronLabsDiffusion3b4bit.defaultPrompt, "Explain quaternions.")
        XCTAssertEqual(
            LLMRegistry.phi354bit.defaultPrompt, "What is the gravity on Mars and the moon?")
        XCTAssertEqual(
            LLMRegistry.phi35MoE.defaultPrompt, "What is the gravity on Mars and the moon?")
        XCTAssertEqual(LLMRegistry.phi4bit.defaultPrompt, "Why is the sky blue?")
        XCTAssertEqual(LLMRegistry.qwen257b.defaultPrompt, "Why is the sky blue?")
        XCTAssertEqual(LLMRegistry.gptOSS20bMXFP4Q8.defaultPrompt, "Why is the sky blue?")
        // Pre-existing lowercase variant — deliberately preserved verbatim.
        XCTAssertEqual(LLMRegistry.qwen205b4bit.defaultPrompt, "why is the sky blue?")
        XCTAssertEqual(
            LLMRegistry.gemma31bQAT4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(
            LLMRegistry.gemma4E4bIT4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(
            LLMRegistry.llama318b4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(LLMRegistry.granite332b4bit.defaultPrompt, "")
        XCTAssertEqual(LLMRegistry.jamba3b4bit.defaultPrompt, "")
    }

    /// All three Gemma 2 configurations share the capitalized lettuce-vs-cabbage
    /// prompt (the historical lowercase variant on `gemma2bQuantized` was
    /// normalized when the shared prompt was extracted to a constant).
    func testGemma2ConfigurationsShareCapitalizedLettucePrompt() {
        for configuration in [
            LLMRegistry.gemma2bQuantized,
            LLMRegistry.gemma29bIT4bit,
            LLMRegistry.gemma22bIT4bit,
        ] {
            XCTAssertEqual(
                configuration.defaultPrompt,
                "What is the difference between lettuce and cabbage?")
        }
    }
}
