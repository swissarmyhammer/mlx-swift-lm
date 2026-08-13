// Copyright © 2026 Apple Inc.

import MLXLLM
import XCTest

final class LLMRegistryTests: XCTestCase {

    func testDeepSeekV4FlashModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "mlx-community/DeepSeek-V4-Flash-4bit"))

        let configuration = LLMRegistry.deepseek_v4_flash_4bit
        XCTAssertEqual(configuration.name, "mlx-community/DeepSeek-V4-Flash-4bit")
        XCTAssertEqual(configuration.defaultPrompt, "Why is the sky blue?")
    }

    func testFalconH1RModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "tiiuae/Falcon-H1R-7B"))

        let configuration = LLMRegistry.falconH1R7B
        XCTAssertEqual(configuration.name, "tiiuae/Falcon-H1R-7B")
    }

    /// Characterization: pins one representative configuration per shared default
    /// prompt so the literal → named-constant extraction cannot silently change
    /// any serialized prompt value.
    func testRegistryDefaultPromptsRemainStable() {
        XCTAssertEqual(
            LLMRegistry.smolLM_135M_4bit.defaultPrompt, "Tell me about the history of Spain.")
        XCTAssertEqual(
            LLMRegistry.deepseek_r1_4bit.defaultPrompt, "Tell me about the history of Spain.")
        XCTAssertEqual(LLMRegistry.mistralNeMo4bit.defaultPrompt, "Explain quaternions.")
        XCTAssertEqual(
            LLMRegistry.nemotron_labs_diffusion_3b_4bit.defaultPrompt, "Explain quaternions.")
        XCTAssertEqual(
            LLMRegistry.phi3_5_4bit.defaultPrompt, "What is the gravity on Mars and the moon?")
        XCTAssertEqual(
            LLMRegistry.phi3_5MoE.defaultPrompt, "What is the gravity on Mars and the moon?")
        XCTAssertEqual(LLMRegistry.phi4bit.defaultPrompt, "Why is the sky blue?")
        XCTAssertEqual(LLMRegistry.qwen2_5_7b.defaultPrompt, "Why is the sky blue?")
        XCTAssertEqual(LLMRegistry.gpt_oss_20b_MXFP4_Q8.defaultPrompt, "Why is the sky blue?")
        // Pre-existing lowercase variant — deliberately preserved verbatim.
        XCTAssertEqual(LLMRegistry.qwen205b4bit.defaultPrompt, "why is the sky blue?")
        XCTAssertEqual(
            LLMRegistry.gemma3_1B_qat_4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(
            LLMRegistry.gemma4_e4b_it_4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(
            LLMRegistry.llama3_1_8B_4bit.defaultPrompt,
            "What is the difference between a fruit and a vegetable?")
        XCTAssertEqual(LLMRegistry.granite3_3_2b_4bit.defaultPrompt, "")
        XCTAssertEqual(LLMRegistry.jamba_3b_4bit.defaultPrompt, "")
    }

    /// All three Gemma 2 configurations share the capitalized lettuce-vs-cabbage
    /// prompt (the historical lowercase variant on `gemma2bQuantized` was
    /// normalized when the shared prompt was extracted to a constant).
    func testGemma2ConfigurationsShareCapitalizedLettucePrompt() {
        for configuration in [
            LLMRegistry.gemma2bQuantized,
            LLMRegistry.gemma_2_9b_it_4bit,
            LLMRegistry.gemma_2_2b_it_4bit,
        ] {
            XCTAssertEqual(
                configuration.defaultPrompt,
                "What is the difference between lettuce and cabbage?")
        }
    }
}
