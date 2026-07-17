// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Creates a function that decodes configuration data and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (Data) throws -> M {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/loadContainer(from:using:configuration:useLatest:progressHandler:)``.
public enum LLMTypeRegistry {

    /// Shared instance with default model types.
    public static let shared: ModelTypeRegistry<LanguageModel> = .init(creators: [
        "mistral": create(LlamaConfiguration.self, LlamaModel.init),
        "mixtral": create(MixtralConfiguration.self, MixtralModel.init),
        "llama": create(LlamaConfiguration.self, LlamaModel.init),
        "phi": create(PhiConfiguration.self, PhiModel.init),
        "phi3": create(Phi3Configuration.self, Phi3Model.init),
        "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
        "gemma": create(GemmaConfiguration.self, GemmaModel.init),
        "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
        "gemma3": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
        "gemma3_text": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
        "gemma3n": create(Gemma3nTextConfiguration.self, Gemma3nTextModel.init),
        "gemma4": create(Gemma4Configuration.self, Gemma4Model.init),
        "gemma4_unified": create(Gemma4Configuration.self, Gemma4Model.init),
        "gemma4_text": create(Gemma4TextConfiguration.self, Gemma4TextModel.init),
        "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
        "qwen3": create(Qwen3Configuration.self, Qwen3Model.init),
        "qwen3_moe": create(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
        "qwen3_next": create(Qwen3NextConfiguration.self, Qwen3NextModel.init),
        "qwen3_5": create(Qwen35Configuration.self, Qwen35Model.init),
        "qwen3_5_moe": create(Qwen35Configuration.self, Qwen35MoEModel.init),
        "qwen3_5_text": create(Qwen35TextConfiguration.self, Qwen35TextModel.init),
        "minicpm": create(MiniCPMConfiguration.self, MiniCPMModel.init),
        "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
        "cohere": create(CohereConfiguration.self, CohereModel.init),
        "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
        "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
        "deepseek_v3": create(DeepseekV3Configuration.self, DeepseekV3Model.init),
        "granite": create(GraniteConfiguration.self, GraniteModel.init),
        "granitemoehybrid": create(
            GraniteMoeHybridConfiguration.self, GraniteMoeHybridModel.init),
        "mimo": create(MiMoConfiguration.self, MiMoModel.init),
        "mimo_v2_flash": create(MiMoV2FlashConfiguration.self, MiMoV2FlashModel.init),
        "minimax": create(MiniMaxConfiguration.self, MiniMaxModel.init),
        "glm4": create(GLM4Configuration.self, GLM4Model.init),
        "glm4_moe": create(GLM4MoEConfiguration.self, GLM4MoEModel.init),
        "glm4_moe_lite": create(GLM4MoELiteConfiguration.self, GLM4MoELiteModel.init),
        "acereason": create(Qwen2Configuration.self, Qwen2Model.init),
        "falcon_h1": create(FalconH1Configuration.self, FalconH1Model.init),
        "bitnet": create(BitnetConfiguration.self, BitnetModel.init),
        "smollm3": create(SmolLM3Configuration.self, SmolLM3Model.init),
        "ernie4_5": create(Ernie45Configuration.self, Ernie45Model.init),
        "lfm2": create(LFM2Configuration.self, LFM2Model.init),
        "baichuan_m1": create(BaichuanM1Configuration.self, BaichuanM1Model.init),
        "exaone4": create(Exaone4Configuration.self, Exaone4Model.init),
        "gpt_oss": create(GPTOSSConfiguration.self, GPTOSSModel.init),
        "lille-130m": create(Lille130mConfiguration.self, Lille130mModel.init),
        "olmoe": create(OlmoEConfiguration.self, OlmoEModel.init),
        "olmo2": create(Olmo2Configuration.self, Olmo2Model.init),
        "olmo3": create(Olmo3Configuration.self, Olmo3Model.init),
        "bailing_moe": create(BailingMoeConfiguration.self, BailingMoeModel.init),
        "lfm2_moe": create(LFM2MoEConfiguration.self, LFM2MoEModel.init),
        "nanochat": create(NanoChatConfiguration.self, NanoChatModel.init),
        "nemotron_h": create(NemotronHConfiguration.self, NemotronHModel.init),
        "afmoe": create(AfMoEConfiguration.self, AfMoEModel.init),
        "jamba": create(JambaConfiguration.self, JambaModel.init),
        "mamba2": create(Mamba2Configuration.self, Mamba2Model.init),
        "mistral3": create(Mistral3TextConfiguration.self, Mistral3TextModel.init),
        // Ministral3 (e.g. Devstral-2-123B) shares the Mistral3 text architecture:
        // Mistral3Text.swift is the port of mlx-lm's ministral3.py.
        "ministral3": create(Mistral3TextConfiguration.self, Mistral3TextModel.init),
        "apertus": create(ApertusConfiguration.self, ApertusModel.init),
        "nemotron_labs_diffusion": create(
            NemotronLabsDiffusionConfiguration.self, NemotronLabsDiffusionModel.init),
    ])
}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The Python tokenizers have a very rich set of implementations and configuration. The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
///
/// `@unchecked Sendable` synchronization invariant: all mutable state (the id → configuration
/// dictionary) lives in `AbstractModelRegistry` and every access is guarded by its `NSLock`.
/// This subclass adds no instance storage of its own — only immutable `static let`
/// configurations — so instances are safe to share across concurrency domains.
public class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = LLMRegistry(modelConfigurations: all())

    /// Model configuration for `mlx-community/SmolLM-135M-Instruct-4bit`.
    static public let smolLm135m4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    /// Model configuration for `mlx-community/Mistral-Nemo-Instruct-2407-4bit`.
    static public let mistralNemo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    /// Model configuration for `mlx-community/Mistral-7B-Instruct-v0.3-4bit`.
    static public let mistral7b4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    /// Model configuration for `mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX`.
    static public let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    /// Model configuration for `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`.
    static public let deepSeekR17b4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    /// Model configuration for `tiiuae/Falcon-H1R-7B`.
    static public let falconH1r7b = ModelConfiguration(
        id: "tiiuae/Falcon-H1R-7B",
        defaultPrompt: "If the product of two numbers is 360 and their GCD is 6, what is their LCM?"
    )

    /// Model configuration for `mlx-community/phi-2-hf-4bit-mlx`.
    static public let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/Phi-3.5-mini-instruct-4bit`.
    static public let phi354bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    /// Model configuration for `mlx-community/Phi-3.5-MoE-instruct-4bit`.
    static public let phi35Moe = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    /// Model configuration for `mlx-community/quantized-gemma-2b-it`.
    static public let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    /// Model configuration for `mlx-community/gemma-2-9b-it-4bit`.
    static public let gemma29bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    /// Model configuration for `mlx-community/gemma-2-2b-it-4bit`.
    static public let gemma22bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    /// Model configuration for `mlx-community/gemma-3-1b-it-qat-4bit`.
    static public let gemma31bQat4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-1b-it-qat-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Model configuration for `mlx-community/gemma-3n-E4B-it-lm-bf16`.
    static public let gemma3nE4bItLmBf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Model configuration for `mlx-community/gemma-3n-E2B-it-lm-bf16`.
    static public let gemma3nE2bItLmBf16 = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-bf16",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Model configuration for `mlx-community/gemma-3n-E4B-it-lm-4bit`.
    static public let gemma3nE4bItLm4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E4B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Model configuration for `mlx-community/gemma-3n-E2B-it-lm-4bit`.
    static public let gemma3nE2bItLm4bit = ModelConfiguration(
        id: "mlx-community/gemma-3n-E2B-it-lm-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        // https://ai.google.dev/gemma/docs/core/prompt-structure
        extraEOSTokens: ["<end_of_turn>"]
    )

    /// Model configuration for `mlx-community/gemma-4-e4b-it-4bit`.
    static public let gemma4E4bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-e4b-it-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<turn|>"]
    )

    /// Model configuration for `mlx-community/gemma-4-e2b-it-4bit`.
    static public let gemma4E2bIt4bit = ModelConfiguration(
        id: "mlx-community/gemma-4-e2b-it-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<turn|>"]
    )

    /// Model configuration for `mlx-community/Qwen1.5-0.5B-Chat-4bit`.
    static public let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        defaultPrompt: "why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen2.5-7B-Instruct-4bit`.
    static public let qwen257b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.
    static public let qwen2515b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3-0.6B-4bit`.
    static public let qwen306b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-0.6B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3-1.7B-4bit`.
    static public let qwen317b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-1.7B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3-4B-4bit`.
    static public let qwen34b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-4B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3-8B-4bit`.
    static public let qwen38b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-8B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3-30B-A3B-4bit`.
    static public let qwen3Moe30bA3b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-30B-A3B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3.5-2B-4bit`.
    static public let qwen352b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.5-2B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/Qwen3.6-27B-4bit`.
    static public let qwen3627b4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.6-27B-4bit",
        defaultPrompt: "Why is the sky blue?",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Model configuration for `mlx-community/OpenELM-270M-Instruct`.
    static public let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    /// Model configuration for `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.
    static public let llama318b4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<|eot_id|>"]
    )

    /// Model configuration for `mlx-community/Meta-Llama-3-8B-Instruct-4bit`.
    static public let llama38b4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<|eot_id|>"]
    )

    /// Model configuration for `mlx-community/Llama-3.2-1B-Instruct-4bit`.
    static public let llama321b4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<|eot_id|>"]
    )

    /// Model configuration for `mlx-community/Llama-3.2-3B-Instruct-4bit`.
    static public let llama323b4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?",
        extraEOSTokens: ["<|eot_id|>"]
    )

    /// Model configuration for `mlx-community/DeepSeek-R1-4bit`.
    static public let deepseekR14bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    /// Model configuration for `mlx-community/granite-3.3-2b-instruct-4bit`.
    static public let granite332b4bit = ModelConfiguration(
        id: "mlx-community/granite-3.3-2b-instruct-4bit",
        defaultPrompt: ""
    )

    /// Model configuration for `mlx-community/MiMo-7B-SFT-4bit`.
    static public let mimo7bSft4bit = ModelConfiguration(
        id: "mlx-community/MiMo-7B-SFT-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/GLM-4-9B-0414-4bit`.
    static public let glm49b4bit = ModelConfiguration(
        id: "mlx-community/GLM-4-9B-0414-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .glm4Bare
    )

    /// Model configuration for `mlx-community/AceReason-Nemotron-7B-4bit`.
    static public let acereason7b4bit = ModelConfiguration(
        id: "mlx-community/AceReason-Nemotron-7B-4bit",
        defaultPrompt: ""
    )

    /// Model configuration for `mlx-community/bitnet-b1.58-2B-4T-4bit`.
    static public let bitnetB1582b4t4bit = ModelConfiguration(
        id: "mlx-community/bitnet-b1.58-2B-4T-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/Baichuan-M1-14B-Instruct-4bit-ft`.
    static public let baichuanM114bInstruct4bit = ModelConfiguration(
        id: "mlx-community/Baichuan-M1-14B-Instruct-4bit-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/SmolLM3-3B-4bit`.
    static public let smolLm33b4bit = ModelConfiguration(
        id: "mlx-community/SmolLM3-3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft`.
    static public let ernie4503bPtBf16Ft = ModelConfiguration(
        id: "mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/LFM2-1.2B-4bit`.
    static public let lfm212b4bit = ModelConfiguration(
        id: "mlx-community/LFM2-1.2B-4bit",
        defaultPrompt: "Why is the sky blue?",
        toolCallFormat: .lfm2
    )

    /// Model configuration for `mlx-community/exaone-4.0-1.2b-4bit`.
    static public let exaone4012b4bit = ModelConfiguration(
        id: "mlx-community/exaone-4.0-1.2b-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/lille-130m-instruct-bf16`.
    static public let lille130mBf16 = ModelConfiguration(
        id: "mlx-community/lille-130m-instruct-bf16",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit`.
    static public let olmoe1b7b0125Instruct4bit = ModelConfiguration(
        id: "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/OLMo-2-1124-7B-Instruct-4bit`.
    static public let olmo211247bInstruct4bit = ModelConfiguration(
        id: "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/Ling-mini-2.0-2bit-DWQ`.
    static public let lingMini2bit = ModelConfiguration(
        id: "mlx-community/Ling-mini-2.0-2bit-DWQ",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/Granite-4.0-H-Tiny-4bit-DWQ`.
    static public let granite40hTiny4bitDwq = ModelConfiguration(
        id: "mlx-community/Granite-4.0-H-Tiny-4bit-DWQ",
        defaultPrompt: ""
    )

    /// Model configuration for `mlx-community/LFM2-8B-A1B-3bit-MLX`.
    static public let lfm28bA1b3bitMlx = ModelConfiguration(
        id: "mlx-community/LFM2-8B-A1B-3bit-MLX",
        defaultPrompt: "",
        toolCallFormat: .lfm2
    )

    /// Model configuration for `dnakov/nanochat-d20-mlx`.
    static public let nanochatD20Mlx = ModelConfiguration(
        id: "dnakov/nanochat-d20-mlx",
        defaultPrompt: ""
    )

    /// Model configuration for `mlx-community/gpt-oss-20b-MXFP4-Q8`.
    static public let gptOss20bMxfp4Q8 = ModelConfiguration(
        id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        defaultPrompt: "Why is the sky blue?"
    )

    /// Model configuration for `mlx-community/AI21-Jamba-Reasoning-3B-4bit`.
    static public let jamba3b4bit = ModelConfiguration(
        id: "mlx-community/AI21-Jamba-Reasoning-3B-4bit",
        defaultPrompt: ""
    )

    /// Model configuration for `mlx-community/Nemotron-Labs-Diffusion-3B-4bit`.
    static public let nemotronLabsDiffusion3b4bit = ModelConfiguration(
        id: "mlx-community/Nemotron-Labs-Diffusion-3B-4bit",
        defaultPrompt: "Explain quaternions."
    )

    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
            deepSeekR17b4bit,
            falconH1r7b,
            gemma2bQuantized,
            gemma22bIt4bit,
            gemma29bIt4bit,
            gemma31bQat4bit,
            gemma3nE4bItLmBf16,
            gemma3nE2bItLmBf16,
            gemma3nE4bItLm4bit,
            gemma3nE2bItLm4bit,
            gemma4E4bIt4bit,
            gemma4E2bIt4bit,
            granite332b4bit,
            granite40hTiny4bitDwq,
            llama318b4bit,
            llama321b4bit,
            llama323b4bit,
            llama38b4bit,
            mistral7b4bit,
            mistralNemo4bit,
            openelm270m4bit,
            phi35Moe,
            phi354bit,
            phi4bit,
            qwen205b4bit,
            qwen257b,
            qwen2515b,
            qwen306b4bit,
            qwen317b4bit,
            qwen34b4bit,
            qwen38b4bit,
            qwen3Moe30bA3b4bit,
            qwen352b4bit,
            qwen3627b4bit,
            smolLm135m4bit,
            deepseekR14bit,
            mimo7bSft4bit,
            glm49b4bit,
            acereason7b4bit,
            bitnetB1582b4t4bit,
            smolLm33b4bit,
            ernie4503bPtBf16Ft,
            lfm212b4bit,
            baichuanM114bInstruct4bit,
            exaone4012b4bit,
            lille130mBf16,
            olmoe1b7b0125Instruct4bit,
            olmo211247bInstruct4bit,
            lingMini2bit,
            lfm28bA1b3bitMlx,
            nanochatD20Mlx,
            gptOss20bMxfp4Q8,
            jamba3b4bit,
            nemotronLabsDiffusion3b4bit,
        ]
    }

}

/// Deprecated alias for ``LLMRegistry``. Use ``LLMRegistry`` directly instead.
@available(*, deprecated, renamed: "LLMRegistry", message: "Please use LLMRegistry directly.")
public typealias ModelRegistry = LLMRegistry

private struct LLMUserInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)

            return LMInput(tokens: MLXArray(promptTokens))
        } catch TokenizerError.missingChatTemplate {
            print(
                "No chat template was included or provided, so converting messages to simple text format. This is not optimal for model performance, so applications should provide a chat template if none is included with the model."
            )
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama38b4bit)
/// ```
public final class LLMModelFactory: GenericModelFactory {

    /// The context type produced by this factory: ``ModelContext``.
    public typealias ContextType = ModelContext

    /// The container type produced by this factory: ``ModelContainer``.
    public typealias ContainerType = ModelContainer

    /// Creates a factory with custom type and model registries.
    ///
    /// - Parameters:
    ///   - typeRegistry: registry mapping model types, e.g. `llama`, to code that can
    ///     decode their configuration and instantiate the model
    ///   - modelRegistry: registry of model id, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`,
    ///     to ``ModelConfiguration``
    public init(
        typeRegistry: ModelTypeRegistry<LanguageModel>, modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry<LanguageModel>

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry: AbstractModelRegistry

    /// Loads a model from the resolved configuration, assembling a ready-to-use ``ModelContext``.
    ///
    /// Decodes `config.json`, instantiates the model via the ``typeRegistry``, loads the
    /// weights and tokenizer in parallel, and applies EOS token, stop string, tool call
    /// format, and reasoning protocol overrides from the configuration files.
    ///
    /// This is the primitive used by the `load(...)` / `loadContainer(...)` family of
    /// methods — callers typically use those instead of calling this directly.
    ///
    /// - Parameters:
    ///   - configuration: the resolved model configuration, including the model directory
    ///   - tokenizerLoader: loader used to produce the tokenizer for the model
    /// - Returns: a ``ModelContext`` holding the model, tokenizer, and input processor
    /// - Throws: ``ModelFactoryError`` if the configuration files cannot be read or decoded,
    ///   or any error thrown while loading the weights or tokenizer
    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> ModelContext {
        let modelDirectory = configuration.modelDirectory

        // Load config.json once and decode for both base config and model-specific config
        let configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // Load EOS token IDs from config.json, with optional override from generation_config.json
        var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        let generationConfig: GenerationConfigFile? =
            if let generationData = try? Data(contentsOf: generationConfigURL) {
                try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: generationData)
            } else {
                nil
            }
        if let genEosIds = generationConfig?.eosTokenIds?.values {
            eosTokenIds = Set(genEosIds)  // Override per Python mlx-lm behavior
        }

        // Build a ModelConfiguration with loaded EOS token IDs and tool call format
        var mutableConfiguration = configuration
        mutableConfiguration.eosTokenIds = eosTokenIds
        mutableConfiguration.stopStrings.formUnion(generationConfig?.stopStrings ?? [])
        if mutableConfiguration.toolCallFormat == nil {
            mutableConfiguration.toolCallFormat = ToolCallFormat.infer(
                from: baseConfig.modelType, configData: configData)
        }
        // Reasoning protocol: registry override wins; otherwise infer from
        // model_type + repo id. `modelID` is load-bearing — R1-Distill reports a
        // base model_type (qwen2/llama) and is only recognizable by id.
        if mutableConfiguration.reasoningConfig == nil {
            mutableConfiguration.reasoningConfig = ReasoningConfig.infer(
                from: baseConfig.modelType, modelID: configuration.name, configData: configData)
        }

        // Load tokenizer and weights in parallel
        async let tokenizerTask = tokenizerLoader.load(
            from: configuration.tokenizerDirectory)

        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let tokenizer = try await tokenizerTask

        let messageGenerator =
            if let model = model as? LLMModel {
                model.messageGenerator(tokenizer: tokenizer)
            } else {
                DefaultMessageGenerator()
            }

        // Build a ModelConfiguration for the ModelContext
        let tokenizerSource: TokenizerSource? =
            configuration.tokenizerDirectory == modelDirectory
            ? nil
            : .directory(configuration.tokenizerDirectory)
        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: tokenizerSource,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: mutableConfiguration.extraEOSTokens,
            stopStrings: mutableConfiguration.stopStrings,
            eosTokenIds: mutableConfiguration.eosTokenIds,
            toolCallFormat: mutableConfiguration.toolCallFormat,
            reasoningConfig: mutableConfiguration.reasoningConfig)

        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer, configuration: modelConfig,
            messageGenerator: messageGenerator)

        return .init(
            configuration: modelConfig, model: model, processor: processor,
            tokenizer: tokenizer)
    }

}

/// Trampoline that exposes ``LLMModelFactory`` to `MLXLMCommon` via dynamic (Objective-C
/// runtime) lookup, letting `MLXLMCommon` discover the LLM factory without a compile-time
/// dependency on this module.
public class TrampolineModelFactory: NSObject, ModelFactoryTrampoline {
    /// Returns the shared ``LLMModelFactory`` instance for use as `MLXLMCommon`'s model factory.
    public static func modelFactory() -> (any MLXLMCommon.ModelFactory)? {
        LLMModelFactory.shared
    }
}
