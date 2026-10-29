// Copyright © 2026 Apple Inc.

import MLXLMTestSupport
import Testing

@testable import MLXLLM

/// The Qwen 3.5 kernel model holds every token of the decode prompts the Qwen 3.5 suites share.
///
/// The embedding of a model reads its table with no bounds check, so a token id past the
/// vocabulary reads the bytes that follow the table in memory. Those bytes change from one
/// process to the next, thus a decode of such a prompt is not repeatable across processes.
@Suite(.serialized)
struct Qwen35KernelModelVocabularyTests {

    @Test(
        "each shared Qwen 3.5 decode prompt is in the vocabulary of the kernel model",
        arguments: [Qwen35CompiledVerifyTests.prompt, Qwen35VerifyRewindTests.decodePrompt])
    func sharedDecodePromptIsInTheKernelModelVocabulary(prompt: [Int]) throws {
        let vocabulary = 0 ..< (try Qwen35CompiledVerifyTests.kernelModel().vocabularySize)

        #expect(prompt.allSatisfy(vocabulary.contains), "\(prompt) in \(vocabulary)")
    }

    @Test("the kernel model reads the vocabulary of the Qwen 3.5 MTP fixtures")
    func kernelModelVocabularyIsTheMTPFixtureVocabulary() throws {
        let model = try Qwen35CompiledVerifyTests.kernelModel()

        #expect(model.vocabularySize == Qwen35MTPFixture.vocabularySize)
    }
}
