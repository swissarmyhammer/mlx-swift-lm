// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// The size of the process-global MLX buffer-reuse pool that ``MLXLanguageModel`` sets, for
/// GPUs of different recommended working sets.
@Suite("MLXLanguageModel GPU buffer pool")
struct MLXLanguageModelGPUCacheLimitTests {

    /// The bytes of one binary gibibyte.
    private static let bytesPerGibibyte = 1 << 30

    /// The bytes of one binary mebibyte.
    private static let bytesPerMebibyte = 1 << 20

    /// The working set an M5 Ultra with 96 GiB of unified memory recommends: 77.8 GiB.
    private static let largeWorkingSetBytes = 83_536_756_736

    /// The bytes a Qwen3.8-27B MTP decode step at 100k tokens of context frees and allocates
    /// again, which a smaller pool hands back to the system on every step: 2 GiB.
    private static let longContextStepBytes = 2 * bytesPerGibibyte

    /// The working set of a phone-class GPU: 5.5 GiB.
    private static let smallWorkingSetBytes = 5_905_580_032

    /// The pool of a GPU that recommends a small working set, or none: 256 MiB.
    private static let smallestPoolBytes = 256 * bytesPerMebibyte

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("A large GPU pools the buffers of a long-context decode step")
    func largeWorkingSetPoolsALongContextStep() {
        #expect(
            MLXLanguageModel.gpuCacheLimitBytes(workingSetBytes: Self.largeWorkingSetBytes)
                >= Self.longContextStepBytes)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("A small GPU keeps the smallest pool")
    func smallWorkingSetKeepsTheSmallestPool() {
        #expect(
            MLXLanguageModel.gpuCacheLimitBytes(workingSetBytes: Self.smallWorkingSetBytes)
                == Self.smallestPoolBytes)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("A GPU that recommends no working set keeps the smallest pool")
    func unknownWorkingSetKeepsTheSmallestPool() {
        #expect(MLXLanguageModel.gpuCacheLimitBytes(workingSetBytes: nil) == Self.smallestPoolBytes)
    }
}

#endif
