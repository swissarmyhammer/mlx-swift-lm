// Copyright © 2026 Apple Inc.
//
// The one shared load of the `mlx-community/DeepSeek-V4-Flash-4bit` checkpoint,
// and the gates that decide whether a test may use it.
//
// The checkpoint holds 141 GiB of weights, thus a test process must load it at
// most once. `DeepseekV4IntegrationTests` and
// `DeepseekV4AgenticPromptCacheAssessmentTests` both await the same load task,
// which lives here so neither suite owns it.
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`, thus `swift build --build-tests` stays at exit 0 with
// a type error in this file. Use
// `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
// as the compile evidence for any change to it.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Gating constants

/// The Hub repository of the checkpoint under test.
let deepseekV4RepositoryID = "mlx-community/DeepSeek-V4-Flash-4bit"

/// The environment key that points a suite at a pre-downloaded checkpoint
/// directory, in place of the Hugging Face cache locations.
let deepseekV4CheckpointOverrideKey = "MLX_DEEPSEEK_V4_CHECKPOINT"

/// The size of the weight files of the published checkpoint, measured on the
/// local snapshot on 2026-08-13: 151,482,475,612 bytes, which is 141 GiB. The
/// routed experts (`ffn.switch_mlp`) hold 137 GiB of that, which is 97%.
let deepseekV4CheckpointBytes = 151_482_475_612

/// The least physical memory a run needs: the 141 GiB of 4-bit weights, plus
/// the MoE working set, the KV cache of a 12k-token generation, and system
/// headroom.
///
/// This is the SKIP GATE, and it is not the wired-memory limit. It answers
/// "is this machine large enough to run at all". The wired limit answers "how
/// much memory must stay resident", and `MLXLMCommon.ModelWeightResidency`
/// owns it, thus a change to one never moves the other.
let deepseekV4RequiredMemoryBytes: UInt64 = 160 * 1_024 * 1_024 * 1_024

// MARK: - Checkpoint location

/// Finds a complete local copy of the checkpoint, and never downloads one.
///
/// The search order is: the ``deepseekV4CheckpointOverrideKey`` directory,
/// the `huggingface_hub` cache snapshot, then the swift-transformers download
/// base. A directory counts only when it holds at least one `*.safetensors`
/// file, because a tokenizer-only snapshot cannot feed a weight load.
///
/// - Returns: the checkpoint directory, or `nil` when no complete local copy
///   exists.
private func localDeepseekV4CheckpointDirectory() -> URL? {
    var candidates: [URL] = []
    if let override = ProcessInfo.processInfo.environment[deepseekV4CheckpointOverrideKey] {
        candidates.append(URL(fileURLWithPath: override, isDirectory: true))
    }
    if let snapshot = hfSnapshotDir(modelId: deepseekV4RepositoryID) {
        candidates.append(snapshot)
    }
    candidates.append(
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Documents/huggingface/models", isDirectory: true)
            .appendingPathComponent(deepseekV4RepositoryID, isDirectory: true))
    return candidates.first(where: directoryHoldsSafetensors)
}

/// Tells whether `directory` holds at least one `*.safetensors` file.
private func directoryHoldsSafetensors(_ directory: URL) -> Bool {
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
    else { return false }
    return entries.contains { $0.pathExtension == "safetensors" }
}

// MARK: - The Metal wired limit

/// Reports the Metal wired limit that the weight load of `MLXLMCommon` raised.
///
/// The library owns the raise. `MLXLMCommon.loadWeights` asks
/// ``MLXLMCommon/ModelWeightResidency`` to cover the weight files before it
/// reads the first one, because a buffer joins the Metal residency set when it
/// is made. This function only reads the outcome of that raise, thus it holds
/// no wired-limit logic of its own.
///
/// Measured on an M3 Ultra (512 GiB) with this checkpoint, card `^3gh7rb5`.
/// With a release build and a direct model call, one decode step takes 2.10 s
/// with the default limit and 0.068 s with the limit raised, which is 31 times
/// faster. Through the integration suite, which builds for debug and decodes
/// through `TokenIterator`, the same two steps take 2.124 s and 0.593 s. The
/// routed experts hold 137 GiB of the 141 GiB total and carry 2.04 s of the
/// 2.10 s.
///
/// - Returns: what the load asked for and what the manager applied, or `nil`
///   when this device has no wired-memory control.
private func wiredMemoryOutcomeOfTheLoad() async -> WiredMemoryOutcome? {
    let outcome = await ModelWeightResidency.shared.outcome
    guard let outcome else {
        print(
            "DeepSeek-V4 wired limit not raised: this device has no wired-memory control, "
                + "thus each decode step pays for the whole checkpoint")
        return nil
    }
    print(
        "DeepSeek-V4 wired limit: asked \(outcome.requestedBytes) bytes, "
            + "applied \(outcome.appliedBytes) bytes")
    if !outcome.isFullyApplied {
        print(
            "DeepSeek-V4 wired limit is short of the request, thus weight buffers stay "
                + "outside the Metal residency set and each decode step stays slow")
    }
    return outcome
}

// MARK: - One shared load

/// What one shared load produced: the container, and the wired-limit outcome
/// of the same process.
struct DeepseekV4LoadResult: Sendable {
    /// The loaded model container.
    let container: LLModelContainer
    /// The wired-limit outcome, or `nil` when the limit was not raised.
    let wiredMemory: WiredMemoryOutcome?
}

/// One shared load of the checkpoint. Each test awaits the same load task,
/// thus the 141 GiB weight load runs at most once per test process.
///
/// The task is `nil` when no complete local checkpoint exists, and the task
/// itself throws when the load fails.
enum DeepseekV4Load {
    /// The shared load task, or `nil` when the checkpoint is absent.
    static let shared: Task<DeepseekV4LoadResult, Error>? = {
        guard let directory = localDeepseekV4CheckpointDirectory() else { return nil }
        return Task {
            print("Loading DeepSeek-V4 from \(directory.path)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
            print("Loaded DeepSeek-V4")
            // The load itself raised the limit, before it made the first weight
            // buffer. This reads what it asked for and what it applied.
            let wiredMemory = await wiredMemoryOutcomeOfTheLoad()
            return DeepseekV4LoadResult(container: container, wiredMemory: wiredMemory)
        }
    }()
}

// MARK: - Gates

/// Loads the shared load result, or prints a skip message and returns `nil`.
///
/// The gates are: enough physical memory, a complete local checkpoint, and a
/// load that completes.
///
/// - Parameter testName: the test to name in a skip message.
/// - Returns: the shared load result, or `nil` when a gate closed.
func deepseekV4LoadResultOrSkip(testName: String) async -> DeepseekV4LoadResult? {
    guard ProcessInfo.processInfo.physicalMemory >= deepseekV4RequiredMemoryBytes else {
        print(
            "Skipping \(testName): physical memory "
                + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                + "\(deepseekV4RequiredMemoryBytes) bytes")
        return nil
    }
    guard let load = DeepseekV4Load.shared else {
        print(
            "Skipping \(testName): no local copy of \(deepseekV4RepositoryID) "
                + "with safetensors files. Download the checkpoint, or set "
                + "\(deepseekV4CheckpointOverrideKey) to a checkpoint directory. "
                + "This test never downloads the \(deepseekV4CheckpointBytes)-byte "
                + "checkpoint itself.")
        return nil
    }
    do {
        return try await load.value
    } catch {
        print("Skipping \(testName): failed to load \(deepseekV4RepositoryID): \(error)")
        return nil
    }
}

/// Loads the shared container, or prints a skip message and returns `nil`.
///
/// The gates are the gates of ``deepseekV4LoadResultOrSkip(testName:)``.
///
/// - Parameter testName: the test to name in a skip message.
/// - Returns: the shared container, or `nil` when a gate closed.
func deepseekV4ContainerOrSkip(testName: String) async -> LLModelContainer? {
    await deepseekV4LoadResultOrSkip(testName: testName)?.container
}
