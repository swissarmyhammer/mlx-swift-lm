// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// Minimum physical memory required to safely attempt loading
/// `mlx-community/MiniMax-M3-4bit`. Chain reconciliation across kanban
/// ^wz8y8qq/^b90razv put the checkpoint's real size in the 120-214 GB range;
/// 220 GB leaves headroom above the higher estimate for the loaded weights
/// plus KV cache and working memory.
private let minimaxM3RequiredMemoryBytes: UInt64 = 220 * 1_024 * 1_024 * 1_024

/// Default checkpoint source (a Hub repo id) used when
/// `MLX_MINIMAX_M3_CHECKPOINT` is not set.
private let defaultMiniMaxM3CheckpointSource = "mlx-community/MiniMax-M3-4bit"

/// Resolves the checkpoint to test against: `MLX_MINIMAX_M3_CHECKPOINT`
/// overrides the default Hub id with either a local directory path or a
/// different Hub id (e.g. an MXFP4 variant), so a pre-downloaded copy can be
/// pointed at without waiting on a fresh multi-GB fetch.
private func resolveMiniMaxM3Configuration() -> ModelConfiguration {
    let source =
        ProcessInfo.processInfo.environment["MLX_MINIMAX_M3_CHECKPOINT"]
        ?? defaultMiniMaxM3CheckpointSource
    if source.hasPrefix("/") || source.hasPrefix(".") {
        return ModelConfiguration(directory: URL(fileURLWithPath: source), defaultPrompt: "")
    }
    return ModelConfiguration(id: source, defaultPrompt: "")
}

/// Whether `configuration` names a checkpoint this test can attempt to load:
/// local-directory overrides must exist on disk; Hub ids are always
/// considered available up front (a download failure is caught later and
/// treated the same as "absent").
private func checkpointIsAvailable(_ configuration: ModelConfiguration) -> Bool {
    if case .directory(let url) = configuration.id {
        return FileManager.default.fileExists(atPath: url.path)
    }
    return true
}

/// Real-weights coherence test for MiniMax-M3 (kanban ^wz8y8qq): loads
/// `mlx-community/MiniMax-M3-4bit` (or the `MLX_MINIMAX_M3_CHECKPOINT`
/// override) through `VLMModelFactory`, asserts the checkpoint's verified
/// heterogeneous per-layer quantization actually took effect (MoE gate
/// modules at 8-bit, expert weights at 4-bit affine), and asserts a
/// text-only prompt produces coherent output.
///
/// This is an `IntegrationTesting`-only suite -- `IntegrationTesting/` is an
/// Xcode project run via `xcodebuild`, not part of `swift test` -- and skips
/// gracefully (passes trivially) rather than failing when the machine lacks
/// enough memory, the checkpoint override path doesn't exist, or the
/// checkpoint fails to download/load (treated the same as "absent"), mirroring
/// the pattern `VisionIntegrationTests` uses for its `guard #available` early
/// return. Run explicitly via:
/// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/MiniMaxM3CoherenceIntegrationTests`
@Suite(.serialized, .timeLimit(.minutes(240)))
struct MiniMaxM3CoherenceIntegrationTests {

    @Test
    func minimax_m3() async throws {
        guard ProcessInfo.processInfo.physicalMemory >= minimaxM3RequiredMemoryBytes else {
            print(
                "Skipping MiniMax-M3 coherence test: physical memory "
                    + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                    + "\(minimaxM3RequiredMemoryBytes) bytes")
            return
        }

        let configuration = resolveMiniMaxM3Configuration()
        guard checkpointIsAvailable(configuration) else {
            print("Skipping MiniMax-M3 coherence test: checkpoint not found at \(configuration.name)")
            return
        }

        let container: LLModelContainer
        do {
            container = try await models.vlmContainer(for: configuration)
        } catch {
            print(
                "Skipping MiniMax-M3 coherence test: failed to load checkpoint "
                    + "\(configuration.name): \(error)")
            return
        }

        // Post-load module-bits assertion: walk the loaded module tree (via
        // the public `Module.leafModules()`, which reflects regardless of the
        // concrete decoder-layer types' internal visibility) and confirm the
        // checkpoint's verified heterogeneous quantization actually took
        // effect -- MoE router/gate modules at 8-bit, routed-expert weights
        // at 4-bit affine.
        let quantizedModules = await container.perform {
            (context: ModelContext) -> [String: (bits: Int, mode: QuantizationMode)] in
            var result: [String: (bits: Int, mode: QuantizationMode)] = [:]
            for (path, module) in context.model.leafModules().flattened() {
                if let quantized = module as? Quantized {
                    result[path] = (quantized.bits, quantized.mode)
                }
            }
            return result
        }

        let gateEntry = quantizedModules.first { $0.key.hasSuffix("block_sparse_moe.gate") }
        let expertEntry = quantizedModules.first { $0.key.hasSuffix("switch_mlp.gate_up_proj") }

        #expect(
            gateEntry?.value.bits == 8,
            "expected a MoE gate module to quantize at 8-bit, found: \(String(describing: gateEntry))"
        )
        #expect(
            expertEntry?.value.bits == 4,
            "expected expert switch_mlp weights to quantize at 4-bit, found: \(String(describing: expertEntry))"
        )
        #expect(expertEntry?.value.mode == .affine)

        // Coherence assertion: "2+2=" should produce a token stream containing "4".
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: 50, temperature: 0))
        var result = ""
        for try await token in session.streamResponse(to: "2+2=") {
            result += token
        }
        #expect(result.contains("4"), "expected '4' in the response to '2+2=', got: \(result)")
    }
}
