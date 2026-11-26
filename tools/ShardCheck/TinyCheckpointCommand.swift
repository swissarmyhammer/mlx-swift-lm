// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXLMTestSupport

/// A command of ``ShardCheck`` that writes tiny checkpoints instead of running a check, so
/// `scripts/shard-check.sh --tiny` checks every model family that shards with no download.
///
/// Usage: `ShardCheck tiny-families`, or `ShardCheck tiny-checkpoint <family> <directory>`.
enum TinyCheckpointCommand {

    /// The arguments of `tiny-checkpoint` after its name: the family and the directory.
    private static let checkpointOperandCount = 2

    /// Prints each model type of ``RemoteModelExecutor/shardableModelFamilies``, one per line.
    case families
    /// Writes the tiny checkpoint of `family` to `directory`, then prints the prompt ids its
    /// vocabulary holds, comma-separated, for `--tokens`.
    case checkpoint(family: String, directory: URL)

    /// The command that the command line, without the program name, names; nil when its first
    /// argument names no command, so the line runs a check.
    ///
    /// - Throws: ``ShardCheckError/usage`` when a command has the wrong arguments.
    init?(_ arguments: ArraySlice<String>) throws {
        let operands = arguments.dropFirst()
        switch arguments.first {
        case "tiny-families":
            guard operands.isEmpty else { throw ShardCheckError.usage }
            self = .families
        case "tiny-checkpoint":
            guard operands.count == Self.checkpointOperandCount,
                let family = operands.first, let directory = operands.last
            else { throw ShardCheckError.usage }
            self = .checkpoint(family: family, directory: URL(fileURLWithPath: directory))
        default:
            return nil
        }
    }

    /// Runs the command.
    func run() async throws {
        switch self {
        case .families:
            for family in RemoteModelExecutor.shardableModelFamilies {
                Console.write(family)
            }
        case .checkpoint(let family, let directory):
            try await TinyShardCheckpoint.write(family: family, to: directory)
            Console.write(TinyShardCheckpoint.prompt.map(String.init).joined(separator: ","))
        }
    }
}
