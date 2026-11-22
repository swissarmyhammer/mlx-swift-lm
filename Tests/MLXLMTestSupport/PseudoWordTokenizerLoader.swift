// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXScriptedLM

/// A tokenizer loader that gives a ``PseudoWordTokenizer`` for any directory, for a factory load
/// of a checkpoint that ships no tokenizer files.
public struct PseudoWordTokenizerLoader: TokenizerLoader {

    /// Makes the loader.
    public init() {}

    /// Gives a new ``PseudoWordTokenizer``.
    ///
    /// - Parameter directory: the directory of the checkpoint, which the loader does not read.
    /// - Returns: a ``PseudoWordTokenizer``.
    public func load(from directory: URL) async throws -> any Tokenizer {
        PseudoWordTokenizer()
    }
}
