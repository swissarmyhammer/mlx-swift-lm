import Foundation
import MLX
import MLXLMCommon
import XCTest

/// The exact contents of an array: its shape, its dtype and a copy of its bytes.
///
/// Two snapshots are equal only when the arrays have the same bits, so the compiled-decode
/// suites compare compiled and eager results with it.
struct ArraySnapshot: Equatable {
    let shape: [Int]
    let dtype: DType
    let bytes: Data

    /// Copies the contents of `array`.
    ///
    /// - Parameter array: The array to record.
    init(_ array: MLXArray) {
        let contents = array.asData(access: .copy)
        self.shape = contents.shape
        self.dtype = contents.dType
        self.bytes = contents.data
    }
}

/// What one decode run of a model leaves behind: the output of each step, the inner state of
/// each cache and the offset of each cache.
struct CompiledDecodeRun {
    /// The output of each forward call, in order.
    let outputs: [ArraySnapshot]

    /// The inner state of each cache of the run, in layer order.
    let cacheState: [ArraySnapshot]

    /// The offset of each cache of the run. `-1` stands for a layer without a cache.
    let offsets: [Int]

    /// Records the outputs of a run and the final state of its caches.
    ///
    /// - Parameters:
    ///   - outputs: The output of each forward call, in order.
    ///   - cache: The caches the run wrote. `nil` stands for a layer without a cache.
    init(outputs: [ArraySnapshot], cache: [KVCache?]) {
        self.outputs = outputs
        self.cacheState = cache.compactMap { $0 }.flatMap { $0.innerState() }
            .map(ArraySnapshot.init)
        self.offsets = cache.map { $0?.offset ?? -1 }
    }

    /// Runs `forward` on each input in order, evaluates and records each output, then records
    /// the caches.
    ///
    /// - Parameters:
    ///   - inputs: The inputs of the forward calls, in order.
    ///   - cache: The caches that `forward` writes.
    ///   - forward: One forward call of the model over the caches.
    /// - Returns: The outputs and the final state of the caches.
    /// - Throws: The error of `forward`.
    static func decoding<Input>(
        _ inputs: [Input],
        cache: [KVCache?],
        forward: (Input) throws -> MLXArray
    ) rethrows -> CompiledDecodeRun {
        let outputs = try inputs.map { input in
            let output = try forward(input)
            eval(output)
            return ArraySnapshot(output)
        }
        return CompiledDecodeRun(outputs: outputs, cache: cache)
    }

    /// Runs one decode eagerly and one compiled, and asserts that both give the same bits.
    ///
    /// - Parameters:
    ///   - context: The text that the failure messages name, or `nil`.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    ///   - run: One decode run. Its argument is true for the compiled run.
    /// - Returns: The compiled run.
    /// - Throws: The error of `run`.
    @discardableResult
    static func assertCompiledMatchesEager(
        _ context: String? = nil,
        file: StaticString = #filePath,
        line: UInt = #line,
        run: (_ compiled: Bool) throws -> CompiledDecodeRun
    ) rethrows -> CompiledDecodeRun {
        let eager = try run(false)
        let compiled = try run(true)
        compiled.assertMatches(eager, context, file: file, line: line)
        return compiled
    }

    /// Asserts that this run has the outputs, the cache state and the offsets of `reference`.
    ///
    /// - Parameters:
    ///   - reference: The run to compare with.
    ///   - context: The text that the failure messages name, or `nil`.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    func assertMatches(
        _ reference: CompiledDecodeRun,
        _ context: String? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let suffix = context.map { " for \($0)" } ?? ""
        XCTAssertEqual(
            outputs, reference.outputs, "outputs differ\(suffix)", file: file, line: line)
        XCTAssertEqual(
            cacheState, reference.cacheState, "cache state differs\(suffix)",
            file: file, line: line)
        XCTAssertEqual(
            offsets, reference.offsets, "cache offsets differ\(suffix)", file: file, line: line)
    }
}
