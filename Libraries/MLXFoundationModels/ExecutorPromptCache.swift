// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import os

// MARK: - Identity

/// Names the one session whose turns share a prompt cache.
struct ExecutorPromptCacheKey: Hashable, Sendable {

    /// The model the cache belongs to, thus two models never share one cache.
    let modelID: String

    /// The session the cache belongs to.
    let sessionID: String
}

// MARK: - What a session carries

/// The live key/value cache of one session, and the tokens it holds.
///
/// This is a reference type because the caches it carries are reference types
/// that generation writes into, and the model state beside them is not
/// `Sendable`. ``ExecutorPromptCacheStore`` hands one entry to at most one turn
/// at a time -- a check-out REMOVES the entry from the store -- thus no two
/// tasks write the same caches or read the same state, and the
/// `@unchecked Sendable` conformance rests on that.
final class ExecutorPromptCacheEntry: @unchecked Sendable {

    /// One cache for each layer of the model.
    let caches: [KVCache]

    /// The exact prompt tokens `caches` represents, in order.
    let tokens: [Int]

    /// The whole prompt the last pass rendered, which is not the same as
    /// `tokens`: the ledger also holds the tokens the model generated after
    /// that render.
    ///
    /// A protocol rule compares this with the next render to prove that the
    /// template rewrote no cached region before it splices past the turn the
    /// model committed. Empty means no render is on record, thus no rule may
    /// splice.
    let renderTokens: [Int]

    /// The model state the last pass left with `caches`, or nil for a model
    /// that carries none.
    ///
    /// A Qwen VL model keys its M-RoPE anchor here and refuses a warm cache
    /// without it, thus the next turn seeds its iterator with this state. The
    /// state is tied to the prefill that made it, thus the caches never rewind
    /// under it.
    let state: LMOutput.State?

    /// The bytes this entry holds in memory: the resident bytes of every
    /// cache, plus the arrays of `state`.
    ///
    /// The count reads only shapes and element types, thus it evaluates
    /// nothing. It is computed one time, when the entry is made, because the
    /// store reads it at each check-in and each eviction.
    let byteCount: Int

    /// Creates an entry for caches that hold `tokens`.
    ///
    /// - Parameters:
    ///   - caches: one cache for each layer of the model.
    ///   - tokens: the exact tokens `caches` represents.
    ///   - renderTokens: the whole prompt the last pass rendered, or empty
    ///     when no render is on record.
    ///   - state: the model state the last pass left, or nil.
    init(
        caches: [KVCache], tokens: [Int], renderTokens: [Int] = [],
        state: LMOutput.State? = nil
    ) {
        self.caches = caches
        self.tokens = tokens
        self.renderTokens = renderTokens
        self.state = state
        self.byteCount =
            caches.reduce(0) { $0 + $1.residentByteCount } + (state?.residentByteCount ?? 0)
    }
}

// MARK: - What a check-out gives

/// The file of one entry that the store wrote to disk, and the session it
/// belongs to.
///
/// The turn that gets a handle owns the file: the store keeps no record of it.
struct ExecutorPromptCacheSpilledHandle: Equatable, Sendable {

    /// The URL of the file.
    let url: URL

    /// The session the file belongs to.
    let key: ExecutorPromptCacheKey
}

/// What ``ExecutorPromptCacheStore/checkOut(_:)`` finds for one session.
enum ExecutorPromptCacheCheckout: Sendable {

    /// The entry was in memory, or its write to disk had not ended.
    case memory(ExecutorPromptCacheEntry)

    /// The entry is on disk, in the file of the handle.
    case spilled(ExecutorPromptCacheSpilledHandle)

    /// The store holds nothing for the session.
    case none

    /// The entry in memory, or nil when the check-out found no entry in
    /// memory.
    var entry: ExecutorPromptCacheEntry? {
        switch self {
        case .memory(let entry): entry
        case .spilled, .none: nil
        }
    }

    /// The file on disk, or nil when the check-out found no file.
    var spilledHandle: ExecutorPromptCacheSpilledHandle? {
        switch self {
        case .spilled(let handle): handle
        case .memory, .none: nil
        }
    }
}

extension ExecutorPromptCacheCheckout: Equatable {

    /// Two check-outs are equal when they give the same entry object, equal
    /// handles, or both give nothing.
    static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case (.memory(let lhsEntry), .memory(let rhsEntry)): lhsEntry === rhsEntry
        case (.spilled(let lhsHandle), .spilled(let rhsHandle)): lhsHandle == rhsHandle
        case (.none, .none): true
        case (.memory, _), (.spilled, _), (.none, _): false
        }
    }
}

// MARK: - The spool writer

/// Writes a prepared entry to a URL. The spool calls it off the store actor,
/// one write at a time.
typealias ExecutorPromptCacheFileWriter = @Sendable (PromptCacheSaveInput, URL) throws -> Void

/// One write of the spool, and what to do when it ends.
struct ExecutorPromptCacheSpillJob: Sendable {

    /// The prepared entry.
    let input: PromptCacheSaveInput

    /// The URL of the file.
    let url: URL

    /// Receives the result of the write and the time it took.
    let didEnd: @Sendable (Result<Void, any Error>, Duration) async -> Void
}

/// Runs the writes of one store in order, one at a time, on a task of its own.
///
/// A write holds the process-wide evaluation lock of MLX for its whole run,
/// because the safetensors save evaluates the arrays. Two writes at the same
/// time thus gain nothing, and one write at a time keeps the store actor free:
/// the actor only puts a job in the queue.
final class ExecutorPromptCacheSpoolWriter: Sendable {

    /// The queue of the writes.
    private let jobs: AsyncStream<ExecutorPromptCacheSpillJob>.Continuation

    /// The task that runs the writes.
    private let consumer: Task<Void, Never>

    /// Creates a writer and starts the task that runs its writes.
    ///
    /// - Parameter write: writes one prepared entry to one URL.
    init(write: @escaping ExecutorPromptCacheFileWriter) {
        let (stream, jobs) = AsyncStream.makeStream(of: ExecutorPromptCacheSpillJob.self)
        self.jobs = jobs
        self.consumer = Task.detached {
            for await job in stream {
                await Self.run(job, write: write)
            }
        }
    }

    deinit {
        jobs.finish()
        consumer.cancel()
    }

    /// Puts `job` at the end of the queue.
    ///
    /// - Parameter job: the write to run.
    func enqueue(_ job: ExecutorPromptCacheSpillJob) {
        jobs.yield(job)
    }

    /// Writes the file of `job`, and gives the result and the duration to the
    /// job.
    ///
    /// - Parameters:
    ///   - job: the write to run.
    ///   - write: writes one prepared entry to one URL.
    private static func run(
        _ job: ExecutorPromptCacheSpillJob, write: ExecutorPromptCacheFileWriter
    ) async {
        let clock = ContinuousClock()
        let start = clock.now
        let result = Result<Void, any Error> { try write(job.input, job.url) }
        await job.didEnd(result, start.duration(to: clock.now))
    }
}

// MARK: - The store

/// Holds the prompt cache of each live session between the turns of that
/// session.
///
/// A turn CHECKS OUT its entry, which removes the entry from the store, and
/// checks an entry back in when the turn ends. Two turns of one session thus
/// never write the same caches at the same time: a second turn that starts
/// while the first still runs finds nothing and starts cold.
///
/// Each entry holds the whole key/value state of one conversation, which is
/// large, and the framework never tells this executor that a session ended.
/// The store thus limits the bytes it holds in memory to
/// ``memoryBudgetBytes``. A count of sessions is the wrong unit, because one
/// long session can be larger than many short ones. When a check-in takes the
/// total past the budget, the least recently used entries leave first, each
/// through ``spill(_:_:)``.
///
/// An entry that leaves memory goes to one file in the spool folder of the
/// store. It has three
/// states: in memory, spilling (its write has not ended) and on disk. Each
/// spill gets a new generation number, and the file name holds it. A write
/// that ends changes the records of its key only when its generation is still
/// the generation of that key. An older write only deletes its own file.
actor ExecutorPromptCacheStore {

    /// The store every executor shares.
    static let shared = ExecutorPromptCacheStore()

    /// The name of the folder, in the temporary directory, that holds the spool
    /// folder of each process.
    static let spoolFolderName = "mlx-prompt-cache"

    /// The spool folder of this process:
    /// `<temporary directory>/mlx-prompt-cache/<pid>-<UUID>/`. The UUID is made
    /// one time in each process, thus a new process never reads the folder of
    /// an old process that had the same process identifier.
    static let processSpoolDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent(spoolFolderName, isDirectory: true)
        .appendingPathComponent(
            "\(ProcessInfo.processInfo.processIdentifier)-\(UUID().uuidString)",
            isDirectory: true)

    /// The store the executor uses: ``shared``, unless a task binds another.
    ///
    /// Swift Testing runs suites in parallel, thus a test binds its own store
    /// with `$current.withValue(store) { ... }` and never shares entries or a
    /// budget with another test.
    @TaskLocal static var current: ExecutorPromptCacheStore = .shared

    /// The part of the free working set the default budget takes: one
    /// quarter. The rest stays free for the weights and the live pass.
    static let defaultBudgetDivisor = 4

    /// The most bytes the checked-in entries may hold in memory.
    ///
    /// The host sets it with ``configure(memoryBudgetBytes:)``. When no host
    /// sets one, the store reads the device at its first use and takes
    /// ``defaultMemoryBudgetBytes(workingSet:active:)``.
    private(set) lazy var memoryBudgetBytes: Int = Self.defaultMemoryBudgetBytes(
        workingSet: Int(clamping: GPU.deviceInfo().maxRecommendedWorkingSetSize),
        active: Memory.activeMemory)

    /// The checked-in entries, by session.
    private var entries: [ExecutorPromptCacheKey: ExecutorPromptCacheEntry] = [:]

    /// The checked-in keys, least recently used first.
    private var usageOrder: [ExecutorPromptCacheKey] = []

    /// The sum of ``ExecutorPromptCacheEntry/byteCount`` over the checked-in
    /// entries.
    private(set) var retainedByteCount = 0

    /// How many sessions hold a cache. Read by tests.
    var retainedSessionCount: Int { entries.count }

    /// The folder of the spill files. The spool writer makes it at the first
    /// write.
    private let directory: URL

    /// Runs the writes of the spill files, one at a time.
    private let spoolWriter: ExecutorPromptCacheSpoolWriter

    /// One entry whose write to disk has not ended.
    private struct SpillingEntry {

        /// The entry. It is still in memory, thus a check-out gives it back.
        let entry: ExecutorPromptCacheEntry

        /// The input the writer writes. It holds its own array handles, thus a
        /// turn that takes `entry` back and writes into its caches does not
        /// change the file.
        let input: PromptCacheSaveInput

        /// The generation of this spill.
        let generation: UInt64

        /// The bytes the entry holds in memory until the write ends.
        let byteCount: Int
    }

    /// The entries whose write to disk has not ended, by session.
    private var spilling: [ExecutorPromptCacheKey: SpillingEntry] = [:]

    /// The sum of the bytes of the entries in ``spilling``. They are still in
    /// memory until their writes end, but they are not in
    /// ``retainedByteCount``.
    private(set) var spillingByteCount = 0

    /// The files of the entries on disk, by session.
    private var onDisk: [ExecutorPromptCacheKey: ExecutorPromptCacheSpilledHandle] = [:]

    /// The generation of the last spill. Each spill takes the next number.
    private var lastGeneration: UInt64 = 0

    /// The writes that are in the queue or that run now.
    private var pendingWriteCount = 0

    /// The tasks that wait in ``waitForSpills()``.
    private var spillWaiters: [CheckedContinuation<Void, Never>] = []

    /// Creates a store that spills to the folder of this process,
    /// ``processSpoolDirectory``.
    init() {
        self.init(directory: Self.processSpoolDirectory)
    }

    /// Creates a store that spills to `directory`.
    ///
    /// - Parameters:
    ///   - directory: the folder of the spill files. The default writer makes
    ///     it at the first write.
    ///   - writer: writes one prepared entry to one file. Tests give a writer
    ///     that holds or counts the writes.
    init(
        directory: URL,
        writer: @escaping ExecutorPromptCacheFileWriter = ExecutorPromptCacheStore.writeSpillFile
    ) {
        self.directory = directory
        self.spoolWriter = ExecutorPromptCacheSpoolWriter(write: writer)
    }

    /// Makes the folder of `url`, when it is not there, and writes `input` to
    /// `url`. This is the default writer of a store.
    ///
    /// - Parameters:
    ///   - input: the prepared entry.
    ///   - url: the URL of the spill file.
    /// - Throws: the error of the file system, or the error of
    ///   ``ExecutorPromptCacheFile/write(_:to:)``.
    static func writeSpillFile(_ input: PromptCacheSaveInput, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ExecutorPromptCacheFile.write(input, to: url)
    }

    /// The budget the store takes when no host sets one: one quarter of the
    /// working set that is not active yet.
    ///
    /// - Parameters:
    ///   - workingSet: the recommended maximum working set of the device, in
    ///     bytes.
    ///   - active: the bytes MLX holds active now.
    /// - Returns: the budget in bytes, or zero when `active` is larger than
    ///   `workingSet`.
    static func defaultMemoryBudgetBytes(workingSet: Int, active: Int) -> Int {
        let (free, overflow) = workingSet.subtractingReportingOverflow(active)
        guard !overflow else { return 0 }
        return max(0, free) / defaultBudgetDivisor
    }

    /// Sets the byte budget and applies it at once: the least recently used
    /// entries leave until the total is at or below `budget`. A larger budget
    /// removes nothing.
    ///
    /// - Parameter budget: the most bytes the entries may hold. Zero or less
    ///   keeps no entry in memory.
    func configure(memoryBudgetBytes budget: Int) {
        memoryBudgetBytes = budget
        evictToBudget()
    }

    /// Takes the entry of `key` out of the store.
    ///
    /// The store looks in memory first, then in the spills that have not
    /// ended, then on disk. An entry whose write has not ended comes back as
    /// `.memory`: the write then records nothing and deletes its own file. A
    /// file on disk comes back as `.spilled`, and the caller then owns it.
    ///
    /// - Parameter key: the session of the turn.
    /// - Returns: the entry, its file, or `.none` when the store holds nothing
    ///   for `key`.
    func checkOut(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheCheckout {
        if let entry = remove(key) {
            return .memory(entry)
        }
        if let spill = removeSpill(of: key) {
            return .memory(spill.entry)
        }
        if let handle = onDisk.removeValue(forKey: key) {
            return .spilled(handle)
        }
        return .none
    }

    /// Returns when every write in the queue has ended. Tests call it before
    /// they read the spool folder.
    func waitForSpills() async {
        guard pendingWriteCount > 0 else { return }
        await withCheckedContinuation { spillWaiters.append($0) }
    }

    /// Reads the entry of `key` and leaves it in the store. Read by tests that
    /// compare a session's ledger with its next render.
    ///
    /// - Returns: the entry, or nil when the store holds none for `key`.
    func peek(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheEntry? {
        entries[key]
    }

    /// Puts `entry` back under `key` as the most recently used entry, or
    /// drops the key when `entry` is nil.
    ///
    /// A check-in is newer than any spill of `key`, thus the store first drops
    /// the spill that has not ended and deletes the file on disk of `key`.
    ///
    /// After the insert, the least recently used entries leave until the
    /// total is at or below ``memoryBudgetBytes``. An entry larger than the
    /// whole budget is not kept in memory, and no other entry leaves for it.
    ///
    /// - Parameters:
    ///   - key: the session of the turn.
    ///   - entry: the entry the turn leaves, or nil when the next turn must
    ///     start cold.
    func checkIn(_ key: ExecutorPromptCacheKey, _ entry: ExecutorPromptCacheEntry?) {
        remove(key)
        removeSpill(of: key)
        if let handle = onDisk.removeValue(forKey: key) {
            ExecutorPromptCacheFile.removeFile(at: handle.url)
        }
        guard let entry else { return }
        guard entry.byteCount <= memoryBudgetBytes else {
            spill(key, entry)
            return
        }
        entries[key] = entry
        usageOrder.append(key)
        retainedByteCount += entry.byteCount
        evictToBudget()
    }

    /// Releases the cache of every session of `modelID`, or of every session
    /// when `modelID` is nil.
    ///
    /// This is a release that the host asked for, not an eviction for the
    /// budget, thus the entries do not go through ``spill(_:_:)``.
    func evict(modelID: String?) {
        let released = entries.keys.filter { modelID == nil || $0.modelID == modelID }
        for key in released {
            remove(key)
        }
    }

    /// Removes the entry of `key` from the entries, the usage order and the
    /// byte total.
    ///
    /// - Returns: the entry, or nil when the store holds none for `key`.
    @discardableResult
    private func remove(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheEntry? {
        guard let entry = entries.removeValue(forKey: key) else { return nil }
        usageOrder.removeAll { $0 == key }
        retainedByteCount -= entry.byteCount
        return entry
    }

    /// Removes the least recently used entries until the total is at or below
    /// ``memoryBudgetBytes``, and gives each one to ``spill(_:_:)``.
    private func evictToBudget() {
        while retainedByteCount > memoryBudgetBytes, let oldest = usageOrder.first,
            let entry = remove(oldest)
        {
            spill(oldest, entry)
        }
    }

    /// Takes an entry that leaves memory for the budget. Every eviction for
    /// the budget comes through here.
    ///
    /// The entry is already out of the store. This function logs the eviction,
    /// prepares the entry on the actor, and puts its write in the queue of the
    /// spool writer. `prepare` copies the array handles and the metadata, thus
    /// the writer holds no reference to the caches: a turn that checks the
    /// entry out during the write can write into the caches, and the file does
    /// not change. The writer, not the actor, evaluates the arrays.
    ///
    /// An entry that cannot be prepared is dropped, thus the next turn of that
    /// session starts cold.
    ///
    /// - Parameters:
    ///   - key: the session of the entry.
    ///   - entry: the entry that leaves memory.
    private func spill(_ key: ExecutorPromptCacheKey, _ entry: ExecutorPromptCacheEntry) {
        ExecutorPromptCacheLog.info(
            ExecutorPromptCacheReport.evictionLine(key: key, byteCount: entry.byteCount))
        let input: PromptCacheSaveInput
        do {
            input = try ExecutorPromptCacheFile.prepare(entry, key: key)
        } catch {
            ExecutorPromptCacheLog.info(
                ExecutorPromptCacheReport.spillLine(
                    key: key, byteCount: entry.byteCount, duration: .zero,
                    outcome: .failed(error)))
            return
        }
        lastGeneration += 1
        let spill = SpillingEntry(
            entry: entry, input: input, generation: lastGeneration, byteCount: entry.byteCount)
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: spill.generation))
        spilling[key] = spill
        spillingByteCount += spill.byteCount
        pendingWriteCount += 1
        spoolWriter.enqueue(
            ExecutorPromptCacheSpillJob(input: spill.input, url: url) {
                [weak self] result, duration in
                await self?.writeDidEnd(
                    ExecutorPromptCacheSpilledHandle(url: url, key: key),
                    generation: spill.generation, byteCount: spill.byteCount,
                    result: result, duration: duration)
            })
    }

    /// Removes the spill of `key` that has not ended, and its bytes from
    /// ``spillingByteCount``. The write of that spill then finds another
    /// generation, or none, and only deletes its own file.
    ///
    /// - Parameter key: the session of the spill.
    /// - Returns: the spill, or nil when no write of `key` is in the queue.
    @discardableResult
    private func removeSpill(of key: ExecutorPromptCacheKey) -> SpillingEntry? {
        guard let spill = spilling.removeValue(forKey: key) else { return nil }
        spillingByteCount -= spill.byteCount
        return spill
    }

    /// Records the end of one write.
    ///
    /// When `generation` is still the generation of the key, the spill ends:
    /// a written file goes on disk, and a failed write leaves nothing. When a
    /// check-out or a check-in removed the spill, the write is stale: its file
    /// is deleted, and no record changes. A stale write never touches the
    /// file of a later spill, because each generation has its own file name.
    ///
    /// - Parameters:
    ///   - handle: the file of the write and its session.
    ///   - generation: the generation of the write.
    ///   - byteCount: the bytes of the entry, for the log line.
    ///   - result: the result of the write.
    ///   - duration: how long the write took.
    private func writeDidEnd(
        _ handle: ExecutorPromptCacheSpilledHandle, generation: UInt64, byteCount: Int,
        result: Result<Void, any Error>, duration: Duration
    ) {
        defer { finishPendingWrite() }
        let isCurrent = spilling[handle.key]?.generation == generation
        if isCurrent {
            removeSpill(of: handle.key)
        }
        let outcome: ExecutorPromptCacheSpillOutcome
        switch (result, isCurrent) {
        case (.success, true):
            onDisk[handle.key] = handle
            outcome = .onDisk
        case (.success, false):
            ExecutorPromptCacheFile.removeFile(at: handle.url)
            outcome = .superseded
        case (.failure(let error), _):
            outcome = .failed(error)
        }
        ExecutorPromptCacheLog.info(
            ExecutorPromptCacheReport.spillLine(
                key: handle.key, byteCount: byteCount, duration: duration, outcome: outcome))
    }

    /// Counts one write as ended, and wakes the tasks in ``waitForSpills()``
    /// when no write is left.
    private func finishPendingWrite() {
        pendingWriteCount -= 1
        guard pendingWriteCount == 0 else { return }
        let waiters = spillWaiters
        spillWaiters = []
        for waiter in waiters {
            waiter.resume()
        }
    }
}

/// What one write of the spool gave.
enum ExecutorPromptCacheSpillOutcome: CustomStringConvertible {

    /// The file is on disk, and a check-out hands it out.
    case onDisk

    /// A check-out or a check-in took the entry back during the write, thus
    /// the file was deleted.
    case superseded

    /// The entry could not be prepared or written, thus nothing is on disk.
    case failed(any Error)

    /// The result as the log line states it.
    var description: String {
        switch self {
        case .onDisk: "on disk"
        case .superseded: "superseded, file deleted"
        case .failed(let error): "failed: \(error)"
        }
    }
}

// MARK: - Planning one generation pass

/// What one generation pass does with the prompt cache of its session.
struct ExecutorPromptCachePlan {

    /// The caches to hand to generation.
    let caches: [KVCache]

    /// The input to feed, narrowed to the tokens `caches` does not already hold.
    let input: LMInput

    /// The leading prompt tokens `caches` already holds, thus the tokens this
    /// pass does not feed to the model.
    let reusedTokenCount: Int

    /// The whole rendered prompt of this pass.
    let promptTokens: [Int]

    /// The tokens `caches` represents once this pass has fed `input`, before
    /// generation. This is `promptTokens` on the standard path. A protocol rule
    /// that splices past a committed turn names the tokens the model wrote
    /// instead, plus the tail of the render it fed.
    let representedTokens: [Int]

    /// The model state to seed the iterator with: the state the entry carried
    /// when this pass reuses its caches, and nil when the pass builds fresh
    /// caches.
    let state: LMOutput.State?

    /// The rule that decided this plan, which the log line of the pass names.
    let decision: ExecutorPromptCacheDecision

    /// Plans what `entry` may serve for `input`, building fresh caches when it
    /// may serve nothing.
    ///
    /// - Parameters:
    ///   - entry: the cache the session carries, or nil for a cold session.
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    /// - Returns: the plan, or nil when a carried cache cannot serve this input
    ///   at all. Media, a mask that hides a token and a batch of more than one
    ///   row each place content in the model's input that a token ledger cannot
    ///   describe; the caller then generates with no carried cache.
    static func make(
        reusing entry: ExecutorPromptCacheEntry?,
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = []
    ) throws -> ExecutorPromptCachePlan? {
        guard let promptTokens = ledgerTokens(of: input), !promptTokens.isEmpty else {
            return nil
        }

        if let entry,
            let reuse = reconcilePromptCache(
                promptTokens: promptTokens, cachedTokens: entry.tokens,
                previousRenderTokens: entry.renderTokens, caches: entry.caches,
                protocolRules: protocolRules, carriesModelState: entry.state != nil),
            reuse.suffixStart < promptTokens.count
        {
            return ExecutorPromptCachePlan(
                caches: entry.caches,
                input: reuse.suffixStart == 0
                    ? input
                    : narrowed(input, to: Array(promptTokens[reuse.suffixStart...])),
                reusedTokenCount: reuse.suffixStart,
                promptTokens: promptTokens,
                representedTokens: reuse.representedTokens,
                state: entry.state,
                decision: decision(of: reuse, render: promptTokens, ledger: entry.tokens))
        }

        return ExecutorPromptCachePlan(
            caches: try model.newCache(parameters: parameters),
            input: input,
            reusedTokenCount: 0,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: nil,
            decision: entry.map { .rebuild(.init(render: promptTokens, ledger: $0.tokens)) }
                ?? .cold)
    }

    /// The decision a reuse of a carried cache stands for.
    ///
    /// - Parameters:
    ///   - reuse: what the caches hold of the render once the decision is
    ///     applied.
    ///   - render: the whole rendered prompt.
    ///   - ledger: the tokens the carried caches held before the decision.
    /// - Returns: the decision, with the seam a rewind went back to.
    private static func decision(
        of reuse: PromptCacheReuse, render: [Int], ledger: [Int]
    ) -> ExecutorPromptCacheDecision {
        switch reuse.kind {
        case .prefill:
            // The carried caches hold nothing, thus the pass is cold in all
            // but name: the whole prompt is fed.
            return .cold
        case .extend:
            return .extend
        case .splice:
            return .splice
        case .rewind:
            return .rewind(ExecutorPromptCacheDivergence(render: render, ledger: ledger))
        }
    }

    /// The rank of a token array a processor batched: one row for each
    /// sequence, and the tokens of that sequence along the other axis.
    private static let batchedTokenRank = 2

    /// The prompt tokens of `input`, or nil when `input` carries content a
    /// token ledger cannot describe.
    ///
    /// A VLM processor batches a text-only prompt to one row and masks every
    /// token present. That input carries nothing a ledger cannot describe, thus
    /// it passes. Media, a batch of more than one row and a mask that hides a
    /// token each carry more than the tokens say, thus they do not.
    private static func ledgerTokens(of input: LMInput) -> [Int]? {
        guard input.image == nil, input.video == nil, input.audio == nil,
            holdsOneSequence(input.text.tokens),
            marksEveryTokenPresent(input.text.mask, of: input.text.tokens)
        else {
            return nil
        }
        return input.text.tokens.asArray(Int.self)
    }

    /// Whether `tokens` holds one sequence: a plain vector, or a batch of one
    /// row.
    private static func holdsOneSequence(_ tokens: MLXArray) -> Bool {
        tokens.ndim == 1 || (tokens.ndim == batchedTokenRank && tokens.dim(0) == 1)
    }

    /// Whether `mask` marks every token of `tokens` present. No mask hides
    /// nothing; a mask of another shape, or one that holds a zero, hides
    /// something the ledger cannot name.
    private static func marksEveryTokenPresent(_ mask: MLXArray?, of tokens: MLXArray) -> Bool {
        guard let mask else { return true }
        return mask.shape == tokens.shape && mask.asArray(Int32.self).allSatisfy { $0 != 0 }
    }

    /// `input` narrowed to `tokens`, in the rank and the mask presence of
    /// `input`, thus the model sees the shape its processor makes.
    private static func narrowed(_ input: LMInput, to tokens: [Int]) -> LMInput {
        var narrowedTokens = MLXArray(tokens)
        if input.text.tokens.ndim == batchedTokenRank {
            narrowedTokens = narrowedTokens.expandedDimensions(axis: 0)
        }
        let narrowedMask = input.text.mask.map { ones(like: narrowedTokens).asType($0.dtype) }
        return LMInput(text: .init(tokens: narrowedTokens, mask: narrowedMask))
    }

    /// The cache this finished pass leaves for the next turn of its session.
    ///
    /// Generation feeds the tokens the model sampled into the caches, thus the
    /// caches hold what this pass represented before generation AND those
    /// generated tokens. The ledger names both, which is the ledger
    /// ``ChatSession`` keeps. A rewind back to the render alone is not available
    /// to every model -- a rotating cache past its sliding window drops the keys
    /// a rewind needs, and a recurrent cache never rewinds -- and this ledger
    /// asks for none.
    ///
    /// The next turn reconciles the two. When its render extends this ledger,
    /// `ExtendCachedPrefixRule` feeds the tail alone. When the render rewrites
    /// the turn the model wrote, a protocol rule splices past the commit that
    /// closed that turn, with the render recorded here as its proof. Otherwise
    /// `RewindToCommonPrefixRule` takes over and answers what the caches allow.
    ///
    /// - Parameters:
    ///   - generatedTokens: every token this pass generated, in order.
    ///     `TokenIterator` feeds a token before it answers it, thus each token
    ///     of this list stands in the caches.
    ///   - state: the model state the prefill of this pass left, which the
    ///     next turn seeds its iterator with, or nil for a model that carries
    ///     none.
    /// - Returns: the entry to check in, or nil when the caches did not land on
    ///   a position this ledger can name and the session must start cold.
    ///   ``commitOutcome(generatedTokens:state:)`` names the reason.
    func committed(
        generatedTokens: [Int], state: LMOutput.State? = nil
    ) -> ExecutorPromptCacheEntry? {
        commitOutcome(generatedTokens: generatedTokens, state: state).entry
    }

    /// The cache this finished pass leaves for the next turn, or the reason it
    /// leaves none.
    ///
    /// This is ``committed(generatedTokens:state:)`` with the reason kept,
    /// thus the log line of the pass can name it.
    ///
    /// - Parameters:
    ///   - generatedTokens: every token this pass generated, in order.
    ///   - state: the model state the prefill of this pass left, or nil.
    /// - Returns: the entry to check in, or the refusal.
    func commitOutcome(
        generatedTokens: [Int], state: LMOutput.State? = nil
    ) -> ExecutorPromptCacheCommitOutcome {
        guard let position = caches.first?.offset else { return .refused(.noCaches) }
        let positions = caches.map(\.offset)
        guard positions.allSatisfy({ $0 == position }) else {
            return .refused(.positionsDisagree(positions))
        }
        let ledgerLength = representedTokens.count
        guard position >= ledgerLength else {
            return .refused(.behindTheLedger(position: position, ledgerLength: ledgerLength))
        }
        let committedGeneratedTokenCount = position - ledgerLength
        guard committedGeneratedTokenCount <= generatedTokens.count else {
            return .refused(
                .pastTheGeneration(
                    position: position, ledgerLength: ledgerLength,
                    generatedTokenCount: generatedTokens.count))
        }
        return .checkedIn(
            ExecutorPromptCacheEntry(
                caches: caches,
                tokens: representedTokens + generatedTokens.prefix(committedGeneratedTokenCount),
                renderTokens: promptTokens,
                state: state))
    }
}

// MARK: - The rule that decided a pass

/// Where a render parts from the ledger it did not extend: the first index
/// where the two differ, and a short window of tokens on each side of it.
struct ExecutorPromptCacheDivergence: Equatable {

    /// How many tokens each side keeps past the seam. Enough to read the seam
    /// in a log line, and no more.
    static let reportedTokenCount = 12

    /// The index of the first token where the render and the ledger differ.
    /// This is the length of the prefix the two share, thus it is the length
    /// of the shorter side when that side is a prefix of the other.
    let index: Int

    /// The render's tokens from `index`, at most ``reportedTokenCount`` of
    /// them. Empty when the render ends at `index`.
    let renderTokens: [Int]

    /// The ledger's tokens from `index`, at most ``reportedTokenCount`` of
    /// them. Empty when the ledger ends at `index`.
    let ledgerTokens: [Int]

    /// Creates a seam at `index` with the tokens each side holds past it.
    ///
    /// - Parameters:
    ///   - index: the first index where the two sides differ.
    ///   - renderTokens: the render's tokens from `index`.
    ///   - ledgerTokens: the ledger's tokens from `index`.
    init(index: Int, renderTokens: [Int], ledgerTokens: [Int]) {
        self.index = index
        self.renderTokens = renderTokens
        self.ledgerTokens = ledgerTokens
    }

    /// Finds the seam between `render` and `ledger`.
    ///
    /// - Parameters:
    ///   - render: the whole rendered prompt.
    ///   - ledger: the tokens the carried caches held.
    init(render: [Int], ledger: [Int]) {
        let index = commonPrefixLength(of: render, and: ledger)
        self.init(
            index: index,
            renderTokens: Self.window(of: render, from: index),
            ledgerTokens: Self.window(of: ledger, from: index))
    }

    /// At most ``reportedTokenCount`` tokens of `tokens` from `index`.
    private static func window(of tokens: [Int], from index: Int) -> [Int] {
        Array(tokens.dropFirst(index).prefix(reportedTokenCount))
    }
}

/// The rule that decided what one generation pass does with the cache its
/// session carries.
///
/// The log line of the pass names it, thus a slow round of an agent run can
/// be attributed: a cold prefill, a rebuild the caches could not avoid, or a
/// long generation on a warm cache.
enum ExecutorPromptCacheDecision: Equatable {

    /// The session carried no cache, or a cache that held nothing, thus the
    /// whole prompt is fed into fresh caches.
    case cold

    /// The render extends the ledger whole, thus the tail alone is fed.
    case extend

    /// A rule of the response protocol kept the tokens the model wrote and fed
    /// the render past the turn the model committed.
    case splice

    /// The caches rewound to the seam, and the render past the seam is fed.
    case rewind(ExecutorPromptCacheDivergence)

    /// The render parts from the ledger at the seam and the caches cannot
    /// rewind to it, thus the whole prompt is fed into fresh caches.
    case rebuild(ExecutorPromptCacheDivergence)

    /// The name the log line gives the rule.
    var name: String {
        switch self {
        case .cold: "cold"
        case .extend: "extend"
        case .splice: "splice"
        case .rewind: "rewind"
        case .rebuild: "rebuild"
        }
    }

    /// The seam the decision names, or nil when the render extends the ledger
    /// or no ledger was carried.
    var divergence: ExecutorPromptCacheDivergence? {
        switch self {
        case .cold, .extend, .splice: nil
        case .rewind(let divergence), .rebuild(let divergence): divergence
        }
    }
}

/// Why a finished pass checked nothing in, thus why the next turn of its
/// session starts cold.
enum ExecutorPromptCacheCommitRefusal: Equatable {

    /// The pass carried no plan: its input carries content a token ledger
    /// cannot describe, or the pass owned its cache.
    case noPlan

    /// The plan carried no cache at all.
    case noCaches

    /// The caches do not agree on one position. The positions stand in cache
    /// order, thus a recurrent layer that never advanced shows as a zero.
    case positionsDisagree([Int])

    /// The caches stand before the end of the tokens the plan represented.
    case behindTheLedger(position: Int, ledgerLength: Int)

    /// The caches stand past the represented tokens plus every token the pass
    /// generated, thus the ledger cannot name what they hold.
    case pastTheGeneration(position: Int, ledgerLength: Int, generatedTokenCount: Int)

    /// The reason as the log line states it.
    var reason: String {
        switch self {
        case .noPlan:
            "the pass carried no plan"
        case .noCaches:
            "the pass carried no cache"
        case .positionsDisagree(let positions):
            "the caches disagree on their position \(positions)"
        case .behindTheLedger(let position, let ledgerLength):
            "the caches stand at \(position), behind the \(ledgerLength)-token ledger"
        case .pastTheGeneration(let position, let ledgerLength, let generatedTokenCount):
            "the caches stand at \(position), past the \(ledgerLength)-token ledger "
                + "and the \(generatedTokenCount) generated tokens"
        }
    }
}

/// What a finished pass leaves for the next turn: an entry to check in, or
/// the reason there is none.
enum ExecutorPromptCacheCommitOutcome {

    /// The entry the next turn of the session reuses.
    case checkedIn(ExecutorPromptCacheEntry)

    /// Nothing is checked in, for this reason.
    case refused(ExecutorPromptCacheCommitRefusal)

    /// The entry to check in, or nil when the pass was refused.
    var entry: ExecutorPromptCacheEntry? {
        switch self {
        case .checkedIn(let entry): entry
        case .refused: nil
        }
    }

    /// The refusal, or nil when an entry was checked in.
    var refusal: ExecutorPromptCacheCommitRefusal? {
        switch self {
        case .checkedIn: nil
        case .refused(let refusal): refusal
        }
    }
}

// MARK: - The log line of one pass

/// Composes the log lines of one generation pass.
///
/// Every function here is pure, thus a unit test reads the exact text that
/// `log show` shows for an agent run.
enum ExecutorPromptCacheReport {

    /// The line that names what the pass does with the carried cache.
    ///
    /// - Parameters:
    ///   - key: the session the cache belongs to, or nil when the request
    ///     names no session.
    ///   - plan: the plan of the pass, or nil when the pass carries no cache.
    ///   - decodeTokens: decodes a token window to the text the seam shows.
    /// - Returns: one line, with the seam and its decoded sides when the
    ///   decision names one.
    static func planLine(
        key: ExecutorPromptCacheKey?, plan: ExecutorPromptCachePlan?,
        decodeTokens: ([Int]) -> String
    ) -> String {
        let head = "prompt cache plan \(session(key)) "
        guard let plan else {
            return head + "rule=none (the input carries media, a batch or a mask)"
        }
        let rendered = plan.promptTokens.count
        let counts =
            "rendered=\(rendered) reused=\(plan.reusedTokenCount) "
            + "fed=\(rendered - plan.reusedTokenCount) rule=\(plan.decision.name)"
        guard let seam = plan.decision.divergence else { return head + counts }
        return head + counts
            + " divergence=\(seam.index) render=<<<\(decodeTokens(seam.renderTokens))>>> "
            + "ledger=<<<\(decodeTokens(seam.ledgerTokens))>>>"
    }

    /// The line of a guided pass, which owns its cache and carries none.
    ///
    /// - Parameter key: the session of the pass, or nil.
    /// - Returns: one line.
    static func guidedLine(key: ExecutorPromptCacheKey?) -> String {
        "prompt cache plan \(session(key)) "
            + "rule=guided (the guided pass owns its cache and carries none)"
    }

    /// The line that names what the finished pass checked in.
    ///
    /// - Parameters:
    ///   - key: the session of the pass, or nil.
    ///   - outcome: what the pass leaves for the next turn.
    /// - Returns: one line with the ledger length, or the reason nothing was
    ///   checked in.
    static func commitLine(
        key: ExecutorPromptCacheKey?, outcome: ExecutorPromptCacheCommitOutcome
    ) -> String {
        let head = "prompt cache commit \(session(key)) "
        switch outcome {
        case .checkedIn(let entry):
            return head + "ledger=\(entry.tokens.count)"
        case .refused(let refusal):
            return head + "checked in nothing: \(refusal.reason)"
        }
    }

    /// The line that names an entry the store evicted for its byte budget.
    ///
    /// - Parameters:
    ///   - key: the session of the evicted entry.
    ///   - byteCount: the bytes the entry held in memory.
    /// - Returns: one line with the session and the bytes.
    static func evictionLine(key: ExecutorPromptCacheKey, byteCount: Int) -> String {
        "prompt cache evict \(session(key)) bytes=\(byteCount)"
    }

    /// The line that names the end of one write of the spool.
    ///
    /// - Parameters:
    ///   - key: the session of the entry.
    ///   - byteCount: the bytes the entry held in memory.
    ///   - duration: how long the write took.
    ///   - outcome: what the write gave.
    /// - Returns: one line with the session, the bytes, the seconds of the
    ///   write and its result.
    static func spillLine(
        key: ExecutorPromptCacheKey, byteCount: Int, duration: Duration,
        outcome: ExecutorPromptCacheSpillOutcome
    ) -> String {
        let seconds = String(format: "%.3f", duration / .seconds(1))
        return "prompt cache spill \(session(key)) bytes=\(byteCount) seconds=\(seconds) "
            + "result=\(outcome.description)"
    }

    /// The line of a turn that found its cache on disk and starts cold,
    /// because this executor cannot read a spilled cache back yet.
    ///
    /// - Parameter key: the session of the turn.
    /// - Returns: one line.
    static func spilledColdStartLine(key: ExecutorPromptCacheKey) -> String {
        "prompt cache spilled \(session(key)) starts cold: the file is deleted, "
            + "because a restore from disk is not available"
    }

    /// The model and the session of `key`, or `none` for each when the
    /// request names no session.
    private static func session(_ key: ExecutorPromptCacheKey?) -> String {
        "model=\(key?.modelID ?? "none") session=\(key?.sessionID ?? "none")"
    }
}

/// The unified log of the executor prompt cache: `info` level, in the
/// `ExecutorPromptCache` category of the `com.apple.FoundationModels-MLX`
/// subsystem.
enum ExecutorPromptCacheLog {

    /// The log every line of the prompt cache goes to.
    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "ExecutorPromptCache")

    /// Writes `line` to the unified log at `info` level. Every field is
    /// public: the line carries token counts, byte counts, a session
    /// identifier and a short decoded seam, and `log show` must show them.
    ///
    /// - Parameter line: the line to write.
    static func info(_ line: String) {
        logger.info("\(line, privacy: .public)")
    }
}

// MARK: - Carrying the cache through one response

/// Carries the prompt cache of one session through one response.
///
/// `MLXLanguageModel.Executor.respond(to:model:streamingInto:)` checks the
/// session's cache out before it generates and checks this slot's entry back in
/// when it finishes, whatever the outcome.
///
/// The slot writes one log line when a pass is planned and one when it is
/// committed, at `info` level in the `com.apple.FoundationModels-MLX`
/// subsystem, thus `log show` names every rebuild of an agent run.
///
/// A response runs one generation pass at a time, and every pass runs inside the
/// model container, which serializes the passes of one model. This box is thus
/// written by one task at a time, and the `@unchecked Sendable` conformance
/// rests on that.
final class ExecutorPromptCacheSlot: @unchecked Sendable {

    /// The cache the session carries into its next turn, or nil when the next
    /// turn must start cold.
    private(set) var entry: ExecutorPromptCacheEntry?

    /// The prompt tokens the last planned pass did not feed to the model.
    ///
    /// Zero until a pass carries a cache. A pass that builds its own cache --
    /// the guided loop does -- leaves the value at zero, which is the measured
    /// truth for that pass.
    private(set) var reusedTokenCount = 0

    /// The session the cache belongs to, which every log line names, or nil
    /// when the request names no session.
    private let key: ExecutorPromptCacheKey?

    /// Receives each log line. The unified log by default; a test reads the
    /// lines through its own sink.
    private let report: (String) -> Void

    /// Creates a slot holding the entry a session checked out.
    ///
    /// - Parameters:
    ///   - entry: the entry the session checked out, or nil for a cold session.
    ///   - key: the session the entry belongs to, or nil when the request
    ///     names no session.
    ///   - report: receives each log line of the response.
    init(
        _ entry: ExecutorPromptCacheEntry?, key: ExecutorPromptCacheKey? = nil,
        report: @escaping (String) -> Void = ExecutorPromptCacheLog.info
    ) {
        self.entry = entry
        self.key = key
        self.report = report
    }

    /// Plans the pass that is about to run, records what that pass reuses,
    /// and logs the decision.
    ///
    /// The slot gives up its entry: the pass owns the caches until
    /// ``commit(_:generatedTokens:state:)`` takes them back.
    ///
    /// - Parameters:
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    ///   - decodeTokens: decodes the tokens on each side of a seam for the log
    ///     line of a rewind or a rebuild.
    /// - Returns: the plan, or nil when the pass carries no cache.
    func plan(
        input: LMInput, model: any LanguageModel, parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = [],
        decodeTokens: ([Int]) -> String
    ) throws -> ExecutorPromptCachePlan? {
        let plan = try ExecutorPromptCachePlan.make(
            reusing: entry, input: input, model: model, parameters: parameters,
            protocolRules: protocolRules)
        entry = nil
        reusedTokenCount = plan?.reusedTokenCount ?? 0
        report(ExecutorPromptCacheReport.planLine(key: key, plan: plan, decodeTokens: decodeTokens))
        return plan
    }

    /// Records that the pass about to run carries no cache from an earlier turn.
    ///
    /// The guided loop owns its key/value cache and accepts none from a caller,
    /// thus a guided pass reuses nothing and must report nothing, even when an
    /// earlier pass of the same response reused a prefix.
    func carriesNoCache() {
        reusedTokenCount = 0
        report(ExecutorPromptCacheReport.guidedLine(key: key))
    }

    /// Records the cache a finished pass leaves for the next turn, and logs
    /// the ledger length or the reason nothing is left.
    ///
    /// - Parameters:
    ///   - plan: the plan the pass ran, or nil when the pass carried no plan.
    ///   - generatedTokens: the tokens the pass generated, in order.
    ///   - state: the model state the prefill of the pass left, or nil.
    func commit(
        _ plan: ExecutorPromptCachePlan?, generatedTokens: [Int], state: LMOutput.State? = nil
    ) {
        let outcome =
            plan?.commitOutcome(generatedTokens: generatedTokens, state: state)
            ?? .refused(.noPlan)
        entry = outcome.entry
        report(ExecutorPromptCacheReport.commitLine(key: key, outcome: outcome))
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
