// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

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
///
/// The files on disk are limited to ``diskBudgetBytes``. When a new file takes
/// the known disk total past the budget, the least recently used files are
/// deleted first. At its first use, the shared store also deletes the spool
/// folder of each process that does not run.
actor ExecutorPromptCacheStore {

    /// The store every executor shares.
    static let shared = ExecutorPromptCacheStore()

    /// The name of the folder, in the temporary directory, that holds the spool
    /// folder of each process.
    static let spoolFolderName = "mlx-prompt-cache"

    /// The folder that holds the spool folder of each process:
    /// `<temporary directory>/mlx-prompt-cache/`.
    static let spoolRootDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent(spoolFolderName, isDirectory: true)

    /// The spool folder of this process:
    /// `<temporary directory>/mlx-prompt-cache/<pid>-<UUID>/`. The UUID is made
    /// one time in each process, thus a new process never reads the folder of
    /// an old process that had the same process identifier.
    static let processSpoolDirectory = spoolRootDirectory.appendingPathComponent(
        "\(ProcessInfo.processInfo.processIdentifier)\(spoolFolderNameSeparator)\(UUID().uuidString)",
        isDirectory: true)

    /// The character between the process identifier and the UUID in the name
    /// of a spool folder.
    static let spoolFolderNameSeparator: Character = "-"

    /// The number of parts that the name of a spool folder has when it is cut
    /// at the first ``spoolFolderNameSeparator``: the process identifier and
    /// the UUID.
    private static let spoolFolderNamePartCount = 2

    /// The result of `kill(pid, 0)` for one process: the return value, and
    /// `errno` immediately after the call.
    typealias ProcessProbeResult = (killResult: Int32, errorNumber: Int32)

    /// The store the executor uses: ``shared``, unless a task binds another.
    ///
    /// Swift Testing runs suites in parallel, thus a test binds its own store
    /// with `$current.withValue(store) { ... }` and never shares entries or a
    /// budget with another test.
    @TaskLocal static var current: ExecutorPromptCacheStore = .shared

    /// The part of a free resource that a default budget takes: one quarter.
    /// In memory, the rest stays free for the weights and the live pass. On
    /// disk, the rest stays free for the other files of the volume.
    static let defaultBudgetDivisor = 4

    /// The most bytes the checked-in entries may hold in memory.
    ///
    /// The host sets it with ``configure(memoryBudgetBytes:)``. When no host
    /// sets one, the store reads the device at its first use and takes
    /// ``defaultMemoryBudgetBytes(workingSet:active:)``.
    private(set) lazy var memoryBudgetBytes: Int = Self.defaultMemoryBudgetBytes(
        workingSet: Int(clamping: GPU.deviceInfo().maxRecommendedWorkingSetSize),
        active: Memory.activeMemory)

    /// The most bytes the spill files may hold on disk.
    ///
    /// ``configure(diskBudgetBytes:)`` sets it. When nothing sets one, the
    /// store reads the free space of the volume of its spool folder at the
    /// first use and takes ``defaultDiskBudgetBytes(availableCapacity:)``.
    private(set) lazy var diskBudgetBytes: Int = Self.defaultDiskBudgetBytes(
        availableCapacity: availableCapacity(ofVolumeOf: directory))

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

    /// Reads the free space of the volume of the spool folder, for the default
    /// ``diskBudgetBytes``.
    private let capacityReader: ExecutorPromptCacheVolumeCapacityReader

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

    /// One spill file on disk.
    private struct DiskFile {

        /// The file and its session.
        let handle: ExecutorPromptCacheSpilledHandle

        /// The size of the file in bytes.
        let byteCount: Int
    }

    /// The files of the entries on disk, by session.
    private var onDisk: [ExecutorPromptCacheKey: DiskFile] = [:]

    /// The keys on disk, least recently used first.
    private var diskUsageOrder: [ExecutorPromptCacheKey] = []

    /// The sum of the sizes of the files in ``onDisk``: the known disk total.
    private(set) var diskByteCount = 0

    /// The three byte totals of the store, read in one step: ``retainedByteCount``,
    /// ``spillingByteCount`` and ``diskByteCount``.
    var usage: (memoryBytes: Int, spillingBytes: Int, diskBytes: Int) {
        (retainedByteCount, spillingByteCount, diskByteCount)
    }

    /// The folder whose spool folders of processes that do not run the store
    /// deletes at its first use, or nil when the clean-up ran or the store has
    /// no such folder.
    private var staleFolderRoot: URL?

    /// The generation of the last spill. Each spill takes the next number.
    private var lastGeneration: UInt64 = 0

    /// The writes that are in the queue or that run now.
    private var pendingWriteCount = 0

    /// The tasks that wait in ``waitForSpills()``.
    private var spillWaiters: [CheckedContinuation<Void, Never>] = []

    /// Creates a store that spills to the folder of this process,
    /// ``processSpoolDirectory``, and that deletes the spool folders of
    /// processes that do not run, in ``spoolRootDirectory``, at its first use.
    init() {
        self.init(
            directory: Self.processSpoolDirectory, staleFolderRoot: Self.spoolRootDirectory)
    }

    /// Creates a store that spills to `directory`.
    ///
    /// - Parameters:
    ///   - directory: the folder of the spill files. The default writer makes
    ///     it at the first write.
    ///   - writer: writes one prepared entry to one file. Tests give a writer
    ///     that holds or counts the writes.
    ///   - staleFolderRoot: the folder whose spool folders of processes that do
    ///     not run the store deletes at its first use, or nil for no clean-up.
    ///   - capacityReader: reads the free space of the volume of `directory`
    ///     for the default disk budget. Tests give a reader that reports a
    ///     fixed volume.
    init(
        directory: URL,
        writer: @escaping ExecutorPromptCacheFileWriter = ExecutorPromptCacheStore.writeSpillFile,
        staleFolderRoot: URL? = nil,
        capacityReader: @escaping ExecutorPromptCacheVolumeCapacityReader =
            ExecutorPromptCacheVolumeCapacity.read(of:)
    ) {
        self.directory = directory
        self.spoolWriter = ExecutorPromptCacheSpoolWriter(write: writer)
        self.staleFolderRoot = staleFolderRoot
        self.capacityReader = capacityReader
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
        return defaultBudget(ofFree: free)
    }

    /// The disk budget the store takes when nothing sets one: one quarter of
    /// the free space of the volume.
    ///
    /// - Parameter availableCapacity: the free space of the volume of the spool
    ///   folder, in bytes.
    /// - Returns: the budget in bytes, or zero when `availableCapacity` is zero
    ///   or less.
    static func defaultDiskBudgetBytes(availableCapacity: Int) -> Int {
        defaultBudget(ofFree: availableCapacity)
    }

    /// One ``defaultBudgetDivisor``-th part of `free`, or zero when `free` is
    /// zero or less.
    ///
    /// - Parameter free: the free bytes of a resource.
    /// - Returns: the budget in bytes.
    private static func defaultBudget(ofFree free: Int) -> Int {
        max(0, free) / defaultBudgetDivisor
    }

    /// The free space of the volume that holds `url`, in bytes: the
    /// ``ExecutorPromptCacheVolumeCapacity/usableFreeSpace`` of what the
    /// capacity reader reports. The store logs the key the value comes from.
    ///
    /// The spool folder is not made before the first write, thus the store
    /// reads the nearest folder above `url` that exists. When the value cannot
    /// be read, the store logs it and takes zero: the spool then keeps no file,
    /// and a disk that the store cannot measure does not fill up.
    ///
    /// - Parameter url: the spool folder.
    /// - Returns: the free space in bytes, or zero when it cannot be read.
    private func availableCapacity(ofVolumeOf url: URL) -> Int {
        let folder = ExecutorPromptCacheVolumeCapacity.nearestExistingFolder(to: url)
        let path = folder.path(percentEncoded: false)
        return Self.byteCountOrZero(of: "the free space of \(path)") {
            guard let freeSpace = try capacityReader(folder).usableFreeSpace else { return nil }
            ExecutorPromptCacheLog.info(
                "prompt cache disk budget reads \(freeSpace.key.rawValue) of \(path): "
                    + "\(freeSpace.bytes) bytes free")
            return freeSpace.bytes
        }
    }

    /// The byte count that `read` gives, or zero when it throws or gives nil.
    ///
    /// This is the one failure policy of the byte counts the store reads from
    /// the file system: a count that cannot be read goes to the log and
    /// counts as zero.
    ///
    /// - Parameters:
    ///   - label: what `read` reads, as the log line names it.
    ///   - read: reads the count, or gives nil when the file system gives none.
    /// - Returns: the count, or zero when it cannot be read.
    private static func byteCountOrZero(of label: String, read: () throws -> Int?) -> Int {
        do {
            guard let byteCount = try read() else { throw CocoaError(.fileReadUnknown) }
            return byteCount
        } catch {
            ExecutorPromptCacheLog.info(
                "prompt cache cannot read \(label): \(error); it counts as zero bytes")
            return 0
        }
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

    /// Sets the disk budget and applies it at once: the least recently used
    /// files are deleted until the known disk total is at or below `budget`.
    /// A larger budget deletes nothing.
    ///
    /// - Parameter budget: the most bytes the spill files may hold. Zero or
    ///   less keeps no file on disk.
    func configure(diskBudgetBytes budget: Int) {
        diskBudgetBytes = budget
        evictDiskToBudget()
    }

    /// Whether `kill(pid, 0)` shows that the process does not run.
    ///
    /// Only `ESRCH` shows that. `EPERM` shows a live process that this process
    /// cannot signal, and a result of zero shows a live process.
    ///
    /// - Parameters:
    ///   - killResult: the return value of `kill(pid, 0)`.
    ///   - errorNumber: `errno` immediately after the call.
    /// - Returns: true when the process does not run.
    static func isStaleProcess(killResult: Int32, errorNumber: Int32) -> Bool {
        killResult == -1 && errorNumber == ESRCH
    }

    /// Sends the signal zero to `pid`, which only checks that the process is
    /// there.
    ///
    /// - Parameter pid: the process identifier.
    /// - Returns: the return value of `kill`, and `errno` immediately after it.
    static func probeProcess(_ pid: pid_t) -> ProcessProbeResult {
        let killResult = kill(pid, 0)
        return (killResult: killResult, errorNumber: errno)
    }

    /// Deletes each spool folder in `root` whose process does not run.
    ///
    /// A spool folder has the name `<pid>-<UUID>`. The function keeps a folder
    /// with another name, and a folder whose process runs or cannot be
    /// signaled. A folder that cannot be read or deleted goes to the log.
    ///
    /// - Parameters:
    ///   - root: the folder that holds the spool folders.
    ///   - probe: checks one process. Tests give a probe that returns fixed
    ///     results.
    static func removeStaleSpoolFolders(
        in root: URL, probe: (pid_t) -> ProcessProbeResult = probeProcess
    ) {
        guard FileManager.default.fileExists(atPath: root.path(percentEncoded: false)) else {
            return
        }
        let folders: [URL]
        do {
            folders = try FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil)
        } catch {
            ExecutorPromptCacheLog.info(
                "prompt cache cannot read the spool folders in \(root.lastPathComponent): \(error)")
            return
        }
        for folder in folders {
            guard let pid = processIdentifier(ofSpoolFolder: folder.lastPathComponent) else {
                continue
            }
            let result = probe(pid)
            guard isStaleProcess(killResult: result.killResult, errorNumber: result.errorNumber)
            else { continue }
            ExecutorPromptCacheLog.info(
                "prompt cache removes the spool folder \(folder.lastPathComponent) "
                    + "of a process that does not run")
            ExecutorPromptCacheFile.removeFile(at: folder)
        }
    }

    /// The process identifier in the name of a spool folder, `<pid>-<UUID>`.
    ///
    /// - Parameter name: the name of the folder.
    /// - Returns: the process identifier, or nil when the name does not start
    ///   with a positive number and a `-`.
    private static func processIdentifier(ofSpoolFolder name: String) -> pid_t? {
        let parts = name.split(
            separator: spoolFolderNameSeparator, maxSplits: spoolFolderNamePartCount - 1)
        guard parts.count == spoolFolderNamePartCount, let pid = pid_t(parts[0]), pid > 0 else {
            return nil
        }
        return pid
    }

    /// Deletes the stale spool folders one time, at the first use of the store.
    private func removeStaleFoldersAtFirstUse() {
        guard let root = staleFolderRoot else { return }
        staleFolderRoot = nil
        Self.removeStaleSpoolFolders(in: root)
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
        removeStaleFoldersAtFirstUse()
        if let entry = removeEntry(key) {
            return .memory(entry)
        }
        if let spill = removeSpill(of: key) {
            return .memory(spill.entry)
        }
        if let file = removeDiskRecord(of: key) {
            return .spilled(file.handle)
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
        removeStaleFoldersAtFirstUse()
        remove(key)
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
    /// when `modelID` is nil: the entries in memory, the spills whose writes
    /// have not ended, and the files on disk.
    ///
    /// This is a release that the host asked for, not an eviction for the
    /// budget, thus the entries do not go through ``spill(_:_:)``.
    ///
    /// - Parameter modelID: the model whose sessions to release, or nil for
    ///   every model.
    func evict(modelID: ExecutorPromptCacheKey.ModelID?) {
        removeAll { modelID == nil || $0.modelID == modelID }
    }

    /// Releases the cache of `sessionID` for every model: the entries in
    /// memory, the spills whose writes have not ended, and the files on disk.
    /// The sessions with another identifier stay.
    ///
    /// - Parameter sessionID: the session to release.
    func evict(sessionID: ExecutorPromptCacheKey.SessionID) {
        removeAll { $0.sessionID == sessionID }
    }

    /// Releases the cache of `sessionID` of `modelID` under every KV-cache
    /// configuration: the entries in memory, the spills whose writes have not
    /// ended, and the files on disk. The other sessions of the model, and the
    /// sessions of other models, stay.
    ///
    /// - Parameters:
    ///   - modelID: the model whose session to release.
    ///   - sessionID: the session to release.
    func evict(
        modelID: ExecutorPromptCacheKey.ModelID, sessionID: ExecutorPromptCacheKey.SessionID
    ) {
        removeAll { $0.modelID == modelID && $0.sessionID == sessionID }
    }

    /// Removes `key` from every tier of the store: the entry in memory, the
    /// spill whose write has not ended, and the file on disk. A key that the
    /// store does not hold changes nothing.
    ///
    /// The write of a removed spill finds another generation, or none, when
    /// it ends. It then records nothing and deletes its own file.
    ///
    /// - Parameter key: the session to remove.
    func remove(_ key: ExecutorPromptCacheKey) {
        removeEntry(key)
        removeSpill(of: key)
        deleteDiskFile(of: key)
    }

    /// Calls ``remove(_:)`` for each key on any tier that `isReleased` accepts.
    ///
    /// - Parameter isReleased: tells if one key is released.
    private func removeAll(where isReleased: (ExecutorPromptCacheKey) -> Bool) {
        let keys = Set(entries.keys).union(spilling.keys).union(onDisk.keys)
        for key in keys where isReleased(key) {
            remove(key)
        }
    }

    /// Removes the entry of `key` from the entries, the usage order and the
    /// byte total.
    ///
    /// - Parameter key: the session of the entry.
    /// - Returns: the entry, or nil when the store holds none for `key`.
    @discardableResult
    private func removeEntry(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheEntry? {
        guard let entry = entries.removeValue(forKey: key) else { return nil }
        usageOrder.removeAll { $0 == key }
        retainedByteCount -= entry.byteCount
        return entry
    }

    /// Removes the least recently used entries until the total is at or below
    /// ``memoryBudgetBytes``, and gives each one to ``spill(_:_:)``.
    private func evictToBudget() {
        while retainedByteCount > memoryBudgetBytes, let oldest = usageOrder.first,
            let entry = removeEntry(oldest)
        {
            spill(oldest, entry)
        }
    }

    /// Records the file of a write that ended as the most recently used file
    /// of its key, and applies the disk budget.
    ///
    /// - Parameter handle: the file and its session.
    private func recordDiskFile(_ handle: ExecutorPromptCacheSpilledHandle) {
        deleteDiskFile(of: handle.key)
        let file = DiskFile(handle: handle, byteCount: Self.fileByteCount(at: handle.url))
        onDisk[handle.key] = file
        diskUsageOrder.append(handle.key)
        diskByteCount += file.byteCount
        evictDiskToBudget()
    }

    /// Removes the disk record of `key` from the records, the usage order and
    /// the disk total. The file stays.
    ///
    /// - Parameter key: the session of the file.
    /// - Returns: the record, or nil when no file of `key` is on disk.
    @discardableResult
    private func removeDiskRecord(of key: ExecutorPromptCacheKey) -> DiskFile? {
        guard let file = onDisk.removeValue(forKey: key) else { return nil }
        diskUsageOrder.removeAll { $0 == key }
        diskByteCount -= file.byteCount
        return file
    }

    /// Removes the disk record of `key` and deletes its file.
    ///
    /// - Parameter key: the session of the file.
    private func deleteDiskFile(of key: ExecutorPromptCacheKey) {
        guard let file = removeDiskRecord(of: key) else { return }
        ExecutorPromptCacheFile.removeFile(at: file.handle.url)
    }

    /// Deletes the least recently used files until the known disk total is at
    /// or below ``diskBudgetBytes``, and logs each one.
    private func evictDiskToBudget() {
        while diskByteCount > diskBudgetBytes, let oldest = diskUsageOrder.first,
            let file = onDisk[oldest]
        {
            ExecutorPromptCacheLog.info(
                ExecutorPromptCacheReport.diskEvictionLine(key: oldest, byteCount: file.byteCount))
            deleteDiskFile(of: oldest)
        }
    }

    /// The size of the file at `url`, in bytes.
    ///
    /// A size that cannot be read goes to the log and counts as zero: the file
    /// stays on record, thus a check-out or a removal still finds it.
    ///
    /// - Parameter url: the URL of the file.
    /// - Returns: the size in bytes, or zero when it cannot be read.
    private static func fileByteCount(at url: URL) -> Int {
        byteCountOrZero(of: "the size of \(url.lastPathComponent)") {
            try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
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
    /// The job holds the store strongly until its write ends, thus each write
    /// reaches ``writeDidEnd(_:generation:byteCount:result:duration:)``: its
    /// file goes on record or is deleted, and ``pendingWriteCount`` drops, so
    /// no task in ``waitForSpills()`` waits forever. The store outlives its
    /// last caller by at most the writes in the queue.
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
            ExecutorPromptCacheSpillJob(input: spill.input, url: url) { result, duration in
                await self.writeDidEnd(
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
    /// check-out, a check-in or a removal removed the spill, the write is stale: its file
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
            recordDiskFile(handle)
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

    /// A check-out, a check-in or a removal took the entry back during the
    /// write, thus the file was deleted.
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

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
