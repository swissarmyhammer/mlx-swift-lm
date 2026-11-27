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
/// The entry of a clustered model holds a ``RemoteKVCache``, whose buffers
/// live on its worker. It counts no byte of the memory budget
/// (``ExecutorPromptCacheEntry/byteCount``), and no eviction for that budget
/// takes it: a prompt cache file cannot hold it, thus a spill would drop it.
/// Its bytes on the workers (``ExecutorPromptCacheEntry/workerByteCount``)
/// count against ``workerBudgetBytes`` instead. When a check-in takes their
/// total past that budget, the least recently used remote entries are
/// released, which frees their caches on the workers, and the next turn of
/// each such session starts cold. Every checked-in remote entry can go: the
/// entry of a live session, and the entry just checked in. Only an entry that
/// a pass holds is safe, because a check-out takes it out of the store.
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

    /// The worker budget the store takes when no host sets one: eight
    /// gibibytes.
    ///
    /// This process cannot read the memory of a worker, and no cluster plan
    /// in this package reserves bytes for the caches on one. The default thus
    /// keeps a few long sessions warm, and leaves the rest of the worker to
    /// its weights and to the caches of the passes that run. A host that knows
    /// the reservation of its workers sets it with
    /// ``configure(workerBudgetBytes:)``.
    static let defaultWorkerBudgetBytes = 8 << 30

    /// The most bytes the caches of the checked-in entries may hold on the
    /// workers of a clustered model, measured by
    /// ``ExecutorPromptCacheEntry/workerByteCount``.
    ///
    /// It is apart from ``memoryBudgetBytes``, which counts only the bytes of
    /// this process. The host sets it with ``configure(workerBudgetBytes:)``;
    /// when no host sets one, it is ``defaultWorkerBudgetBytes``.
    private(set) var workerBudgetBytes = defaultWorkerBudgetBytes

    /// The checked-in entries, by session, least recently used first.
    private var entries = ExecutorPromptCacheLRUTier<ExecutorPromptCacheEntry>()

    /// The sum of ``ExecutorPromptCacheEntry/byteCount`` over the checked-in
    /// entries.
    var retainedByteCount: Int { entries.byteCount }

    /// The sum of ``ExecutorPromptCacheEntry/workerByteCount`` over the
    /// checked-in entries: the bytes their caches hold on the workers.
    var workerByteCount: Int {
        entries.keys.reduce(0) { $0 + (entries[$1]?.workerByteCount ?? 0) }
    }

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
    private struct SpillingEntry: ExecutorPromptCacheTierRecord {

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

    /// The entries whose write to disk has not ended, by session. A write
    /// ends on its own, not by a budget, thus this tier keeps no usage order.
    private var spilling = ExecutorPromptCacheTier<SpillingEntry>()

    /// The sum of the bytes of the entries in ``spilling``. They are still in
    /// memory until their writes end, but they are not in
    /// ``retainedByteCount``.
    var spillingByteCount: Int { spilling.byteCount }

    /// One spill file on disk.
    private struct DiskFile: ExecutorPromptCacheTierRecord {

        /// The file and its session.
        let handle: ExecutorPromptCacheSpilledHandle

        /// The size of the file in bytes.
        let byteCount: Int

        /// The length of the token ledger of the entry in the file.
        let tokenCount: Int
    }

    /// The files of the entries on disk, by session, least recently used
    /// first.
    private var onDisk = ExecutorPromptCacheLRUTier<DiskFile>()

    /// The sum of the sizes of the files in ``onDisk``: the known disk total.
    var diskByteCount: Int { onDisk.byteCount }

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

    /// How many check-outs of each key have no check-in yet: the passes that hold the cache of
    /// the key now. ``copy(modelID:from:to:)`` reads it, because a held cache is in no tier.
    private var checkOutCounts: [ExecutorPromptCacheKey: Int] = [:]

    /// Whether a pass holds the cache of `key` now: a check-out of `key` has no check-in yet.
    ///
    /// - Parameter key: the session of the cache.
    /// - Returns: true when a pass holds the cache.
    func isCheckedOut(_ key: ExecutorPromptCacheKey) -> Bool {
        checkOutCounts[key] != nil
    }

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
    ///   keeps no entry in memory but the entries that hold a remote cache.
    func configure(memoryBudgetBytes budget: Int) {
        memoryBudgetBytes = budget
        evictToBudget()
    }

    /// Sets the worker budget and applies it at once: the least recently used
    /// remote entries are released until their worker bytes are at or below
    /// `budget`. A larger budget releases nothing.
    ///
    /// - Parameter budget: the most bytes the caches of the entries may hold
    ///   on the workers. Zero or less keeps no remote entry.
    func configure(workerBudgetBytes budget: Int) {
        workerBudgetBytes = budget
        releaseToWorkerBudget()
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
    /// Each check-out counts as a pass that holds the cache of `key` until
    /// the next ``checkIn(_:_:)`` of `key`, also when it finds nothing.
    ///
    /// - Parameter key: the session of the turn.
    /// - Returns: the entry, its file, or `.none` when the store holds nothing
    ///   for `key`.
    func checkOut(_ key: ExecutorPromptCacheKey) -> ExecutorPromptCacheCheckout {
        removeStaleFoldersAtFirstUse()
        checkOutCounts[key, default: 0] += 1
        if let entry = entries.remove(key) {
            return .memory(entry)
        }
        if let spill = spilling.remove(key) {
            return .memory(spill.entry)
        }
        if let file = onDisk.remove(key) {
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
    /// An entry that holds a remote cache stays in memory whatever the memory
    /// budget, and the least recently used remote entries are then released
    /// until their worker bytes are at or below ``workerBudgetBytes``: the
    /// entry just checked in, too, when it alone is over that budget.
    ///
    /// - Parameters:
    ///   - key: the session of the turn.
    ///   - entry: the entry the turn leaves, or nil when the next turn must
    ///     start cold.
    func checkIn(_ key: ExecutorPromptCacheKey, _ entry: ExecutorPromptCacheEntry?) {
        removeStaleFoldersAtFirstUse()
        endCheckOut(of: key)
        remove(key)
        guard let entry else { return }
        keepInMemory(entry, for: key)
    }

    /// Counts one check-out of `key` as ended. A check-in with no check-out changes no count.
    ///
    /// - Parameter key: the session of the check-in.
    private func endCheckOut(of key: ExecutorPromptCacheKey) {
        guard let count = checkOutCounts[key] else { return }
        checkOutCounts[key] = count > 1 ? count - 1 : nil
    }

    /// Keeps `entry` under `key` as the most recently used entry, then applies the memory
    /// budget and the worker budget. An entry larger than the whole memory budget goes to
    /// ``spill(_:_:)`` at once, and no other entry leaves for it. An entry that holds a remote
    /// cache never goes to ``spill(_:_:)``: the worker budget alone can release it.
    ///
    /// - Parameters:
    ///   - entry: the entry.
    ///   - key: the session of the entry. The store holds nothing for it on any tier.
    private func keepInMemory(_ entry: ExecutorPromptCacheEntry, for key: ExecutorPromptCacheKey) {
        guard entry.holdsRemoteCache || entry.byteCount <= memoryBudgetBytes else {
            spill(key, entry)
            return
        }
        entries.insert(entry, for: key)
        evictToBudget()
        releaseToWorkerBudget()
    }

    // MARK: - Copy of a session

    /// Copies the cache of `source` of `modelID`, under each KV-cache configuration, to
    /// `destination`. The cache of `source` does not change.
    ///
    /// The copy reads each entry in place on the actor, and it never checks an entry out.
    ///
    /// - An entry in memory, or an entry whose write to disk has not ended, gives a new entry
    ///   (``ExecutorPromptCacheEntry/copy()``). The copy goes in like a check-in: it is the
    ///   most recently used entry, it counts its whole ``ExecutorPromptCacheEntry/byteCount``,
    ///   and the memory budget applies. The entry of the source becomes the most recently used
    ///   entry before the copy, thus the budget does not send it to disk for its own copy.
    /// - A file on disk gives a clone of the file under a new name of `destination` (on APFS,
    ///   the two files share their blocks until one changes). The new file keeps the metadata
    ///   of the source, thus its handle records the key of the source for the read. The disk
    ///   budget applies to the new file.
    ///
    /// The store copies nothing when a pass holds a cache of `source` or of `destination`, or
    /// when it holds a cache of `destination`.
    ///
    /// - Parameters:
    ///   - modelID: the model of the two sessions.
    ///   - source: the session to copy.
    ///   - destination: the new session.
    /// - Returns: what the copy did.
    func copy(
        modelID: ExecutorPromptCacheKey.ModelID, from source: ExecutorPromptCacheKey.SessionID,
        to destination: ExecutorPromptCacheKey.SessionID
    ) -> PromptCacheCopyResult {
        removeStaleFoldersAtFirstUse()
        guard source != destination else { return .refused(.sameSession) }
        let storedKeys = Set(entries.keys).union(spilling.keys).union(onDisk.keys)
        let names = { (session: ExecutorPromptCacheKey.SessionID) in
            { (key: ExecutorPromptCacheKey) in key.modelID == modelID && key.sessionID == session }
        }
        if storedKeys.contains(where: names(destination)) {
            return .refused(.destinationHasCache)
        }
        if checkOutCounts.keys.contains(where: names(destination)) {
            return .refused(.destinationInUse)
        }
        if checkOutCounts.keys.contains(where: names(source)) {
            return .sourceInUse
        }
        let sourceKeys = storedKeys.filter(names(source))
        guard !sourceKeys.isEmpty else { return .sourceNotFound }
        var tokenCounts: [Int] = []
        for key in sourceKeys {
            let copyKey = ExecutorPromptCacheKey(
                modelID: modelID, sessionID: destination, kvCache: key.kvCache)
            if let tokenCount = copyEntry(of: key, to: copyKey) {
                tokenCounts.append(tokenCount)
            }
        }
        guard let tokenCount = tokenCounts.max() else { return .refused(.copyFailed) }
        return .copied(tokenCount: tokenCount)
    }

    /// Copies what the store holds for `key`, on its tier, to `copyKey`.
    ///
    /// - Parameters:
    ///   - key: a key of the source that the store holds on one tier.
    ///   - copyKey: the key of the copy. The store holds nothing for it.
    /// - Returns: the length of the token ledger of the copy, or nil when the file could not
    ///   be cloned.
    private func copyEntry(
        of key: ExecutorPromptCacheKey, to copyKey: ExecutorPromptCacheKey
    ) -> Int? {
        if let entry = entries[key] {
            entries.insert(entry, for: key)
            keepInMemory(entry.copy(), for: copyKey)
            return entry.tokens.count
        }
        if let spill = spilling[key] {
            keepInMemory(spill.entry.copy(), for: copyKey)
            return spill.entry.tokens.count
        }
        guard let file = onDisk[key] else { return nil }
        lastGeneration += 1
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: copyKey, generation: lastGeneration))
        do {
            try FileManager.default.copyItem(at: file.handle.url, to: url)
        } catch {
            ExecutorPromptCacheLog.info(
                "prompt cache cannot copy the file of \(key.sessionID.rawValue) to "
                    + "\(copyKey.sessionID.rawValue): \(error)")
            return nil
        }
        onDisk.insert(file, for: key)
        onDisk.insert(
            DiskFile(
                handle: ExecutorPromptCacheSpilledHandle(
                    url: url, key: copyKey, fileKey: file.handle.fileKey),
                byteCount: file.byteCount, tokenCount: file.tokenCount),
            for: copyKey)
        evictDiskToBudget()
        return file.tokenCount
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
        entries.remove(key)
        spilling.remove(key)
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

    /// Removes the least recently used entries until the total is at or below
    /// ``memoryBudgetBytes``, and gives each one to ``spill(_:_:)``. An entry
    /// that holds a remote cache never leaves: a spill would drop it, and it
    /// frees no byte of this process.
    private func evictToBudget() {
        while let oldest = entries.removeLeastRecentlyUsed(
            whileOver: memoryBudgetBytes, where: { !$0.holdsRemoteCache })
        {
            spill(oldest.key, oldest.record)
        }
    }

    /// Whether the checked-in remote entries are over ``workerBudgetBytes``.
    ///
    /// A budget of zero or less keeps no remote entry, thus every remote
    /// entry is over it, also one whose caches report zero bytes on the
    /// workers. A positive budget is over when ``workerByteCount`` is past it.
    private var isOverWorkerBudget: Bool {
        workerBudgetBytes <= 0 || workerByteCount > workerBudgetBytes
    }

    /// Releases the least recently used remote entries until the store is no
    /// longer over its worker budget (see ``isOverWorkerBudget``), and logs
    /// each one. A released entry never goes to ``spill(_:_:)``: a prompt
    /// cache file cannot hold it. The store drops its last reference, thus
    /// each of its ``RemoteKVCache`` releases its caches on the worker. A local
    /// entry never leaves here.
    private func releaseToWorkerBudget() {
        while isOverWorkerBudget,
            let oldest = entries.removeLeastRecentlyUsed(where: \.holdsRemoteCache)
        {
            ExecutorPromptCacheLog.info(
                ExecutorPromptCacheReport.workerReleaseLine(
                    key: oldest.key, workerByteCount: oldest.record.workerByteCount))
        }
    }

    /// Records the file of a write that ended as the most recently used file
    /// of its key, and applies the disk budget.
    ///
    /// - Parameters:
    ///   - handle: the file and its session.
    ///   - tokenCount: the length of the token ledger of the entry in the file.
    private func recordDiskFile(_ handle: ExecutorPromptCacheSpilledHandle, tokenCount: Int) {
        deleteDiskFile(of: handle.key)
        onDisk.insert(
            DiskFile(
                handle: handle, byteCount: Self.fileByteCount(at: handle.url),
                tokenCount: tokenCount),
            for: handle.key)
        evictDiskToBudget()
    }

    /// Removes the disk record of `key` and deletes its file.
    ///
    /// - Parameter key: the session of the file.
    private func deleteDiskFile(of key: ExecutorPromptCacheKey) {
        guard let file = onDisk.remove(key) else { return }
        ExecutorPromptCacheFile.removeFile(at: file.handle.url)
    }

    /// Deletes the least recently used files until the known disk total is at
    /// or below ``diskBudgetBytes``, and logs each one.
    private func evictDiskToBudget() {
        while let oldest = onDisk.removeLeastRecentlyUsed(whileOver: diskBudgetBytes) {
            ExecutorPromptCacheLog.info(
                ExecutorPromptCacheReport.diskEvictionLine(
                    key: oldest.key, byteCount: oldest.record.byteCount))
            ExecutorPromptCacheFile.removeFile(at: oldest.record.handle.url)
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
    /// reaches ``writeDidEnd(_:generation:byteCount:tokenCount:result:duration:)``: its
    /// file goes on record or is deleted, and ``pendingWriteCount`` drops, so
    /// no task in ``waitForSpills()`` waits forever. The store outlives its
    /// last caller by at most the writes in the queue.
    ///
    /// - Parameters:
    ///   - key: the session of the entry.
    ///   - entry: the entry that leaves memory. It holds no remote cache: a
    ///     prompt cache file cannot hold one, thus ``keepInMemory(_:for:)``
    ///     and ``evictToBudget()`` never give one here.
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
        spilling.insert(spill, for: key)
        pendingWriteCount += 1
        spoolWriter.enqueue(
            ExecutorPromptCacheSpillJob(input: spill.input, url: url) { result, duration in
                await self.writeDidEnd(
                    ExecutorPromptCacheSpilledHandle(url: url, key: key),
                    generation: spill.generation, byteCount: spill.byteCount,
                    tokenCount: spill.entry.tokens.count, result: result, duration: duration)
            })
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
    ///   - tokenCount: the length of the token ledger of the entry.
    ///   - result: the result of the write.
    ///   - duration: how long the write took.
    private func writeDidEnd(
        _ handle: ExecutorPromptCacheSpilledHandle, generation: UInt64, byteCount: Int,
        tokenCount: Int, result: Result<Void, any Error>, duration: Duration
    ) {
        defer { finishPendingWrite() }
        let isCurrent = spilling[handle.key]?.generation == generation
        if isCurrent {
            spilling.remove(handle.key)
        }
        let outcome: ExecutorPromptCacheSpillOutcome
        switch (result, isCurrent) {
        case (.success, true):
            recordDiskFile(handle, tokenCount: tokenCount)
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

/// An entry in memory is a record of the memory tier of the store, which counts its bytes.
extension ExecutorPromptCacheEntry: ExecutorPromptCacheTierRecord {}

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
