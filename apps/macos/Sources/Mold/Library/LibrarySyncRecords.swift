import Foundation
import MoldClient

/// Library Sync's bookkeeping, per destination instance: the origin-to-local
/// copy records, pending organization, each source's mirror cursor and the
/// retry set. A JSON document in Application Support, written off the main
/// actor; per-copy receipts go to an append-only, synchronized journal first,
/// so a receipt is durable before the retained-input transfer that follows it
/// without rewriting the whole document per print.
///
/// Without a directory it lives in memory only -- every test's default, one
/// fresh store per `LibraryStore`. Only `AppStores` persists to Application
/// Support.
///
/// A document this build cannot read (corrupt, or written by a newer
/// release) is moved aside, never overwritten, and the store is `blocked`:
/// it writes nothing and Sync All refuses to run on its empty stand-in --
/// an empty record set would recreate every copy deliberately removed here.
@MainActor
final class LibrarySyncRecords {
    /// One source's position in its replica, and the route it was read through.
    nonisolated struct SourceCursor: Codable, Equatable, Sendable {
        let cursor: ReplicaCursor
        let route: String
    }

    /// One destination instance's bookkeeping. The key "" holds what older
    /// releases recorded without knowing the instance, until one is known.
    nonisolated struct Bucket: Codable, Equatable, Sendable {
        var records: [String: LocalSyncRecord] = [:]
        var pending: [String: String] = [:]
        /// Per source host.
        var cursors: [String: SourceCursor] = [:]
        var destinationCursor: SourceCursor?
        /// Source keys a cursor-scoped plan must consider again.
        var retry: Set<String> = []
        /// Source keys last found kept removed from This Mac, so an
        /// unconsidered one is still reported as such.
        var keptRemoved: Set<String> = []
        /// When a run last planned every print. Cursor-scoped runs fall back
        /// to a full plan once this is a day old.
        var lastFullPlan: Date?
    }

    nonisolated struct Document: Codable, Equatable, Sendable {
        static let currentVersion = 1
        var version = currentVersion
        var seq = 0
        var buckets: [String: Bucket] = [:]
        var lastBucket: String?
    }

    nonisolated enum JournalEntry: Codable, Sendable {
        case record(bucket: String, key: String, record: LocalSyncRecord?)
        case pending(bucket: String, key: String, filename: String?)
    }

    nonisolated struct JournalLine: Codable, Sendable {
        let seq: Int
        let entry: JournalEntry
    }

    /// Application Support (the throwaway one under native UAT), migrating
    /// what older releases kept in the defaults suite.
    static func standard() -> LibrarySyncRecords {
        LibrarySyncRecords(directory: SecretStore.applicationSupport().appending(path: "LibrarySync"),
                           migrating: AppStorageSuite.defaults)
    }

    private let journal: LibrarySyncJournal?
    private let migrationDefaults: UserDefaults?
    /// Loaded on first use. Mutated in place, never through a copy: a copy
    /// of a big bucket per receipt would make a long run quadratic.
    private var doc = Document()
    private var loaded = false
    private var nextSeq = 1
    private var compactTask: Task<Void, Never>?
    /// Where an unreadable document was moved; non-nil means blocked.
    private var unreadable: URL?
    /// Bumped when links may have changed: on load and once per finished run,
    /// never per copied print.
    private(set) var linksRevision = 0
    private var linkCache: (revision: Int, bucket: String, links: [PrintID: PrintID])?
    private var reverseCache: (revision: Int, bucket: String, index: [String: [String]])?
    private var recordsRevision = 0

    init(directory: URL? = nil, migrating defaults: UserDefaults? = nil) {
        journal = directory.map(LibrarySyncJournal.init)
        migrationDefaults = defaults
    }

    // MARK: - Reading

    private func load() {
        guard !loaded else { return }
        loaded = true
        guard let journal else { return }
        switch journal.load() {
        case let .ready(document, newest):
            var read = document
            if let migrationDefaults { Self.migrate(from: migrationDefaults, into: &read, journal: journal) }
            nextSeq = max(read.seq, newest) + 1
            doc = read
        case let .unreadable(movedTo):
            unreadable = movedTo
        }
    }

    /// Why Sync All must not run, or nil. See the type's note.
    var blockedReason: String? {
        load()
        guard let unreadable else { return nil }
        return "This Mac’s Library Sync records could not be read (\(unreadable.lastPathComponent) was set aside). Sync All is paused so copies you removed here are not recreated; remove that file from \(unreadable.deletingLastPathComponent().path) to start over."
    }

    static func bucketKey(_ instance: String?) -> String { instance ?? "" }

    /// This instance's bucket. The first time an instance is known, what was
    /// recorded before it was folds into it (its own entries win).
    func bucket(_ instance: String?) -> Bucket {
        adopt(instance)
        return doc.buckets[Self.bucketKey(instance)] ?? Bucket()
    }

    private func adopt(_ instance: String?) {
        load()
        let key = Self.bucketKey(instance)
        guard !key.isEmpty, let legacy = doc.buckets.removeValue(forKey: "") else { return }
        doc.buckets[key, default: Bucket()].records.merge(legacy.records) { mine, _ in mine }
        doc.buckets[key, default: Bucket()].pending.merge(legacy.pending) { mine, _ in mine }
        doc.buckets[key, default: Bucket()].retry.formUnion(legacy.retry)
        recordsRevision += 1
        linksRevision += 1
        scheduleCompaction()
    }

    func records(_ instance: String?) -> [String: LocalSyncRecord] { bucket(instance).records }

    /// Destination filename -> the source keys whose copy it is.
    func sourceKeys(copiedAs instance: String?) -> [String: [String]] {
        let key = Self.bucketKey(instance)
        adopt(instance)
        if let cached = reverseCache, cached.revision == recordsRevision, cached.bucket == key { return cached.index }
        var index: [String: [String]] = [:]
        for (source, record) in doc.buckets[key]?.records ?? [:] { index[record.destinationFilename, default: []].append(source) }
        reverseCache = (recordsRevision, key, index)
        return index
    }

    /// Every record as a two-way link between a source print and the This Mac
    /// copy made of it -- what `LibraryMerge` joins exactly. Rebuilt once per
    /// `linksRevision`, not per listing or per copied print.
    func links(localHost: MoldHost.ID, instance: String?) -> [PrintID: PrintID] {
        load()
        let key = instance.map(Self.bucketKey) ?? doc.lastBucket ?? ""
        if let cached = linkCache, cached.revision == linksRevision, cached.bucket == key { return cached.links }
        var links: [PrintID: PrintID] = [:]
        for (sourceKey, record) in doc.buckets[key]?.records ?? [:] {
            guard let source = LocalSyncRecord.source(of: sourceKey) else { continue }
            let origin = PrintID(host: source.host, filename: source.filename)
            let copy = PrintID(host: localHost, filename: record.destinationFilename)
            links[origin] = copy
            links[copy] = origin
        }
        linkCache = (linksRevision, key, links)
        return links
    }

    // MARK: - Durable per-copy writes

    /// An output receipt or a sealed record, durable on disk before this
    /// returns -- or it throws and nothing changed. Links are left alone
    /// until the run finishes (`finishRun`).
    func checkpoint(_ record: LocalSyncRecord, key: String, instance: String?) async throws {
        let bucketKey = Self.bucketKey(instance)
        adopt(instance)
        let previous = doc.buckets[bucketKey]?.records[key]
        doc.buckets[bucketKey, default: Bucket()].records[key] = record
        recordsRevision += 1
        do {
            try await append(.record(bucket: bucketKey, key: key, record: record))
        } catch {
            doc.buckets[bucketKey]?.records[key] = previous
            recordsRevision += 1
            throw error
        }
    }

    func pending(_ key: String, instance: String?) -> String? {
        adopt(instance)
        return doc.buckets[Self.bucketKey(instance)]?.pending[key]
    }

    func markPending(_ key: String, filename: String, instance: String?) async throws {
        let bucketKey = Self.bucketKey(instance)
        adopt(instance)
        let previous = doc.buckets[bucketKey]?.pending[key]
        guard previous != filename else { return }
        doc.buckets[bucketKey, default: Bucket()].pending[key] = filename
        do {
            try await append(.pending(bucket: bucketKey, key: key, filename: filename))
        } catch {
            doc.buckets[bucketKey]?.pending[key] = previous
            throw error
        }
    }

    // MARK: - A finished run

    struct RunOutcome {
        var records: [String: LocalSyncRecord] = [:]
        var removed: Set<String> = []
        var clearedPending: Set<String> = []
        var sourceCursors: [String: SourceCursor] = [:]
        var destinationCursor: SourceCursor?
        var retried: Set<String> = []
        var settled: Set<String> = []
        var keptRemoved: Set<String> = []
        /// Every key the run looked at: its kept-removed state is now known.
        var reconsidered: Set<String> = []
        /// Set when this run planned every print and was not stopped.
        var fullPlan: Date?
    }

    /// Applies one run's outcome and writes the document now (off main).
    func finishRun(_ outcome: RunOutcome, instance: String?) async {
        let key = Self.bucketKey(instance)
        adopt(instance)
        var bucket = doc.buckets.removeValue(forKey: key) ?? Bucket()
        bucket.records.merge(outcome.records) { _, new in new }
        for removed in outcome.removed {
            bucket.records[removed] = nil
            bucket.pending[removed] = nil
        }
        for cleared in outcome.clearedPending { bucket.pending[cleared] = nil }
        bucket.cursors.merge(outcome.sourceCursors) { _, new in new }
        if let destination = outcome.destinationCursor { bucket.destinationCursor = destination }
        bucket.retry.subtract(outcome.settled)
        bucket.retry.formUnion(outcome.retried)
        bucket.retry.subtract(outcome.removed)
        bucket.keptRemoved.subtract(outcome.reconsidered)
        bucket.keptRemoved.formUnion(outcome.keptRemoved)
        bucket.keptRemoved.subtract(outcome.removed)
        if let fullPlan = outcome.fullPlan { bucket.lastFullPlan = fullPlan }
        doc.buckets[key] = bucket
        doc.lastBucket = key
        recordsRevision += 1
        linksRevision += 1
        await flush()
    }

    // MARK: - Persistence

    struct Blocked: LocalizedError {
        var errorDescription: String? { "This Mac’s Library Sync records could not be read, so nothing new is recorded." }
    }

    private func append(_ entry: JournalEntry) async throws {
        guard let journal else { return }
        guard unreadable == nil else { throw Blocked() }
        let seq = nextSeq
        nextSeq += 1
        try await journal.append(JournalLine(seq: seq, entry: entry))
        scheduleCompaction()
    }

    private func scheduleCompaction() {
        guard journal != nil, compactTask == nil else { return }
        compactTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            await self?.flush()
        }
    }

    /// Writes the whole document now and trims the journal behind it.
    func flush() async {
        compactTask?.cancel()
        compactTask = nil
        load()
        guard let journal, unreadable == nil else { return }
        doc.seq = nextSeq - 1
        await journal.compact(doc)
    }

    // MARK: - Migration

    private static let syncRecordStorageKey = "librarySyncCopiesV1"
    private static let pendingStorageKey = "librarySyncPendingOrganizationV1"
    private static let completedPrefix = "librarySyncCompletedCopyV1."
    private nonisolated struct CompletedCopy: Codable { let key: String; let record: LocalSyncRecord }

    /// One-time: what older releases kept in the defaults suite moves into
    /// the document (its own entries win), which is written before the
    /// defaults keys are removed.
    static func migrate(from defaults: UserDefaults, into document: inout Document, journal: LibrarySyncJournal?) {
        var records = defaults.data(forKey: syncRecordStorageKey)
            .flatMap { try? MoldJSON.decoder.decode([String: LocalSyncRecord].self, from: $0) } ?? [:]
        var receipts: [String] = []
        for (key, value) in defaults.dictionaryRepresentation() where key.hasPrefix(completedPrefix) {
            receipts.append(key)
            if let data = value as? Data, let completed = try? MoldJSON.decoder.decode(CompletedCopy.self, from: data) {
                records[completed.key] = completed.record
            }
        }
        let pending = defaults.data(forKey: pendingStorageKey)
            .flatMap { try? MoldJSON.decoder.decode([String: String].self, from: $0) } ?? [:]
        let hadLegacy = defaults.object(forKey: syncRecordStorageKey) != nil
            || defaults.object(forKey: pendingStorageKey) != nil || !receipts.isEmpty
        guard hadLegacy else { return }
        for (key, record) in records {
            document.buckets[bucketKey(record.destinationInstance), default: Bucket()].records[key]
                = document.buckets[bucketKey(record.destinationInstance)]?.records[key] ?? record
        }
        // Pending organization never named its instance.
        document.buckets["", default: Bucket()].pending.merge(pending) { mine, _ in mine }
        if let journal {
            guard journal.writeNow(document) else { return }
        }
        defaults.removeObject(forKey: syncRecordStorageKey)
        defaults.removeObject(forKey: pendingStorageKey)
        for key in defaults.dictionaryRepresentation().keys where key.hasPrefix(completedPrefix) {
            defaults.removeObject(forKey: key)
        }
    }
}

/// The files behind `LibrarySyncRecords`: `records.json`, written atomically,
/// and `journal.jsonl`, one synchronized line per receipt since.
actor LibrarySyncJournal {
    nonisolated let directory: URL
    private var compactedSeq = 0
    /// Lines appended since the last compaction, kept so a compaction can
    /// rewrite the journal with only what it did not include.
    private var tail: [LibrarySyncRecords.JournalLine] = []

    init(directory: URL) { self.directory = directory }

    nonisolated var documentURL: URL { directory.appending(path: "records.json") }
    nonisolated var journalURL: URL { directory.appending(path: "journal.jsonl") }
    nonisolated static let unreadablePrefix = "records.json.unreadable-"

    enum Loaded {
        case ready(LibrarySyncRecords.Document, newest: Int)
        /// The document this build cannot read, where it now is.
        case unreadable(URL)
    }

    /// The document with every journal line newer than it replayed, and the
    /// newest sequence number seen. A document that exists but cannot be
    /// read -- or a set-aside one with no readable successor -- blocks.
    nonisolated func load() -> Loaded {
        let files = FileManager.default
        var document = LibrarySyncRecords.Document()
        if let data = try? Data(contentsOf: documentURL) {
            guard let read = try? MoldJSON.localDecoder.decode(LibrarySyncRecords.Document.self, from: data),
                  read.version == LibrarySyncRecords.Document.currentVersion else {
                let stamp = Int(Date().timeIntervalSince1970 * 1000)
                let aside = directory.appending(path: Self.unreadablePrefix + String(stamp))
                if (try? files.moveItem(at: documentURL, to: aside)) == nil {
                    return .unreadable(documentURL)
                }
                return .unreadable(aside)
            }
            document = read
        } else if let aside = (try? files.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
            .first(where: { $0.lastPathComponent.hasPrefix(Self.unreadablePrefix) }) {
            // Set aside by an earlier launch and not resolved yet.
            return .unreadable(aside)
        }
        var newest = document.seq
        let lines = (try? Data(contentsOf: journalURL)).map { $0.split(separator: UInt8(ascii: "\n")) } ?? []
        for line in lines {
            guard let entry = try? MoldJSON.localDecoder.decode(LibrarySyncRecords.JournalLine.self, from: Data(line)),
                  entry.seq > document.seq else { continue }
            newest = max(newest, entry.seq)
            switch entry.entry {
            case let .record(bucket, key, record):
                document.buckets[bucket, default: .init()].records[key] = record
            case let .pending(bucket, key, filename):
                document.buckets[bucket, default: .init()].pending[key] = filename
            }
        }
        return .ready(document, newest: newest)
    }

    /// Appends one line and synchronizes it to disk, or throws.
    func append(_ line: LibrarySyncRecords.JournalLine) throws {
        guard line.seq > compactedSeq else { return }
        var data = try MoldJSON.localEncoder.encode(line)
        data.append(UInt8(ascii: "\n"))
        let files = FileManager.default
        try files.createDirectory(at: directory, withIntermediateDirectories: true)
        if !files.fileExists(atPath: journalURL.path) {
            guard files.createFile(atPath: journalURL.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: journalURL.path])
            }
        }
        let handle = try FileHandle(forWritingTo: journalURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.synchronize()
        tail.append(line)
    }

    func compact(_ document: LibrarySyncRecords.Document) {
        guard writeNow(document) else { return }
        compactedSeq = max(compactedSeq, document.seq)
        tail.removeAll { $0.seq <= compactedSeq }
        var lines = Data()
        for line in tail {
            guard let data = try? MoldJSON.localEncoder.encode(line) else { continue }
            lines.append(data)
            lines.append(UInt8(ascii: "\n"))
        }
        try? lines.write(to: journalURL, options: .atomic)
    }

    @discardableResult
    nonisolated func writeNow(_ document: LibrarySyncRecords.Document) -> Bool {
        guard let data = try? MoldJSON.localEncoder.encode(document) else { return false }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try data.write(to: documentURL, options: .atomic)
            return true
        } catch {
            return false
        }
    }
}
