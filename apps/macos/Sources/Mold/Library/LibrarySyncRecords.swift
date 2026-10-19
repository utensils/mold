import Foundation
import Observation
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
    private var linkCache: (revision: Int, bucket: String, authority: [UUID: String]?, links: [PrintID: PrintID])?
    private var reverseCache: (revision: Int, bucket: String, index: [String: [String]])?
    private var recordsRevision = 0
    private let privacyChanges = LibrarySyncPrivacyRevision()
    /// Observable hide-only identity revision; ordinary bookkeeping does not
    /// invalidate source previews/indexes one write at a time.
    var privacyRevision: Int { privacyChanges.value }

    /// The record's whole contribution to `privacyLinks`; the identity clock
    /// moves only when this changes, never per organization checkpoint.
    private static func privacyEdge(_ key: String, _ record: LocalSyncRecord?,
                                    localHost: MoldHost.ID = MoldEngine.localHostID) -> SyncPrivacyEdge? {
        guard let record, let source = LocalSyncRecord.source(of: key) else { return nil }
        return SyncPrivacyEdge(origin: PrintID(host: source.host, filename: source.filename),
            destinationHost: record.destinationHost, destinationFilename: record.destinationFilename,
            localHost: localHost, sourceRoute: record.sourceRoute, destinationRoute: record.destinationRoute,
            destinationInstance: record.destinationInstance)
    }

    private var lineageCache: (revision: Int, authority: [UUID: String]?, incoming: [PrintID: [(key: String, record: LocalSyncRecord)]])?
    private var componentCache: (revision: Int, authority: [UUID: String]?, members: [PrintID: [PrintID]])?
    private var privacyCache: (revision: Int, authority: [UUID: String]?, links: [PrintID: PrintID])?

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
        guard !key.isEmpty, !key.contains("\n"), let legacy = doc.buckets.removeValue(forKey: "") else { return }
        doc.buckets[key, default: Bucket()].records.merge(legacy.records) { mine, _ in mine }
        doc.buckets[key, default: Bucket()].pending.merge(legacy.pending) { mine, _ in mine }
        doc.buckets[key, default: Bucket()].retry.formUnion(legacy.retry)
        recordsRevision += 1
        privacyChanges.value &+= 1
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

    /// Verified copy components across all destinations, represented by one
    /// stable member each. Rebuilt once per links revision or authority change,
    /// never once per copied print. Unknown/replaced routes cannot join.
    func links(localHost: MoldHost.ID, instance: String?, authority: [UUID: String]? = nil) -> [PrintID: PrintID] {
        load()
        let key = instance.map(Self.bucketKey) ?? doc.lastBucket ?? ""
        if let cached = linkCache, cached.revision == linksRevision, cached.bucket == key, cached.authority == authority { return cached.links }
        var edges: [(PrintID, PrintID)] = []
        // Every destination bucket participates; compressing to components
        // avoids one-neighbor dictionaries losing A->B/A->C fan-out.
        for bucket in doc.buckets.values {
            for (sourceKey, record) in bucket.records {
                guard record.provisional != true, let source = LocalSyncRecord.source(of: sourceKey) else { continue }
                let origin = PrintID(host: source.host, filename: source.filename)
                let copy = PrintID(host: record.destinationHost ?? localHost, filename: record.destinationFilename)
                if let authority {
                    guard let currentOrigin = authority[origin.host], record.sourceRoute == currentOrigin,
                          let route = record.destinationRoute,
                          authority[copy.host] == [copy.host.uuidString, route, record.destinationInstance ?? ""].joined(separator: "\n")
                    else { continue }
                }
                edges.append((copy, origin))
            }
        }
        let links = LibraryMerge.copyComponents(edges: edges)
        linkCache = (linksRevision, key, authority, links)
        return links
    }

    /// Hide-only links never authorize transfer skipping or organization ownership.
    /// Every record contributes, provisional or not: a conservative superset of
    /// the verified `links` snapshot, so a copy verified mid-run never drops out
    /// of privacy evidence before `finishRun` publishes it. Cached on the
    /// identity clock, which moves only when some record's edge does.
    func privacyLinks(localHost: MoldHost.ID, instance: String?, authority: [UUID: String]? = nil) -> [PrintID: PrintID] {
        load()
        if let cached = privacyCache, cached.revision == privacyRevision, cached.authority == authority { return cached.links }
        let edges = doc.buckets.values.flatMap { bucket in
            bucket.records.compactMap { Self.privacyEdge($0.key, $0.value, localHost: localHost) }
        }
        let result = SyncPrivacyEdge.components(edges, authority: authority)
        privacyCache = (privacyRevision, authority, result)
        return result
    }

    /// Incoming verified edges are immutable for a run. Rebuilding this per
    /// receipt or scanning all records per print would make a large sync quadratic.
    private func incomingRecords(authority: [UUID: String]? = nil) -> [PrintID: [(key: String, record: LocalSyncRecord)]] {
        load()
        if let cached = lineageCache, cached.revision == linksRevision, cached.authority == authority { return cached.incoming }
        var index: [PrintID: [(key: String, record: LocalSyncRecord)]] = [:]
        for bucket in doc.buckets.values {
            for (key, record) in bucket.records where record.provisional != true {
                let copy = PrintID(host: record.destinationHost ?? MoldEngine.localHostID, filename: record.destinationFilename)
                // Privacy reconciliation may journal a redundant path back
                // to its own root; that does not make the original forwarded.
                if record.originKey == LocalSyncRecord.key(hostID: copy.host, filename: copy.filename) { continue }
                if let authority {
                    guard let source = LocalSyncRecord.source(of: key),
                          let currentOrigin = authority[source.host], record.sourceRoute == currentOrigin,
                          let route = record.destinationRoute,
                          authority[copy.host] == [copy.host.uuidString, route, record.destinationInstance ?? ""].joined(separator: "\n")
                    else { continue }
                }
                index[copy, default: []].append((key, record))
            }
        }
        for id in Array(index.keys) { index[id]?.sort { $0.key < $1.key } }
        lineageCache = (linksRevision, authority, index)
        return index
    }

    /// Follow durable verified lineage, never filename/seed display grouping.
    func verifiedOrigin(of copy: PrintID, authority: [UUID: String]? = nil) -> String? {
        let incoming = incomingRecords(authority: authority)
        var current = copy
        var visited = Set<PrintID>()
        while visited.insert(current).inserted {
            let candidates = incoming[current] ?? []
            let roots = Set(candidates.map { $0.record.originKey ?? $0.key })
            guard roots.count <= 1 else { return nil }
            guard let edge = candidates.first,
                  let source = LocalSyncRecord.source(of: edge.key) else { break }
            if let root = edge.record.originKey { return root }
            current = PrintID(host: source.host, filename: source.filename)
        }
        return current == copy ? nil : LocalSyncRecord.key(hostID: current.host, filename: current.filename)
    }

    func hasVerifiedIncoming(_ copy: PrintID, authority: [UUID: String]? = nil) -> Bool { !(incomingRecords(authority: authority)[copy] ?? []).isEmpty }

    func organizationAtOrigin(of copy: PrintID, authority: [UUID: String]? = nil) -> LibraryOrganizationSnapshot? {
        let contributors = incomingRecords(authority: authority)[copy] ?? []
        let roots = Set(contributors.map { $0.record.originKey ?? $0.key })
        guard roots.count == 1 else { return nil }
        return contributors.first?.record.organization?.observed
    }

    func verifiedCopies(of source: PrintID, authority: [UUID: String]? = nil) -> [PrintID] {
        let graph = links(localHost: MoldEngine.localHostID, instance: nil, authority: authority)
        guard let component = graph[source] else { return [] }
        if componentCache?.revision != linksRevision || componentCache?.authority != authority {
            var members: [PrintID: [PrintID]] = [:]
            for (id, representative) in graph { members[representative, default: []].append(id) }
            for key in Array(members.keys) {
                members[key]?.sort { $0.host.uuidString + $0.filename < $1.host.uuidString + $1.filename }
            }
            componentCache = (linksRevision, authority, members)
        }
        return componentCache?.members[component] ?? []
    }

    func verifiedCopy(of source: PrintID, on host: UUID, authority: [UUID: String]? = nil) -> PrintID? {
        verifiedCopies(of: source, authority: authority).first { $0.host == host }
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
        if Self.privacyEdge(key, previous) != Self.privacyEdge(key, record) { privacyChanges.value &+= 1 }
        do {
            try await append(.record(bucket: bucketKey, key: key, record: record))
        } catch {
            doc.buckets[bucketKey]?.records[key] = previous
            recordsRevision += 1
            if Self.privacyEdge(key, previous) != Self.privacyEdge(key, record) { privacyChanges.value &+= 1 }
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
        privacyChanges.value &+= 1
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


/// Observation stays narrowly on the privacy identity clock; journal caches
/// and document mutations must not become view dependencies.
@MainActor
@Observable
private final class LibrarySyncPrivacyRevision {
    var value = 0
}
