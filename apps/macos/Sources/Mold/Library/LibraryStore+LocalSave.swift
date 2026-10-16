import CryptoKit
import Foundation
import MoldClient

private struct MirrorTarget: Sendable {
    let hostID: MoldHost.ID
    let print: GalleryPrint
    let source: any MoldBackend
    let originContext: String
    let isCurrent: @MainActor @Sendable () -> Bool
    let collections: [(slug: String, name: String)]
}

private struct MirrorResult: Sendable {
    let filename: String?
    let alreadyLocal: Bool
    let error: String?
    var sourceMedia: RetainedSourceMedia.MirrorResult = .complete
    var repeatAcknowledgable = false
}


/// Each stage receives the captured authority it needs and returns its own result.
/// Only LibraryStore publishes progress, issues and the final report.
private enum MirrorDiscovery {
    case ready(MirrorSources)
    case stopped([String])
}

private struct MirrorSources {
    let candidates: [LibraryEntry]
    let targets: [MirrorTarget]
    let machines: [MoldHost.ID: (MoldHost, any MoldBackend)]
    let collections: [MoldHost.ID: [String: Collection]]
    let checkpoints: [MoldHost.ID: GallerySyncCheckpoint]
    let unresolvedCollections: Set<String>
    let failures: [String]
}

private enum MirrorPreparation {
    case ready(MirrorDestination)
    case unavailable(String)
}

private struct MirrorDestination {
    let host: MoldHost
    let backend: any MoldBackend
    let prints: [String: GalleryPrint]
    let trashNames: Set<String>
    let checkpoint: GallerySyncCheckpoint?
    let records: [String: LocalSyncRecord]
    let pendingOrganization: PendingSyncOrganization?
    let collectionNames: [String: String]
    let createdCollections: Int
    let failedCollectionSlugs: Set<String>
    let failures: [String]
}

private struct MirrorCopy {
    let target: MirrorTarget
    let filename: String
}

private struct MirrorWorkPlan {
    let items: [MirrorWorkItem]
    let cachedCopies: [MirrorCopy]
    let keptRemoved: Int
}

private struct MirrorWorkItem: Sendable {
    let index: Int
    let target: MirrorTarget
    let filename: String
    let occupied: Bool
    let cachedFilename: String?

    func perform(to destination: any MoldBackend, knownPrints: [String: GalleryPrint],
                 pending: PendingSyncOrganization?, destinationInstance: String?) async -> (Int, MirrorResult) {
        if let cachedFilename {
            return (index, await repairCachedSources(target, to: destination, as: cachedFilename))
        }
        return (index, await mirror(target, to: destination, as: filename, occupied: occupied,
                                   knownPrints: knownPrints, pending: pending,
                                   destinationInstance: destinationInstance))
    }
}

private struct MirrorAssignments {
    let collectionNames: [String: String]
    let collectionFiles: [String: [String]]
    let titles: [GalleryTitleAssignment]
    let favorites: [String]
    let tags: [String: [String]]
}

private struct MirrorCopies {
    let successful: [MirrorCopy]
    let records: [String: LocalSyncRecord]
    let assignments: MirrorAssignments
    let transferred: Int
    let alreadyLocal: Int
    let failures: [String]
}

private struct MirrorOrganization {
    let failedFiles: Set<String>
    let failures: [String]
}

private func targetVersioned(_ print: GalleryPrint) -> Bool {
    print.mediaVersion != nil && print.canonicalMetadataJSON != nil
}

private func canAcknowledgeSyncError(_ error: Error) -> Bool {
    if error is URLError || error is CancellationError { return false }
    if let client = error as? MoldClientError {
        switch client {
        case .unreachable, .unauthorized, .licenseRequired, .malformedResponse: return false
        case let .http(status, code, message):
            if code == "RETAINED_MEDIA_COPY_INCOMPLETE", message?.contains("could not verify") == true { return false }
            return status != 401 && status != 403 && status != 429 && status < 500
        }
    }
    // Retained-source errors are stable per-output facts; acknowledgment
    // suppresses repeated interruption, never retries or the full issue list.
    return !(error is SyncDiskSpaceError)
}

private struct SyncDiskSpaceError: LocalizedError {
    let filename: String
    var errorDescription: String? {
        "There isn’t enough free space to stage \(filename). Free space on This Mac, then try Sync All again."
    }
}

private func checkSyncStagingSpace(for print: GalleryPrint) throws {
    guard let bytes = print.sizeBytes, bytes > 512 * 1_024 * 1_024 else { return }
    // The app's download, upload envelope, and local engine staging can
    // briefly coexist. Keep the arithmetic checked for untrusted gallery rows.
    let staged = Int64(bytes).multipliedReportingOverflow(by: 3)
    let required = staged.partialValue.addingReportingOverflow(512 * 1_024 * 1_024)
    guard !staged.overflow, !required.overflow,
          let free = try FileManager.default.attributesOfFileSystem(
            forPath: FileManager.default.temporaryDirectory.path)[.systemFreeSize] as? NSNumber,
          free.int64Value > required.partialValue
    else { throw SyncDiskSpaceError(filename: print.filename) }
}

/// A durable origin-to-local filename link. A gallery's media version is its
/// server-owned change token; the recipe digest and source identity prevent
/// two hosts with the same basename from sharing a link. The destination is
/// checked against the current local listing before a transfer is skipped.
private struct LocalSyncRecord: Codable {
    let sourceVersion: String
    let sourceTimestamp: UInt64
    let sourceSize: Int
    let sourceRecipe: String
    let destinationFilename: String
    let destinationVersion: String
    let destinationTimestamp: UInt64
    let destinationSize: Int
    let destinationRecipe: String
    let destinationInstance: String?
    var sourceCheckpoint: String?
    var destinationCheckpoint: String?
    var sourceRoute: String?
    var destinationRoute: String?


    private static let syncRecordStorageKey = "librarySyncCopiesV1"

    static func key(hostID: MoldHost.ID, filename: String) -> String {
        "\(hostID.uuidString):\(filename)"
    }

    static func recipe(_ print: GalleryPrint) -> String? {
        guard let encoded = print.canonicalMetadataJSON else { return nil }
        return SHA256.hash(data: encoded).map { String(format: "%02x", $0) }.joined()
    }

    func matches(source: GalleryPrint, local: GalleryPrint) -> Bool {
        source.mediaVersion == sourceVersion && source.timestamp == sourceTimestamp
            && source.sizeBytes == sourceSize && Self.recipe(source) == sourceRecipe
            && local.filename == destinationFilename
            && local.mediaVersion == destinationVersion
            && local.timestamp == destinationTimestamp
            && local.sizeBytes == destinationSize
            && Self.recipe(local) == destinationRecipe
    }

    /// Every sync record as a two-way link between a source print and the
    /// This Mac copy made of it -- what `LibraryMerge` joins exactly. Decoded
    /// only when the stored bytes change: `rebuild()` runs on every listing.
    @MainActor static func links(localHost: MoldHost.ID) -> [PrintID: PrintID] {
        let data = AppStorageSuite.defaults.data(forKey: syncRecordStorageKey)
        if let cached = linkCache, cached.data == data { return cached.links }
        var links: [PrintID: PrintID] = [:]
        for (key, record) in load() {
            guard let colon = key.firstIndex(of: ":"),
                  let host = UUID(uuidString: String(key[..<colon])) else { continue }
            let source = PrintID(host: host, filename: String(key[key.index(after: colon)...]))
            let copy = PrintID(host: localHost, filename: record.destinationFilename)
            links[source] = copy
            links[copy] = source
        }
        linkCache = (data, links)
        return links
    }

    @MainActor private static var linkCache: (data: Data?, links: [PrintID: PrintID])?

    private struct CompletedCopy: Codable { let key: String; let record: LocalSyncRecord }
    private static let completedPrefix = "librarySyncCompletedCopyV1."
    @MainActor static func checkpoint(_ record: Self, key: String) {
        guard let data = try? MoldJSON.encoder.encode(CompletedCopy(key: key, record: record)) else { return }
        let digest = SHA256.hash(data: Data(key.utf8)).map { String(format: "%02x", $0) }.joined()
        AppStorageSuite.defaults.set(data, forKey: completedPrefix + digest)
        linkCache = nil
    }

    static func load() -> [String: Self] {
        let data = AppStorageSuite.defaults.data(forKey: syncRecordStorageKey)
        var records = data.flatMap { try? MoldJSON.decoder.decode([String: Self].self, from: $0) } ?? [:]
        for (key, value) in AppStorageSuite.defaults.dictionaryRepresentation() where key.hasPrefix(completedPrefix) {
            if let data = value as? Data, let completed = try? MoldJSON.decoder.decode(CompletedCopy.self, from: data) {
                records[completed.key] = completed.record
            }
        }
        return records
    }

    @MainActor static func save(_ records: [String: Self]) {
        var merged = records
        // Include output-only receipts written by a transfer that failed
        // before returning a complete MirrorResult to the caller.
        for (key, value) in AppStorageSuite.defaults.dictionaryRepresentation() where key.hasPrefix(completedPrefix) {
            if let data = value as? Data, let completed = try? MoldJSON.decoder.decode(CompletedCopy.self, from: data) {
                merged[completed.key] = completed.record
            }
        }
        guard let data = try? MoldJSON.encoder.encode(merged) else { return }
        AppStorageSuite.defaults.set(data, forKey: syncRecordStorageKey)
        for key in AppStorageSuite.defaults.dictionaryRepresentation().keys where key.hasPrefix(completedPrefix) {
            AppStorageSuite.defaults.removeObject(forKey: key)
        }
        linkCache = nil
    }

    init?(source: GalleryPrint, local: GalleryPrint, destinationInstance: String? = nil) {
        guard let sourceVersion = source.mediaVersion,
              let sourceSize = source.sizeBytes,
              let sourceRecipe = Self.recipe(source),
              let destinationVersion = local.mediaVersion,
              let destinationSize = local.sizeBytes,
              let destinationRecipe = Self.recipe(local)
        else { return nil }
        self.sourceVersion = sourceVersion
        sourceTimestamp = source.timestamp
        self.sourceSize = sourceSize
        self.sourceRecipe = sourceRecipe
        destinationFilename = local.filename
        self.destinationVersion = destinationVersion
        destinationTimestamp = local.timestamp
        self.destinationSize = destinationSize
        self.destinationRecipe = destinationRecipe
        self.destinationInstance = destinationInstance
    }
}

/// Written before an import starts, so a relaunch can finish organizing a
/// print that landed just before the app stopped. A pre-existing local match
/// never gets an entry and keeps its own title, favourite, and tags.
@MainActor
private final class PendingSyncOrganization {
    private static let pendingStorageKey = "librarySyncPendingOrganizationV1"
    private var destinations: [String: String]

    init() {
        let data = AppStorageSuite.defaults.data(forKey: Self.pendingStorageKey)
        destinations = data.flatMap { try? MoldJSON.decoder.decode([String: String].self, from: $0) } ?? [:]
    }

    func destination(for key: String) -> String? { destinations[key] }

    func mark(_ key: String, filename: String) {
        destinations[key] = filename
        save()
    }

    func clear(_ key: String) {
        destinations.removeValue(forKey: key)
        save()
    }

    private func save() {
        guard let data = try? MoldJSON.encoder.encode(destinations) else { return }
        AppStorageSuite.defaults.set(data, forKey: Self.pendingStorageKey)
    }
}

private func sameDescriptor(_ source: GalleryPrint, _ local: GalleryPrint) -> Bool {
    source.timestamp == local.timestamp
        && RetainedSourceMedia.mirrorMetadataMatches(source.canonicalMetadataJSON, local.canonicalMetadataJSON)
        && (source.metadataSynthetic ?? false) == (local.metadataSynthetic ?? false)
}

private func boundedMirrorName(_ filename: String, marker: String) -> String {
    let url = URL(fileURLWithPath: filename)
    var stem = url.deletingPathExtension().lastPathComponent
    let suffix = "~\(marker).\(url.pathExtension)"
    while (stem + suffix).utf8.count > SafeFilename.maxBytes && !stem.isEmpty {
        stem.removeLast()
    }
    return stem + suffix
}

private func collisionName(for target: MirrorTarget, file: URL) async throws -> String {
    let hash = try await LibraryMirrorFiles.sha256(file)
    return boundedMirrorName(target.print.filename,
                             marker: "\(target.hostID.uuidString.lowercased())-\(hash)")
}

private func repairCachedSources(_ target: MirrorTarget, to destination: any MoldBackend,
                                 as filename: String) async -> MirrorResult {
    do {
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
            from: target.source, to: destination, as: filename)
        return MirrorResult(filename: filename, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
    } catch {
        return MirrorResult(filename: nil, alreadyLocal: false, error: error.sentence, repeatAcknowledgable: canAcknowledgeSyncError(error))
    }
}

private func mirror(_ target: MirrorTarget, to destination: any MoldBackend,
                    as requestedName: String, occupied: Bool,
                    knownPrints: [String: GalleryPrint],
                    pending: PendingSyncOrganization?, destinationInstance: String?) async -> MirrorResult {
    do {
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        try checkSyncStagingSpace(for: target.print)
        let sourceIdentity = try await RetainedSourceMedia.preflightMirror(
            for: target.print.filename, metadata: target.print.metadata, from: target.source, to: destination)
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        let file = try await target.source.mediaFile(target.print.filename, trashed: false)
        defer { try? FileManager.default.removeItem(at: file) }
        if occupied {
            let existing = try await destination.mediaFile(requestedName, trashed: false)
            defer { try? FileManager.default.removeItem(at: existing) }
            let equal: Bool
            if let local = knownPrints[requestedName], sameDescriptor(target.print, local) {
                equal = try await LibraryMirrorFiles.equal(file, existing)
            } else {
                equal = false
            }
            if equal {
                guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: requestedName, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file)
                return MirrorResult(filename: requestedName, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
            }
        }
        let filename = occupied ? try await collisionName(for: target, file: file) : requestedName
        if occupied, let local = knownPrints[filename] {
            let existing = try await destination.mediaFile(filename, trashed: false)
            defer { try? FileManager.default.removeItem(at: existing) }
            let equal = sameDescriptor(target.print, local)
                ? try await LibraryMirrorFiles.equal(file, existing) : false
            if equal {
                guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: filename, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file)
                return MirrorResult(filename: filename, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
            }
        }
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        let item = try await LibraryMirrorFiles.importDescriptor(for: target.print, file: file)
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        let sourceKey = LocalSyncRecord.key(hostID: target.hostID,
                                            filename: target.print.filename)
        await pending?.mark(sourceKey, filename: filename)
        do {
            let imported = try await destination.importPrint(item, as: filename)
            if imported != filename { await pending?.mark(sourceKey, filename: imported) }
            if pending != nil, let localPrint = try? await destination.galleryPrint(imported),
               let checkpoint = LocalSyncRecord(source: target.print, local: localPrint, destinationInstance: destinationInstance) {
                // Output receipt is durable even if retained-input transfer is
                // interrupted. No checkpoint tokens means resume MUST repair
                // retained inputs before declaring the copy complete.
                await LocalSyncRecord.checkpoint(checkpoint, key: sourceKey)
            }
            let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                from: target.source, to: destination, as: imported, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file)
            return MirrorResult(filename: imported, alreadyLocal: false, error: nil, sourceMedia: sourceMedia)
        } catch {
            // The server may have committed the file before the response was
            // lost. Keep the pending marker for the next fresh listing.
            throw error
        }
    } catch {
        return MirrorResult(filename: nil, alreadyLocal: false,
                            error: error.sentence, repeatAcknowledgable: canAcknowledgeSyncError(error))
    }
}

struct SyncListingSnapshot {
    let host: MoldHost
    let instance: String?
    let prints: [GalleryPrint]
    let etag: String?
}

@MainActor
extension LibraryStore {
    private func syncListing(on host: MoldHost, from backend: any MoldBackend, trash: Bool = false) async throws -> [GalleryPrint] {
        let key = host.id.uuidString + (trash ? ":trash" : ":active")
        let instance = hosts.instanceID(of: host.id)
        let cached = syncListings[key].flatMap { $0.host == host && $0.instance == instance ? $0 : nil }
        let fetched = try await (trash ? backend.trashedPrints(etag: cached?.etag) : backend.gallery(etag: cached?.etag))
        guard hosts.host(host.id) == host, hosts.instanceID(of: host.id) == instance else {
            throw MoldClientError.unreachable("The machine changed while reading its Library.")
        }
        switch fetched {
        case let .fresh(prints, etag):
            syncListings[key] = SyncListingSnapshot(host: host, instance: instance, prints: prints, etag: etag)
            return prints
        case .notModified:
            guard let cached else { throw MoldClientError.malformedResponse }
            return cached.prints
        }
    }

    static func canSaveLocally(_ entry: LibraryEntry) -> Bool {
        guard entry.print.kind == .picture else { return false }
        return canSyncLocally(entry)
            && ["png", "jpg", "jpeg", "webp"].contains(
                URL(fileURLWithPath: entry.print.filename).pathExtension.lowercased())
    }

    static func canSyncLocally(_ entry: LibraryEntry) -> Bool {
        guard entry.hostID != MoldEngine.localHostID else {
            return false
        }
        return ["png", "jpg", "jpeg", "webp", "gif", "apng", "mp4", "wav", "glb"].contains(
            URL(fileURLWithPath: entry.print.filename).pathExtension.lowercased())
    }

    func saveLocally(_ selection: [LibraryEntry]) async {
        await mirrorLocally(selection, syncAll: false)
        syncSession.runFinished(in: self)
    }

    func syncAllLocally() async {
        await mirrorLocally([], syncAll: true)
        syncSession.runFinished(in: self)
    }

    private func mirrorLocally(_ selection: [LibraryEntry], syncAll: Bool) async {
        guard !localSaveRunning else { return }
        localSaveRunning = true
        defer { localSaveRunning = false }
        localSaveIssueKeys = [:]
        localSaveNotices = []
        localSaveAlertPresented = false
        let visibilityGeneration = collectionVisibility.generation
        defer { localSaveTask = nil }
        let candidates = selection.filter(Self.canSaveLocally)
        guard syncAll || !candidates.isEmpty else { return }
        guard let local = hosts.host(MoldEngine.localHostID), hosts.isUp(local) else {
            localSaveReport = "Start This Mac’s engine to save remote prints in its Library."
            localSaveFailures = []
            localSaveAlertPresented = true
            return
        }
        localSaveSuppressImports = true
        localSaveProgress = syncAll ? "Finding remote prints and collections…"
            : "Preparing \(candidates.count.formatted()) prints…"
        defer {
            localSaveProgress = nil
            localSaveSuppressImports = false
            localSaveStopRequested = false
        }

        let sources: MirrorSources
        switch await discoverMirrorSources(candidates, local: local, syncAll: syncAll) {
        case let .ready(value): sources = value
        case let .stopped(failures):
            localSaveReport = "Sync stopped before any prints were copied."
            localSaveFailures = failures + ["Save stopped. Remaining prints were not transferred."]
            localSaveAlertPresented = true
            return
        }
        let prepared: MirrorDestination
        switch await prepareMirrorDestination(sources, local: local, syncAll: syncAll,
                                              visibilityGeneration: visibilityGeneration) {
        case let .ready(value): prepared = value
        case let .unavailable(message):
            localSaveReport = message
            localSaveFailures = sources.failures
            localSaveAlertPresented = true
            return
        }
        let plan = await planMirrorWork(sources, destination: prepared, syncAll: syncAll)
        let copies = await copyMirrorWork(plan, sources: sources, prepared: prepared, syncAll: syncAll)
        let organization = await organizeMirrorCopies(copies.assignments, destination: prepared.backend)
        var failures = copies.failures + organization.failures
        if copies.transferred > 0 || !copies.assignments.collectionFiles.isEmpty || prepared.createdCollections > 0 {
            if let collections = try? await prepared.backend.collections() {
                collectionsPerHost[prepared.host.id] = collections
            }
        }
        // End suppression before the final read. An unrelated client may
        // import while collections load; the read then catches it, and later
        // events apply normally.
        localSaveSuppressImports = false
        localSaveProgress = "Refreshing This Mac’s Library…"
        await refresh(on: prepared.host.id)
        await reconcileCollectionVisibility()
        if syncAll && !plan.items.isEmpty {
            failures += await verifyMirrorCopies(sources: sources, prepared: prepared,
                plan: plan, copies: copies, organization: organization)
        }
        publishMirrorReport(selectedCount: selection.count, syncAll: syncAll,
            sources: sources, prepared: prepared, plan: plan, copies: copies, failures: failures)
    }

    private func discoverMirrorSources(_ initialCandidates: [LibraryEntry], local: MoldHost,
                                       syncAll: Bool) async -> MirrorDiscovery {
        var candidates = initialCandidates
        var failures: [String] = []
        var sourcePrints: [MoldHost.ID: [String: GalleryPrint]] = [:]
        var sourceMachines: [MoldHost.ID: (MoldHost, any MoldBackend)] = [:]
        var originContexts: [MoldHost.ID: String] = [:]
        var sourceCollections: [MoldHost.ID: [String: Collection]] = [:]
        var sourceCheckpoints: [MoldHost.ID: GallerySyncCheckpoint] = [:]
        var hostErrors: [MoldHost.ID: String] = [:]
        let sourceHostIDs = syncAll
            ? Set(hosts.hosts.filter { $0.id != local.id }.map(\.id))
            : Set(candidates.map(\.hostID))
        for hostID in sourceHostIDs.sorted() {
            if Task.isCancelled || localSaveStopRequested { break }
            guard let sourceHost = hosts.host(hostID), let source = hosts.backend(for: hostID) else {
                hostErrors[hostID] = "its machine is no longer available"
                if syncAll {
                    failures.append("\(hosts.name(of: hostID) ?? hostID.uuidString): its machine is no longer available")
                }
                continue
            }
            sourceMachines[hostID] = (sourceHost, source)
            originContexts[hostID] = [hostID.uuidString, hosts.host(hostID)?.baseURL.absoluteString ?? "",
                                     hosts.instanceID(of: hostID) ?? ""].joined(separator: "\n")
            do {
                let prints = try await syncListing(on: sourceHost, from: source)
                guard hosts.host(hostID) == sourceHost else { throw MoldClientError.unreachable("The machine route changed while reading its Library.") }
                if syncAll, let checkpoint = try? await source.gallerySyncCheckpoint() {
                    sourceCheckpoints[hostID] = checkpoint
                }
                sourcePrints[hostID] = Dictionary(prints.map { ($0.filename, $0) },
                                                  uniquingKeysWith: { first, _ in first })
                if syncAll {
                    candidates += prints.map { LibraryEntry(host: sourceHost, print: $0) }
                        .filter(Self.canSyncLocally)
                }
                let selectedNames = Set(candidates.filter { $0.hostID == hostID }
                    .map { $0.print.filename })
                if syncAll || prints.contains(where: { selectedNames.contains($0.filename)
                    && !$0.collectionList.isEmpty }) {
                    do {
                        let collections = try await source.collections()
                        guard hosts.host(hostID) == sourceHost else { throw MoldClientError.unreachable("The machine route changed while reading its collections.") }
                        sourceCollections[hostID] = Dictionary(collections.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
                    } catch {
                        failures.append("Collections on \(hostID): \(error.sentence)")
                    }
                }
            } catch {
                hostErrors[hostID] = error.sentence
                if syncAll { failures.append("\(hosts.name(of: hostID) ?? hostID.uuidString): \(error.sentence)") }
            }
        }

        if Task.isCancelled || localSaveStopRequested { return .stopped(failures) }

        if syncAll { localSaveProgress = "Preparing \(candidates.count.formatted()) prints…" }

        var targets: [MirrorTarget] = []
        var unresolvedCollections: Set<String> = []
        for entry in candidates {
            if let error = hostErrors[entry.hostID] {
                failures.append("\(entry.print.filename): \(error)")
                continue
            }
            guard let print = sourcePrints[entry.hostID]?[entry.print.filename],
                  let (sourceHost, source) = sourceMachines[entry.hostID], hosts.host(entry.hostID) == sourceHost else {
                failures.append("\(entry.print.filename): its recipe is no longer available")
                continue
            }
            let collectionIDs = sourceCollections[entry.hostID] ?? [:]
            let names = print.collectionList.compactMap { id -> (slug: String, name: String)? in
                guard let collection = collectionIDs[id] else { return nil }
                return (collection.slug, collection.name)
            }
            if names.count != print.collectionList.count {
                failures.append("\(print.filename): a source collection is no longer available")
                unresolvedCollections.insert(LocalSyncRecord.key(
                    hostID: entry.hostID, filename: print.filename))
            }
            targets.append(MirrorTarget(hostID: entry.hostID, print: print,
                                        source: source, originContext: originContexts[entry.hostID] ?? "",
                                        isCurrent: { [weak self] in
                                            self?.hosts.host(sourceHost.id) == sourceHost && self?.hosts.host(local.id) == local
                                        }, collections: names))
        }

        return .ready(MirrorSources(candidates: candidates, targets: targets,
            machines: sourceMachines, collections: sourceCollections, checkpoints: sourceCheckpoints,
            unresolvedCollections: unresolvedCollections, failures: failures))
    }

    private func prepareMirrorDestination(_ sources: MirrorSources, local: MoldHost, syncAll: Bool,
                                          visibilityGeneration: UUID) async -> MirrorPreparation {
        let targets = sources.targets
        let sourceCollections = sources.collections
        var failures: [String] = []
        let destination = hosts.backend(for: local)
        let localPrints: [GalleryPrint]
        do {
            localPrints = try await syncListing(on: local, from: destination)
        } catch {
            return .unavailable("Couldn’t read This Mac’s Library: \(error.sentence)")
        }
        let localTrashNames: Set<String>
        if syncAll {
            do {
                let prints = try await syncListing(on: local, from: destination, trash: true)
                localTrashNames = Set(prints.map(\.filename))
            } catch {
                return .unavailable("Couldn’t read This Mac’s Trash safely: \(error.sentence)")
            }
        } else { localTrashNames = [] }
        let localByName = Dictionary(localPrints.map { ($0.filename, $0) },
                                     uniquingKeysWith: { first, _ in first })
        let destinationCheckpoint = syncAll ? try? await destination.gallerySyncCheckpoint() : nil
        let syncedCopies = syncAll ? LocalSyncRecord.load() : [:]
        let pendingOrganization = syncAll ? PendingSyncOrganization() : nil
        let localCollections: [Collection]
        do {
            localCollections = try await destination.collections()
        } catch {
            failures.append("This Mac’s collections: \(error.sentence)")
            localCollections = collectionsPerHost[local.id] ?? []
        }
        var localCollectionBySlug = Dictionary(localCollections.map { ($0.slug, $0) }, uniquingKeysWith: { first, _ in first })
        var localNames = Dictionary(localCollections.map {
            ($0.slug, $0.name)
        }, uniquingKeysWith: { first, _ in first })
        var createdCollections = 0
        var failedCollectionSlugs: Set<String> = []
        // Visibility is a shared shelf attribute. Include existing and empty
        // shelves, not just memberships discovered while copying outputs.
        var collectionInventories = collectionsPerHost
        for (host, collections) in sourceCollections { collectionInventories[host] = Array(collections.values) }
        collectionInventories[local.id] = localCollections
        let desiredShelves = CollectionShelf.merge(collectionInventories)
        for shelf in desiredShelves {
            if Task.isCancelled || localSaveStopRequested { break }
            guard syncAll || targets.contains(where: { $0.collections.contains { $0.slug == shelf.slug } }) else { continue }
            do {
                let existing: Collection
                if let collection = localCollectionBySlug[shelf.slug] {
                    existing = collection
                } else {
                    let description = collectionInventories.values.flatMap { $0 }
                        .first { $0.slug == shelf.slug && $0.description != nil }?.description
                    existing = try await destination.createCollection(name: shelf.name, description: description)
                    guard existing.slug == shelf.slug else { throw MoldClientError.malformedResponse }
                    localCollectionBySlug[shelf.slug] = existing
                    localNames[shelf.slug] = existing.name
                    createdCollections += 1
                }
                localCollectionBySlug[shelf.slug] = try await syncCollectionVisibility(
                    existing, fallbackHidden: shelf.hidden, generation: visibilityGeneration,
                    destination: local, backend: destination)
            } catch {
                failures.append("Collection “\(shelf.name)”: \(error.sentence)")
                failedCollectionSlugs.insert(shelf.slug)
            }
        }
        return .ready(MirrorDestination(host: local, backend: destination, prints: localByName,
            trashNames: localTrashNames, checkpoint: destinationCheckpoint, records: syncedCopies,
            pendingOrganization: pendingOrganization, collectionNames: localNames,
            createdCollections: createdCollections, failedCollectionSlugs: failedCollectionSlugs,
            failures: failures))
    }

    private func planMirrorWork(_ sources: MirrorSources, destination: MirrorDestination,
                                syncAll: Bool) async -> MirrorWorkPlan {
        let targets = sources.targets
        let sourceCheckpoints = sources.checkpoints
        let local = destination.host
        let localByName = destination.prints
        let localTrashNames = destination.trashNames
        let syncedCopies = destination.records
        let pendingOrganization = destination.pendingOrganization
        let destinationCheckpoint = destination.checkpoint
        var work: [MirrorWorkItem] = []
        var cachedCopies: [MirrorCopy] = []
        var claimedOriginals: Set<String> = []
        var keptRemoved = 0
        for (index, target) in targets.enumerated() {
            if Task.isCancelled || localSaveStopRequested { break }
            let original = target.print.filename
            let sourceKey = LocalSyncRecord.key(hostID: target.hostID, filename: original)
            if syncAll, localTrashNames.contains(original)
                || (syncedCopies[sourceKey].map { $0.destinationInstance == hosts.instanceID(of: local.id) && localByName[$0.destinationFilename] == nil } ?? false) {
                keptRemoved += 1
                continue
            }
            let canClaimOriginal = !syncAll || !claimedOriginals.contains(original)
            let requestedName: String
            if let pendingName = pendingOrganization?.destination(for: sourceKey) {
                requestedName = pendingName
                claimedOriginals.insert(original)
            } else if canClaimOriginal {
                requestedName = original
                claimedOriginals.insert(original)
            } else {
                requestedName = boundedMirrorName(
                    original, marker: target.hostID.uuidString.lowercased())
            }
            if syncAll,
               let cached = syncedCopies[sourceKey],
               let existing = localByName[cached.destinationFilename],
               cached.matches(source: target.print, local: existing) {
                let sourceToken = sourceCheckpoints[target.hostID].flatMap { checkpoint in
                    checkpoint.revisions[original].map { checkpoint.instanceId + ":" + $0 }
                }
                let destinationToken = destinationCheckpoint.flatMap { checkpoint in
                    checkpoint.revisions[existing.filename].map { checkpoint.instanceId + ":" + $0 }
                }
                if pendingOrganization?.destination(for: sourceKey) == nil,
                   await target.isCurrent(),
                   let sourceToken, let destinationToken,
                   cached.sourceCheckpoint == sourceToken, cached.destinationCheckpoint == destinationToken,
                   cached.sourceRoute == target.originContext,
                   cached.destinationRoute == local.baseURL.absoluteString,
                   cached.destinationInstance == hosts.instanceID(of: local.id) {
                    cachedCopies.append(MirrorCopy(target: target, filename: existing.filename))
                    continue
                }
                work.append(MirrorWorkItem(index: index, target: target, filename: requestedName, occupied: true, cachedFilename: existing.filename))
                continue
            }
            work.append(MirrorWorkItem(index: index, target: target, filename: requestedName, occupied: localByName[requestedName] != nil, cachedFilename: nil))
        }

        return MirrorWorkPlan(items: work, cachedCopies: cachedCopies, keptRemoved: keptRemoved)
    }

    private func copyMirrorWork(_ plan: MirrorWorkPlan, sources: MirrorSources,
                                prepared: MirrorDestination, syncAll: Bool) async -> MirrorCopies {
        let targets = sources.targets
        let work = plan.items
        let local = prepared.host
        let destination = prepared.backend
        let localByName = prepared.prints
        let localNames = prepared.collectionNames
        let pendingOrganization = prepared.pendingOrganization
        var syncedCopies = prepared.records
        var failures = sources.failures + prepared.failures
        var collectionNames: [String: String] = [:]
        var collectionFiles: [String: [String]] = [:]
        var titleAssignments: [GalleryTitleAssignment] = []
        var favoriteFiles: [String] = []
        var tagFiles: [String: [String]] = [:]
        var successfulCopies: [MirrorCopy] = []
        var alreadyLocal = 0
        var transferred = 0
        var completed = 0
        func record(_ target: MirrorTarget, as filename: String, organize: Bool) {
            if syncAll { successfulCopies.append(MirrorCopy(target: target, filename: filename)) }
            let sourceKey = LocalSyncRecord.key(hostID: target.hostID, filename: target.print.filename)
            let fileMembership = !syncAll || syncedCopies[sourceKey] == nil
                || syncedCopies[sourceKey]?.destinationInstance != hosts.instanceID(of: local.id)
            for collection in target.collections where fileMembership {
                let name = localNames[collection.slug] ?? collection.name
                if let previous = collectionNames[collection.slug], previous != name {
                    failures.append("Collection “\(collection.slug)” has different names on source machines; using “\(previous)”.")
                } else {
                    collectionNames[collection.slug] = name
                }
                collectionFiles[collection.slug, default: []].append(filename)
            }
            guard organize else { return }
            if let title = target.print.title, !title.isEmpty {
                titleAssignments.append(.init(filename: filename, title: title))
            }
            if target.print.isFavorite { favoriteFiles.append(filename) }
            for tag in target.print.tagList {
                tagFiles[tag, default: []].append(filename)
            }
        }

        for cached in plan.cachedCopies {
            alreadyLocal += 1
            record(cached.target, as: cached.filename, organize: false)
        }

        // Large clips can briefly occupy download and upload staging space.
        // Keep those batches serial; do the same when free space cannot
        // comfortably hold three 512 MiB transfers with three staged copies.
        var next = 0
        let destinationInstance = hosts.instanceID(of: local.id)
        localSaveProgress = "Checking \(work.count.formatted()) changed prints · Already here \(alreadyLocal.formatted())"
        await withTaskGroup(of: (Int, MirrorResult).self) { group in
            let free = (try? FileManager.default.attributesOfFileSystem(
                forPath: FileManager.default.temporaryDirectory.path)[.systemFreeSize]
                as? NSNumber)?.int64Value ?? 0
            let largeTransfer = work.contains {
                ($0.target.print.sizeBytes ?? 0) > 512 * 1_024 * 1_024
            }
            let lanes = largeTransfer || free < 5 * 1_024 * 1_024 * 1_024 ? 1 : 3
            for _ in 0..<min(lanes, work.count) {
                let item = work[next]
                next += 1
                group.addTask {
                    await item.perform(to: destination, knownPrints: localByName,
                                       pending: pendingOrganization, destinationInstance: destinationInstance)
                }
            }
            for await (index, result) in group {
                completed += 1
                if let name = result.filename {
                    if result.sourceMedia == .legacyInputsUnavailable {
                        let target = targets[index]
                        let notice = "\(hosts.name(of: target.hostID) ?? target.hostID.uuidString) · \(target.print.filename): Original inputs are unavailable; the finished file is saved."
                        localSaveNotices.append(notice)
                        recordMirrorIssue(notice, target: target, local: local)
                    }
                    if result.alreadyLocal { alreadyLocal += 1 }
                    else { transferred += 1 }
                    let sourceKey = LocalSyncRecord.key(hostID: targets[index].hostID,
                                                        filename: targets[index].print.filename)
                    let pendingMatch = pendingOrganization?.destination(for: sourceKey) == name
                    record(targets[index], as: name,
                           organize: !result.alreadyLocal || pendingMatch)
                    // A response-proven output and retained copy gets its own
                    // durable checkpoint before organization. Targeted listing
                    // reads one row rather than the whole destination gallery.
                    if syncAll, let localPrint = try? await destination.galleryPrint(name),
                       let checkpoint = LocalSyncRecord(source: targets[index].print, local: localPrint,
                           destinationInstance: hosts.instanceID(of: local.id)) {
                        syncedCopies[sourceKey] = checkpoint
                        LocalSyncRecord.checkpoint(checkpoint, key: sourceKey)
                    }
                } else {
                    let failure = "\(targets[index].print.filename): \(result.error ?? "save failed")"
                    failures.append(failure)
                    if result.repeatAcknowledgable { recordMirrorIssue(failure, target: targets[index], local: local) }
                }
                if completed % 10 == 0 || completed == work.count {
                    localSaveProgress = "Checked \(completed.formatted()) of \(work.count.formatted()) · Copied \(transferred.formatted()) · Already here \(alreadyLocal.formatted()) · Issues \(failures.count.formatted())"
                }
                if next < work.count && !Task.isCancelled && !localSaveStopRequested {
                    let item = work[next]
                    next += 1
                    group.addTask {
                        await item.perform(to: destination, knownPrints: localByName,
                                           pending: pendingOrganization, destinationInstance: destinationInstance)
                    }
                }
            }
        }
        if Task.isCancelled || localSaveStopRequested {
            failures.append("Save stopped. Remaining prints were not transferred.")
        }

        return MirrorCopies(successful: successfulCopies, records: syncedCopies,
            assignments: MirrorAssignments(collectionNames: collectionNames, collectionFiles: collectionFiles,
                titles: titleAssignments, favorites: favoriteFiles, tags: tagFiles),
            transferred: transferred, alreadyLocal: alreadyLocal, failures: failures)
    }

    private func recordMirrorIssue(_ message: String, target: MirrorTarget, local: MoldHost) {
        guard targetVersioned(target.print) else { return }
        let context = [target.originContext,
                       target.print.mediaVersion ?? "", LocalSyncRecord.recipe(target.print) ?? "",
                       local.baseURL.absoluteString, hosts.instanceID(of: local.id) ?? "", message].joined(separator: "\n")
        let key = SHA256.hash(data: Data(context.utf8)).map { String(format: "%02x", $0) }.joined()
        localSaveIssueKeys[message] = Array(Set((localSaveIssueKeys[message]?.split(separator: ":").map(String.init) ?? []) + [key])).sorted().joined(separator: ":")
    }

    private func organizeMirrorCopies(_ assignments: MirrorAssignments,
                                       destination: any MoldBackend) async -> MirrorOrganization {
        let collectionNames = assignments.collectionNames
        let collectionFiles = assignments.collectionFiles
        let titleAssignments = assignments.titles
        let favoriteFiles = assignments.favorites
        let tagFiles = assignments.tags
        var failures: [String] = []
        var organizationFailedFiles: Set<String> = []
        localSaveProgress = "Organizing copied prints and collections…"
        // A host resolves a collection name to its own slug and creates the
        // local copy once. Never send the source machine's collection ID.
        for slug in collectionFiles.keys.sorted() {
            guard let name = collectionNames[slug],
                  let filenames = collectionFiles[slug], !filenames.isEmpty else { continue }
            do {
                try await destination.mutate(GalleryBulkMutation(
                    filenames: Array(Set(filenames)).sorted(), addToCollection: .named(name)))
            } catch {
                failures.append("Collection “\(name)”: \(error.sentence)")
                organizationFailedFiles.formUnion(filenames)
            }
        }
        if !titleAssignments.isEmpty {
            do {
                try await destination.mutate(GalleryBulkMutation(
                    filenames: [], titles: titleAssignments))
            } catch {
                failures.append("Print titles: \(error.sentence)")
                organizationFailedFiles.formUnion(titleAssignments.map(\.filename))
            }
        }
        if !favoriteFiles.isEmpty {
            do {
                try await destination.mutate(GalleryBulkMutation(
                    filenames: Array(Set(favoriteFiles)).sorted(), favorite: true))
            } catch {
                failures.append("Favourites: \(error.sentence)")
                organizationFailedFiles.formUnion(favoriteFiles)
            }
        }
        for tag in tagFiles.keys.sorted() {
            do {
                try await destination.mutate(GalleryBulkMutation(
                    filenames: Array(Set(tagFiles[tag] ?? [])).sorted(), addTags: [tag]))
            } catch {
                failures.append("Tag “\(tag)”: \(error.sentence)")
                organizationFailedFiles.formUnion(tagFiles[tag] ?? [])
            }
        }

        return MirrorOrganization(failedFiles: organizationFailedFiles, failures: failures)
    }

    private func verifyMirrorCopies(sources: MirrorSources, prepared: MirrorDestination,
                                     plan: MirrorWorkPlan, copies: MirrorCopies,
                                     organization: MirrorOrganization) async -> [String] {
        let work = plan.items
        let destination = prepared.backend
        let local = prepared.host
        let sourceCheckpoints = sources.checkpoints
        let sourceMachines = sources.machines
        let successfulCopies = copies.successful
        let organizationFailedFiles = organization.failedFiles
        let unresolvedCollections = sources.unresolvedCollections
        let failedCollectionSlugs = prepared.failedCollectionSlugs
        let pendingOrganization = prepared.pendingOrganization
        var syncedCopies = copies.records
        var failures: [String] = []
        localSaveProgress = "Verifying copied prints and retained inputs…"
        let verifiedDestinationCheckpoint = try? await destination.gallerySyncCheckpoint()
        var checkpointFailures: Set<String> = []
        let checkedKeys = Set(work.map { LocalSyncRecord.key(hostID: $0.target.hostID, filename: $0.target.print.filename) })
        if verifiedDestinationCheckpoint != nil {
            // Capture evidence BEFORE checking bindings. A foreign writer
            // changing a binding after copy must not seal a stale success.
            for copy in successfulCopies where checkedKeys.contains(LocalSyncRecord.key(hostID: copy.target.hostID, filename: copy.target.print.filename)) {
                let target = copy.target, filename = copy.filename
                if Task.isCancelled || localSaveStopRequested { checkpointFailures.insert(filename); continue }
                let result = await repairCachedSources(target, to: destination, as: filename)
                if result.filename == nil {
                    checkpointFailures.insert(filename)
                    failures.append("\(target.print.filename): \(result.error ?? "copy verification failed")")
                }
            }
        }
        let finalDestinationCheckpoint = try? await destination.gallerySyncCheckpoint()
        var stableSourceCheckpoints: [MoldHost.ID: GallerySyncCheckpoint] = [:]
        for (hostID, before) in sourceCheckpoints {
            if let backend = sourceMachines[hostID]?.1,
               let after = try? await backend.gallerySyncCheckpoint(),
               after.instanceId == before.instanceId {
                stableSourceCheckpoints[hostID] = after
            }
        }
        let refreshed = Dictionary((perHost[local.id] ?? []).map {
            ($0.print.filename, $0.print)
        }, uniquingKeysWith: { first, _ in first })
        for copy in successfulCopies {
            let target = copy.target, filename = copy.filename
            let sourceKey = LocalSyncRecord.key(hostID: target.hostID,
                                                filename: target.print.filename)
            guard checkedKeys.contains(sourceKey),
                  !organizationFailedFiles.contains(filename),
                  !unresolvedCollections.contains(sourceKey),
                  !target.collections.contains(where: {
                      failedCollectionSlugs.contains($0.slug)
                  }) else { continue }
            guard let localPrint = refreshed[filename],
                  var record = LocalSyncRecord(source: target.print, local: localPrint, destinationInstance: hosts.instanceID(of: local.id))
            else { continue }
            if let before = sourceCheckpoints[target.hostID],
               let after = stableSourceCheckpoints[target.hostID],
               let token = before.revisions[target.print.filename],
               after.revisions[target.print.filename] == token {
                record.sourceCheckpoint = before.instanceId + ":" + token
            }
            if !checkpointFailures.contains(filename),
               await target.isCurrent(),
               let checkpoint = finalDestinationCheckpoint,
               let verified = verifiedDestinationCheckpoint,
               checkpoint.instanceId == verified.instanceId,
               let token = checkpoint.revisions[filename],
               verified.revisions[filename] == token {
                record.destinationCheckpoint = checkpoint.instanceId + ":" + token
            }
            record.sourceRoute = target.originContext
            record.destinationRoute = local.baseURL.absoluteString
            syncedCopies[sourceKey] = record
            pendingOrganization?.clear(sourceKey)
            LocalSyncRecord.checkpoint(record, key: sourceKey)
        }
        LocalSyncRecord.save(syncedCopies)
        return failures
    }

    private func publishMirrorReport(selectedCount: Int, syncAll: Bool, sources: MirrorSources,
                                     prepared: MirrorDestination, plan: MirrorWorkPlan,
                                     copies: MirrorCopies, failures: [String]) {
        let transferred = copies.transferred
        let alreadyLocal = copies.alreadyLocal
        let keptRemoved = plan.keptRemoved
        let createdCollections = prepared.createdCollections
        let skipped = syncAll ? 0 : selectedCount - sources.candidates.count
        var summary = "Copied \(transferred) \(transferred == 1 ? "print" : "prints") to This Mac’s Library."
        if alreadyLocal > 0 {
            summary += alreadyLocal == 1 ? " 1 was already here." : " \(alreadyLocal) were already here."
        }
        summary += Self.legacyInputNotice(count: localSaveNotices.count)
        if createdCollections > 0 {
            summary += " Created \(createdCollections) \(createdCollections == 1 ? "collection" : "collections")."
        }
        let unsaved = sources.candidates.count - transferred - alreadyLocal - keptRemoved
        if keptRemoved > 0 { summary += " \(keptRemoved) copies were kept removed from This Mac; use Save to copy them again deliberately." }
        if unsaved > 0 { summary += " \(unsaved) were not copied." }
        if skipped > 0 { summary += " Skipped \(skipped) local or unsupported prints." }
        if !failures.isEmpty { summary += " \(failures.count) issues need attention." }
        localSaveReport = summary
        localSaveFailures = failures
        localSaveAlertPresented = failures.contains(where: isUnacknowledgedSyncIssue)
    }

}

@MainActor
extension LibraryStore {
    /// See `LocalSyncRecord.links`.
    func syncLinks() -> [PrintID: PrintID] {
        LocalSyncRecord.links(localHost: MoldEngine.localHostID)
    }
}
