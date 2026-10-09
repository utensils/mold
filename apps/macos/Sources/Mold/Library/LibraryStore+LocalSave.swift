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

    static func load() -> [String: Self] {
        guard let data = AppStorageSuite.defaults.data(forKey: syncRecordStorageKey),
              let records = try? MoldJSON.decoder.decode([String: Self].self, from: data)
        else { return [:] }
        return records
    }

    static func save(_ records: [String: Self]) {
        guard let data = try? MoldJSON.encoder.encode(records) else { return }
        AppStorageSuite.defaults.set(data, forKey: syncRecordStorageKey)
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

private func filesEqual(_ first: URL, _ second: URL) throws -> Bool {
    let manager = FileManager.default
    let firstSize = try manager.attributesOfItem(atPath: first.path)[.size] as? NSNumber
    let secondSize = try manager.attributesOfItem(atPath: second.path)[.size] as? NSNumber
    guard firstSize == secondSize else { return false }
    let left = try FileHandle(forReadingFrom: first)
    let right = try FileHandle(forReadingFrom: second)
    defer { try? left.close(); try? right.close() }
    while true {
        let a = try left.read(upToCount: 1_024 * 1_024) ?? Data()
        let b = try right.read(upToCount: 1_024 * 1_024) ?? Data()
        guard a == b else { return false }
        if a.isEmpty { return true }
    }
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

private func collisionName(for target: MirrorTarget, file: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var digest = SHA256()
    while let chunk = try handle.read(upToCount: 1_024 * 1_024), !chunk.isEmpty {
        digest.update(data: chunk)
    }
    let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
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
                    pending: PendingSyncOrganization?) async -> MirrorResult {
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
            let equal: Bool
            if let local = knownPrints[requestedName], sameDescriptor(target.print, local) {
                equal = try filesEqual(file, existing)
            } else {
                equal = false
            }
            try? FileManager.default.removeItem(at: existing)
            if equal {
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: requestedName, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file)
                return MirrorResult(filename: requestedName, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
            }
        }
        let filename = occupied ? try collisionName(for: target, file: file) : requestedName
        if occupied, let local = knownPrints[filename] {
            let existing = try await destination.mediaFile(filename, trashed: false)
            let equal = sameDescriptor(target.print, local)
                ? try filesEqual(file, existing) : false
            try? FileManager.default.removeItem(at: existing)
            if equal {
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: filename, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file)
                return MirrorResult(filename: filename, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
            }
        }
        guard await target.isCurrent() else { throw MoldClientError.unreachable("The machine route changed. Refresh the Library before copying.") }
        let item = try GalleryImport(mirroring: target.print, fileAt: file)
        let sourceKey = LocalSyncRecord.key(hostID: target.hostID,
                                            filename: target.print.filename)
        await pending?.mark(sourceKey, filename: filename)
        do {
            let imported = try await destination.importPrint(item, as: filename)
            if imported != filename { await pending?.mark(sourceKey, filename: imported) }
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

@MainActor
extension LibraryStore {
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
    }

    func syncAllLocally() async {
        await mirrorLocally([], syncAll: true)
    }

    private func mirrorLocally(_ selection: [LibraryEntry], syncAll: Bool) async {
        guard !localSaveRunning else { return }
        localSaveRunning = true
        defer { localSaveRunning = false }
        localSaveIssueKeys = [:]
        localSaveAlertPresented = false
        let visibilityGeneration = collectionVisibility.generation
        defer { localSaveTask = nil }
        var candidates = selection.filter(Self.canSaveLocally)
        guard syncAll || !candidates.isEmpty else { return }
        guard let local = hosts.host(MoldEngine.localHostID), hosts.isUp(local) else {
            localSaveReport = "Start This Mac’s engine to save remote prints in its Library."
            localSaveFailures = []
            localSaveAlertPresented = true
            return
        }
        localSaveProgress = syncAll ? "Finding remote prints and collections…"
            : "Preparing \(candidates.count.formatted()) prints…"
        defer {
            localSaveProgress = nil
            localSaveStopRequested = false
        }

        var failures: [String] = []
        var sourcePrints: [MoldHost.ID: [String: GalleryPrint]] = [:]
        var sourceMachines: [MoldHost.ID: (MoldHost, any MoldBackend)] = [:]
        var originContexts: [MoldHost.ID: String] = [:]
        var sourceCollections: [MoldHost.ID: [String: Collection]] = [:]
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
                guard case let .fresh(prints, _) = try await source.gallery(etag: nil) else {
                    throw MoldClientError.malformedResponse
                }
                guard hosts.host(hostID) == sourceHost else { throw MoldClientError.unreachable("The machine route changed while reading its Library.") }
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

        if Task.isCancelled || localSaveStopRequested {
            localSaveReport = "Sync stopped before any prints were copied."
            failures.append("Save stopped. Remaining prints were not transferred.")
            localSaveFailures = failures
            localSaveAlertPresented = true
            return
        }

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

        let destination = hosts.backend(for: local)
        let localPrints: [GalleryPrint]
        do {
            guard case let .fresh(prints, _) = try await destination.gallery(etag: nil) else {
                throw MoldClientError.malformedResponse
            }
            localPrints = prints
        } catch {
            localSaveReport = "Couldn’t read This Mac’s Library: \(error.sentence)"
            localSaveFailures = failures
            localSaveAlertPresented = true
            return
        }
        let localTrashNames: Set<String>
        if syncAll {
            do {
                guard case let .fresh(prints, _) = try await destination.trashedPrints(etag: nil) else {
                    throw MoldClientError.malformedResponse
                }
                localTrashNames = Set(prints.map(\.filename))
            } catch {
                localSaveReport = "Couldn’t read This Mac’s Trash safely: \(error.sentence)"
                localSaveFailures = failures
                localSaveAlertPresented = true
                return
            }
        } else { localTrashNames = [] }
        let localByName = Dictionary(localPrints.map { ($0.filename, $0) },
                                     uniquingKeysWith: { first, _ in first })
        var syncedCopies = syncAll ? LocalSyncRecord.load() : [:]
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
        var collectionNames: [String: String] = [:]
        var collectionFiles: [String: [String]] = [:]
        var titleAssignments: [GalleryTitleAssignment] = []
        var favoriteFiles: [String] = []
        var tagFiles: [String: [String]] = [:]
        var successfulCopies: [(MirrorTarget, String)] = []
        var organizationFailedFiles: Set<String> = []
        var alreadyLocal = 0
        var transferred = 0
        var legacyInputsUnavailable = 0
        var completed = 0
        func record(_ target: MirrorTarget, as filename: String, organize: Bool) {
            if syncAll { successfulCopies.append((target, filename)) }
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

        var work: [(Int, MirrorTarget, String, Bool, String?)] = []
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
               pendingOrganization?.destination(for: sourceKey) == nil,
               let cached = syncedCopies[sourceKey],
               let existing = localByName[cached.destinationFilename],
               cached.matches(source: target.print, local: existing) {
                work.append((index, target, requestedName, true, existing.filename))
                continue
            }
            work.append((index, target, requestedName, localByName[requestedName] != nil, nil))
        }

        // Large clips can briefly occupy download and upload staging space.
        // Keep those batches serial; do the same when free space cannot
        // comfortably hold three 512 MiB transfers with three staged copies.
        var next = 0
        await withTaskGroup(of: (Int, MirrorResult).self) { group in
            let free = (try? FileManager.default.attributesOfFileSystem(
                forPath: FileManager.default.temporaryDirectory.path)[.systemFreeSize]
                as? NSNumber)?.int64Value ?? 0
            let largeTransfer = work.contains {
                ($0.1.print.sizeBytes ?? 0) > 512 * 1_024 * 1_024
            }
            let lanes = largeTransfer || free < 5 * 1_024 * 1_024 * 1_024 ? 1 : 3
            for _ in 0..<min(lanes, work.count) {
                let (index, target, filename, occupied, cached) = work[next]
                next += 1
                group.addTask {
                    if let cached { return (index, await repairCachedSources(target, to: destination, as: cached)) }
                    return (index, await mirror(target, to: destination,
                                                as: filename, occupied: occupied,
                                                knownPrints: localByName,
                                                pending: pendingOrganization))
                }
            }
            for await (index, result) in group {
                completed += 1
                if let name = result.filename {
                    if result.sourceMedia == .legacyInputsUnavailable { legacyInputsUnavailable += 1 }
                    if result.alreadyLocal { alreadyLocal += 1 }
                    else { transferred += 1 }
                    let sourceKey = LocalSyncRecord.key(hostID: targets[index].hostID,
                                                        filename: targets[index].print.filename)
                    let pendingMatch = pendingOrganization?.destination(for: sourceKey) == name
                    record(targets[index], as: name,
                           organize: !result.alreadyLocal || pendingMatch)
                } else {
                    let failure = "\(targets[index].print.filename): \(result.error ?? "save failed")"
                    failures.append(failure)
                    if result.repeatAcknowledgable, targetVersioned(targets[index].print) {
                        let target = targets[index]
                        let context = [target.originContext,
                                       target.print.mediaVersion ?? "",
                                       LocalSyncRecord.recipe(target.print) ?? "",
                                       local.baseURL.absoluteString,
                                       hosts.instanceID(of: local.id) ?? "", failure].joined(separator: "\n")
                        // Same text from two sources must acknowledge BOTH
                        // current identities, not overwrite one with another.
                        let key = SHA256.hash(data: Data(context.utf8)).map { String(format: "%02x", $0) }.joined()
                        localSaveIssueKeys[failure] = Array(Set((localSaveIssueKeys[failure]?.split(separator: ":").map(String.init) ?? []) + [key])).sorted().joined(separator: ":")
                    }
                }
                if completed % 10 == 0 || completed == targets.count {
                    localSaveProgress = "Saving \(completed.formatted()) of \(targets.count.formatted())…"
                }
                if next < work.count && !Task.isCancelled && !localSaveStopRequested {
                    let (index, target, filename, occupied, cached) = work[next]
                    next += 1
                    group.addTask {
                        if let cached { return (index, await repairCachedSources(target, to: destination, as: cached)) }
                        return (index, await mirror(target, to: destination,
                                                    as: filename, occupied: occupied,
                                                    knownPrints: localByName,
                                                    pending: pendingOrganization))
                    }
                }
            }
        }
        if Task.isCancelled || localSaveStopRequested {
            failures.append("Save stopped. Remaining prints were not transferred.")
        }

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

        if transferred > 0 || !collectionFiles.isEmpty || createdCollections > 0 {
            if let collections = try? await destination.collections() {
                collectionsPerHost[local.id] = collections
            }
        }
        // End suppression before the final read. An unrelated client may
        // import while collections load; the read then catches it, and later
        // events apply normally.
        localSaveProgress = nil
        await refresh(on: local.id)
        await reconcileCollectionVisibility()
        if syncAll {
            let refreshed = Dictionary((perHost[local.id] ?? []).map {
                ($0.print.filename, $0.print)
            }, uniquingKeysWith: { first, _ in first })
            for (target, filename) in successfulCopies {
                let sourceKey = LocalSyncRecord.key(hostID: target.hostID,
                                                    filename: target.print.filename)
                guard !organizationFailedFiles.contains(filename),
                      !unresolvedCollections.contains(sourceKey),
                      !target.collections.contains(where: {
                          failedCollectionSlugs.contains($0.slug)
                      }) else { continue }
                guard let localPrint = refreshed[filename],
                      let record = LocalSyncRecord(source: target.print, local: localPrint, destinationInstance: hosts.instanceID(of: local.id))
                else { continue }
                syncedCopies[sourceKey] = record
                pendingOrganization?.clear(sourceKey)
            }
            LocalSyncRecord.save(syncedCopies)
        }
        let skipped = syncAll ? 0 : selection.count - candidates.count
        var summary = "Copied \(transferred) \(transferred == 1 ? "print" : "prints") to This Mac’s Library."
        if alreadyLocal > 0 {
            summary += alreadyLocal == 1 ? " 1 was already here." : " \(alreadyLocal) were already here."
        }
        if legacyInputsUnavailable > 0 {
            summary += legacyInputsUnavailable == 1
                ? " Original inputs are unavailable on the source machine for 1 legacy print; its finished file is saved."
                : " Original inputs are unavailable on the source machine for \(legacyInputsUnavailable) legacy prints; their finished files are saved."
        }
        if createdCollections > 0 {
            summary += " Created \(createdCollections) \(createdCollections == 1 ? "collection" : "collections")."
        }
        let unsaved = candidates.count - transferred - alreadyLocal - keptRemoved
        if keptRemoved > 0 { summary += " \(keptRemoved) copies were kept removed from This Mac; use Save to copy them again deliberately." }
        if unsaved > 0 { summary += " \(unsaved) were not copied." }
        if skipped > 0 { summary += " Skipped \(skipped) local or unsupported prints." }
        if !failures.isEmpty { summary += " \(failures.count) issues need attention." }
        localSaveReport = summary
        localSaveFailures = failures
        localSaveAlertPresented = failures.contains { failure in
            guard let key = localSaveIssueKeys[failure] else { return true }
            return syncSession.hasNewIssues([key])
        }
    }

}

@MainActor
extension LibraryStore {
    /// See `LocalSyncRecord.links`.
    func syncLinks() -> [PrintID: PrintID] {
        LocalSyncRecord.links(localHost: MoldEngine.localHostID)
    }
}
