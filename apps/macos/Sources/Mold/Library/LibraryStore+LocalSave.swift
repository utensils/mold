import CryptoKit
import Foundation
import MoldClient

struct MirrorTarget: Sendable {
    let hostID: MoldHost.ID
    let print: GalleryPrint
    let source: any MoldBackend
    let originContext: String
    let isCurrent: @MainActor @Sendable () -> Bool
    let collections: [(slug: String, name: String)]
    var privacyOnly = false

    var key: String { LocalSyncRecord.key(hostID: hostID, filename: print.filename) }
}

struct MirrorResult: Sendable {
    let filename: String?
    let alreadyLocal: Bool
    let error: String?
    var sourceMedia: RetainedSourceMedia.MirrorResult = .complete
    var repeatAcknowledgable = false
    /// This run wrote an output receipt for it (even if it then failed).
    var receipted = false
}

/// Each stage receives the captured authority it needs and returns its own result.
/// Only LibraryStore publishes progress, issues and the final report.
enum MirrorDiscovery {
    case ready(MirrorSources)
    case stopped([String])
}

/// One source machine as discovery read it: its replica at one revision.
struct MirrorSourceHost {
    let host: MoldHost
    let backend: any MoldBackend
    let originContext: String
    let view: ReplicaMirrorView
    /// Live names this Mac can hold, sorted.
    let candidates: [String]
    /// Names changed since this source's persisted mirror cursor; nil when
    /// the replica's log cannot say (first run, relaunch, route change).
    let changed: Set<String>?
    let collections: [String: Collection]?
}

struct MirrorSources {
    /// Save: the selected entries this Mac can hold. Sync All: empty.
    let selected: [LibraryEntry]
    let hosts: [MoldHost.ID: MirrorSourceHost]
    let hostErrors: [MoldHost.ID: String]
    let failures: [String]

    var candidateCount: Int {
        selected.isEmpty ? hosts.values.reduce(0) { $0 + $1.candidates.count } : selected.count
    }
}

enum MirrorPreparation {
    case ready(MirrorDestination)
    case unavailable(String)
}

struct MirrorDestination {
    let host: MoldHost
    let backend: any MoldBackend
    let instance: String?
    let recordScope: String?
    let view: ReplicaMirrorView
    /// Names changed on This Mac since the persisted destination cursor;
    /// nil when the log cannot say.
    let changed: Set<String>?
    let bucket: LibrarySyncRecords.Bucket
    let receipts: MirrorReceipts?
    let collectionNames: [String: String]
    let createdCollections: Int
    let failedCollectionSlugs: Set<String>
    let failures: [String]

    var records: [String: LocalSyncRecord] { bucket.records }
}

/// Where a lane writes its durable receipts: the destination's record bucket.
struct MirrorReceipts: Sendable {
    let records: LibrarySyncRecords
    let instance: String?
    var destinationInstance: String? = nil
    var destinationHost: UUID? = nil
    var destinationRoute: String? = nil

    func markPending(_ key: String, _ filename: String) async throws {
        try await records.markPending(key, filename: filename, instance: instance)
    }

    func receipt(_ record: LocalSyncRecord, key: String, sourceRoute: String? = nil) async throws {
        var bound = record
        bound.destinationHost = destinationHost
        bound.sourceRoute = sourceRoute ?? bound.sourceRoute
        bound.destinationRoute = destinationRoute ?? bound.destinationRoute
        if let previous = await records.records(instance)[key], previous.destinationFilename == bound.destinationFilename {
            bound.privacyPending = bound.privacyPending ?? previous.privacyPending
            bound.initialOrganizationPending = bound.initialOrganizationPending ?? previous.initialOrganizationPending
            bound.organization = bound.organization ?? previous.organization
            bound.organizationIntent = bound.organizationIntent ?? previous.organizationIntent
            bound.originKey = bound.originKey ?? previous.originKey
        }
        try await records.checkpoint(bound, key: key, instance: instance)
    }
}

struct MirrorCopy {
    let target: MirrorTarget
    let filename: String
    var newCopy = false
}

struct MirrorWorkPlan {
    let targets: [MirrorTarget]
    let items: [MirrorWorkItem]
    let cachedCopies: [MirrorCopy]
    let keptRemoved: Int
    /// Unconsidered prints already here (cursor-scoped plans only).
    let unchanged: Int
    /// Every discovered machine was planned in full.
    let plannedEverything: Bool
    /// Every source key this plan looked at.
    let considered: Set<String>
    /// Considered keys that need no further work: cached, or kept removed.
    let settled: Set<String>
    let keptRemovedKeys: Set<String>
    let unresolvedCollections: Set<String>
    let sourceCheckpoints: [MoldHost.ID: GallerySyncCheckpoint]
    /// Records whose source and copy are both gone for good.
    let tombstones: Set<String>
    let failures: [String]
}

struct MirrorWorkItem: Sendable {
    let index: Int
    let target: MirrorTarget
    let filename: String
    let occupied: Bool
    let cachedFilename: String?

    func perform(to destination: any MoldBackend, knownPrints: ReplicaMirrorView?,
                 receipts: MirrorReceipts?) async -> (Int, MirrorResult) {
        if let cachedFilename {
            return (index, await repairCachedSources(target, to: destination, as: cachedFilename))
        }
        return (index, await mirror(target, to: destination, as: filename, occupied: occupied,
                                   knownPrints: knownPrints, receipts: receipts))
    }
}

struct MirrorAssignments {
    let collectionNames: [String: String]
    let collectionFiles: [String: [String]]
    let titles: [GalleryTitleAssignment]
    let favorites: [String]
    let tags: [String: [String]]
}

struct MirrorCopies {
    let successful: [MirrorCopy]
    /// Source keys whose output receipt this run wrote.
    let receipted: Set<String>
    let assignments: MirrorAssignments
    let transferred: Int
    let alreadyLocal: Int
    let failures: [String]
}

struct MirrorOrganization {
    var failedFiles: Set<String>
    var failures: [String]
    var states: [String: LibraryOrganizationState] = [:]
    var expected: [String: LibraryOrganizationSnapshot] = [:]
}

func targetVersioned(_ print: GalleryPrint) -> Bool {
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

func sameDescriptor(_ source: GalleryPrint, _ local: GalleryPrint) -> Bool {
    source.timestamp == local.timestamp
        && RetainedSourceMedia.mirrorMetadataMatches(source.canonicalMetadataJSON, local.canonicalMetadataJSON)
        && (source.metadataSynthetic ?? false) == (local.metadataSynthetic ?? false)
}

func boundedMirrorName(_ filename: String, marker: String) -> String {
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

private let routeChanged = "The machine route changed. Refresh the Library before copying."

func repairCachedSources(_ target: MirrorTarget, to destination: any MoldBackend,
                         as filename: String) async -> MirrorResult {
    do {
        guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
        let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
            from: target.source, to: destination, as: filename, isCurrent: { await target.isCurrent() })
        return MirrorResult(filename: filename, alreadyLocal: true, error: nil, sourceMedia: sourceMedia)
    } catch {
        return MirrorResult(filename: nil, alreadyLocal: false, error: error.sentence, repeatAcknowledgable: canAcknowledgeSyncError(error))
    }
}

private func mirror(_ target: MirrorTarget, to destination: any MoldBackend,
                    as requestedName: String, occupied: Bool,
                    knownPrints: ReplicaMirrorView?, receipts: MirrorReceipts?) async -> MirrorResult {
    var receipted = false
    do {
        guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
        try checkSyncStagingSpace(for: target.print)
        let sourceIdentity = try await RetainedSourceMedia.preflightMirror(
            for: target.print.filename, metadata: target.print.metadata, from: target.source, to: destination)
        guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
        let file = try await target.source.mediaFile(target.print.filename, trashed: false)
        defer { try? FileManager.default.removeItem(at: file) }
        if occupied {
            let existing = try await destination.mediaFile(requestedName, trashed: false)
            defer { try? FileManager.default.removeItem(at: existing) }
            let equal: Bool
            if let local = knownPrints?.live(requestedName), sameDescriptor(target.print, local) {
                equal = try await LibraryMirrorFiles.equal(file, existing)
            } else {
                equal = false
            }
            if equal {
                guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
                if let receipts, let receipt = LocalSyncRecord(receipt: target.print, from: target.hostID,
                    as: requestedName, destinationInstance: receipts.destinationInstance) {
                    try await receipts.receipt(receipt, key: target.key, sourceRoute: target.originContext)
                    receipted = true
                }
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: requestedName, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file, isCurrent: { await target.isCurrent() })
                return MirrorResult(filename: requestedName, alreadyLocal: true, error: nil, sourceMedia: sourceMedia, receipted: receipted)
            }
        }
        let filename = occupied ? try await collisionName(for: target, file: file) : requestedName
        if occupied, let local = knownPrints?.live(filename) {
            let existing = try await destination.mediaFile(filename, trashed: false)
            defer { try? FileManager.default.removeItem(at: existing) }
            let equal = sameDescriptor(target.print, local)
                ? try await LibraryMirrorFiles.equal(file, existing) : false
            if equal {
                guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
                if let receipts, let receipt = LocalSyncRecord(receipt: target.print, from: target.hostID,
                    as: filename, destinationInstance: receipts.destinationInstance) {
                    try await receipts.receipt(receipt, key: target.key, sourceRoute: target.originContext)
                    receipted = true
                }
                let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
                    from: target.source, to: destination, as: filename, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file, isCurrent: { await target.isCurrent() })
                return MirrorResult(filename: filename, alreadyLocal: true, error: nil, sourceMedia: sourceMedia, receipted: receipted)
            }
        }
        guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
        let item = try await LibraryMirrorFiles.importDescriptor(for: target.print, file: file)
        guard await target.isCurrent() else { throw MoldClientError.unreachable(routeChanged) }
        try await receipts?.markPending(target.key, filename)
        // The server may commit the file before its response is lost; the
        // pending marker stays for the next fresh listing either way.
        let imported = try await destination.importPrint(item, as: filename)
        if imported != filename { try await receipts?.markPending(target.key, imported) }
        // The import's answer proves the copy's name: that is the durable
        // output receipt, written before the retained-input transfer. No
        // checkpoint tokens means a resume MUST repair retained inputs
        // before declaring the copy complete. A receipt that cannot be made
        // durable fails the item here, before any retained input moves.
        if let receipts, var receipt = LocalSyncRecord(receipt: target.print, from: target.hostID, as: imported,
                                                       destinationInstance: receipts.destinationInstance) {
            receipt.initialOrganizationPending = true
            try await receipts.receipt(receipt, key: target.key, sourceRoute: target.originContext)
            receipted = true
        }
        let sourceMedia = try await RetainedSourceMedia.mirrorSources(for: target.print.filename, metadata: target.print.metadata,
            from: target.source, to: destination, as: imported, expectedSourceArchiveIdentity: sourceIdentity, downloadedOutput: file, isCurrent: { await target.isCurrent() })
        return MirrorResult(filename: imported, alreadyLocal: false, error: nil, sourceMedia: sourceMedia, receipted: true)
    } catch {
        return MirrorResult(filename: nil, alreadyLocal: false,
                            error: error.sentence, repeatAcknowledgable: canAcknowledgeSyncError(error), receipted: receipted)
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
        entry.hostID != MoldEngine.localHostID && syncableName(entry.print.filename)
    }

    /// Whether This Mac's Library can hold a file of this kind. Pure, so
    /// discovery filters a machine's rows off the main actor.
    nonisolated static func syncableName(_ filename: String) -> Bool {
        guard let dot = filename.lastIndex(of: ".") else { return false }
        return ["png", "jpg", "jpeg", "webp", "gif", "apng", "mp4", "wav", "glb"]
            .contains(filename[filename.index(after: dot)...].lowercased())
    }

    func saveLocally(_ selection: [LibraryEntry]) async {
        await mirrorLocally(selection, syncAll: false)
        syncSession.runFinished(in: self)
    }

    /// `fullPlan`: an explicit press plans every print, skipping only on
    /// matching source and destination checkpoint evidence; Keep in Sync's
    /// automatic runs pass false and plan from each machine's cursor.
    func copy(_ selection: [LibraryEntry], to destination: MoldHost) async {
        await mirrorLocally(selection, syncAll: false, destination: destination)
        syncSession.runFinished(in: self)
    }

    func syncAll(to destination: MoldHost, fullPlan: Bool = true, automatic: Bool = false) async {
        await mirrorLocally([], syncAll: true, fullPlan: fullPlan, destination: destination, automatic: automatic)
        syncSession.runFinished(in: self)
    }

    func syncAllLocally(fullPlan: Bool = true) async {
        await mirrorLocally([], syncAll: true, fullPlan: fullPlan)
        syncSession.runFinished(in: self)
    }

    /// At most ~10 updates a second: every published string re-renders the
    /// status banner. Stage changes pass `force`.
    func publishSyncProgress(_ text: String, force: Bool = false) {
        let now = ContinuousClock.now
        guard force || localSaveProgressPublished.map({ now - $0 >= .milliseconds(100) }) ?? true else { return }
        localSaveProgressPublished = now
        localSaveProgress = text
    }

    var syncStopped: Bool { Task.isCancelled || localSaveStopRequested }

    private func mirrorLocally(_ selection: [LibraryEntry], syncAll: Bool, fullPlan: Bool = true, destination: MoldHost? = nil, automatic: Bool = false) async {
        guard !localSaveRunning else { return }
        localSaveRunning = true
        defer { localSaveRunning = false }
        localSaveIssueKeys = [:]
        localSaveNotices = []
        localSaveAlertPresented = false
        defer { if automatic { localSaveAlertPresented = false } }
        let visibilityGeneration = collectionVisibility.generation
        defer { localSaveTask = nil }
        guard let local = destination ?? hosts.host(MoldEngine.localHostID), hosts.isUp(local),
              hosts.host(local.id) == local else {
            localSaveReport = destination == nil
                ? "Start This Mac’s engine to save remote prints in its Library."
                : "Connect to the destination machine before copying prints."
            localSaveFailures = []
            localSaveAlertPresented = true
            return
        }
        syncDestinationHost = local.id
        defer { syncDestinationHost = nil }
        let eligibleSelection = destination == nil && !syncAll ? selection.filter(Self.canSaveLocally) : selection
        let candidates = eligibleSelection.compactMap { entry in
            entry.everyCopy.first { $0.hostID != local.id && Self.syncableName($0.print.filename) }
        }
        guard syncAll || !candidates.isEmpty else { return }
        // Never on an empty stand-in for records this build cannot read.
        if syncAll, let blocked = syncRecords.blockedReason {
            localSaveReport = "Sync All is paused."
            localSaveFailures = [blocked]
            localSaveAlertPresented = true
            return
        }
        localSaveSuppressImports = true
        publishSyncProgress(syncAll ? "Finding remote prints and collections…"
            : "Preparing \(candidates.count.formatted()) prints…", force: true)
        defer {
            localSaveProgress = nil
            localSaveProgressPublished = nil
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
        let plan = await planMirrorWork(sources, destination: prepared, syncAll: syncAll, fullPlan: fullPlan)
        let copies = await copyMirrorWork(plan, sources: sources, prepared: prepared, syncAll: syncAll)
        let organization = await reconcileMirrorOrganization(copies, sources: sources, prepared: prepared, explicitCopy: !syncAll)
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
        publishSyncProgress("Refreshing \(prepared.host.name)’s Library…", force: true)
        await refresh(on: prepared.host.id)
        await reconcileCollectionVisibility()
        do {
            failures += await verifyMirrorCopies(sources: sources, prepared: prepared,
                plan: plan, copies: copies, organization: organization)
        }
        publishMirrorReport(selectedCount: selection.count, syncAll: syncAll,
            sources: sources, prepared: prepared, plan: plan, copies: copies, failures: failures)
    }

    private func copyMirrorWork(_ plan: MirrorWorkPlan, sources: MirrorSources,
                                prepared: MirrorDestination, syncAll: Bool) async -> MirrorCopies {
        let targets = plan.targets
        let work = plan.items
        let local = prepared.host
        let destination = prepared.backend
        let knownPrints = prepared.view
        let localNames = prepared.collectionNames
        let receipts = prepared.receipts
        let syncedCopies = prepared.records
        var failures = sources.failures + prepared.failures + plan.failures
        var collectionNames: [String: String] = [:]
        var collectionFiles: [String: [String]] = [:]
        var titleAssignments: [GalleryTitleAssignment] = []
        var favoriteFiles: [String] = []
        var tagFiles: [String: [String]] = [:]
        var successfulCopies: [MirrorCopy] = []
        var receipted: Set<String> = []
        var alreadyLocal = plan.unchanged
        var transferred = 0
        var completed = 0
        func record(_ target: MirrorTarget, as filename: String, organize: Bool) {
            successfulCopies.append(MirrorCopy(target: target, filename: filename, newCopy: (organize && syncedCopies[target.key] == nil) || syncedCopies[target.key]?.initialOrganizationPending == true))
            let existing = syncedCopies[target.key]
            // A receipt is not a finished copy: its organization may never
            // have landed, so file its memberships again.
            let fileMembership = !syncAll || existing == nil || existing?.provisional == true
                || existing?.destinationInstance != prepared.instance
                || syncRecords.pending(target.key, instance: prepared.recordScope) != nil
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
        publishSyncProgress("Checking \(work.count.formatted()) changed prints · Already here \(alreadyLocal.formatted())", force: true)
        await withTaskGroup(of: (Int, MirrorResult).self) { group in
            let free = (try? FileManager.default.attributesOfFileSystem(
                forPath: FileManager.default.temporaryDirectory.path)[.systemFreeSize]
                as? NSNumber)?.int64Value ?? 0
            let largeTransfer = work.contains {
                ($0.target.print.sizeBytes ?? 0) > 512 * 1_024 * 1_024
            }
            let lanes = largeTransfer || free < 5 * 1_024 * 1_024 * 1_024 ? 1 : 3
            func refill(stopped: Bool) {
                guard next < work.count, !stopped else { return }
                let item = work[next]
                next += 1
                group.addTask { await item.perform(to: destination, knownPrints: knownPrints, receipts: receipts) }
            }
            for _ in 0..<min(lanes, work.count) { refill(stopped: syncStopped) }
            for await (index, result) in group {
                // The next lane starts before this result's bookkeeping.
                refill(stopped: syncStopped)
                completed += 1
                let target = targets[index]
                if result.receipted { receipted.insert(target.key) }
                if let name = result.filename {
                    if result.sourceMedia == .legacyInputsUnavailable {
                        let notice = "\(hosts.name(of: target.hostID) ?? target.hostID.uuidString) · \(target.print.filename): Original inputs are unavailable; the finished file is saved."
                        localSaveNotices.append(notice)
                        recordMirrorIssue(notice, target: target, local: local)
                    }
                    if result.alreadyLocal { alreadyLocal += 1 }
                    else { transferred += 1 }
                    let pendingMatch = syncRecords.pending(target.key, instance: prepared.recordScope) == name
                    record(target, as: name, organize: !result.alreadyLocal || pendingMatch)
                    // A response-proven copy gets its own durable receipt
                    // before organization, from what is already known: no
                    // per-print listing read, and never ahead of a refill.
                    if let receipts, syncedCopies[target.key]?.destinationFilename != name,
                       let receipt = LocalSyncRecord(receipt: target.print, from: target.hostID, as: name,
                                                     destinationInstance: prepared.instance) {
                        do {
                            try await receipts.receipt(receipt, key: target.key, sourceRoute: target.originContext)
                            receipted.insert(target.key)
                        } catch {
                            failures.append("\(target.print.filename): its sync receipt could not be saved: \(error.sentence)")
                        }
                    }
                } else {
                    let failure = "\(target.print.filename): \(result.error ?? "save failed")"
                    failures.append(failure)
                    if result.repeatAcknowledgable { recordMirrorIssue(failure, target: target, local: local) }
                }
                publishSyncProgress("Checked \(completed.formatted()) of \(work.count.formatted()) · Copied \(transferred.formatted()) · Already here \(alreadyLocal.formatted()) · Issues \(failures.count.formatted())",
                                    force: completed == work.count)
            }
        }
        if syncStopped {
            failures.append("Save stopped. Remaining prints were not transferred.")
        }

        return MirrorCopies(successful: successfulCopies, receipted: receipted,
            assignments: MirrorAssignments(collectionNames: collectionNames, collectionFiles: collectionFiles,
                titles: titleAssignments, favorites: favoriteFiles, tags: tagFiles),
            transferred: transferred, alreadyLocal: alreadyLocal, failures: failures)
    }

    func recordMirrorIssue(_ message: String, target: MirrorTarget, local: MoldHost) {
        guard targetVersioned(target.print) else { return }
        let context = [target.originContext,
                       target.print.mediaVersion ?? "", LocalSyncRecord.recipe(target.print, host: target.hostID) ?? "",
                       local.baseURL.absoluteString, hosts.instanceID(of: local.id) ?? "", message].joined(separator: "\n")
        let key = SHA256.hash(data: Data(context.utf8)).map { String(format: "%02x", $0) }.joined()
        localSaveIssueKeys[message] = Array(Set((localSaveIssueKeys[message]?.split(separator: ":").map(String.init) ?? []) + [key])).sorted().joined(separator: ":")
    }

    private func publishMirrorReport(selectedCount: Int, syncAll: Bool, sources: MirrorSources,
                                     prepared: MirrorDestination, plan: MirrorWorkPlan,
                                     copies: MirrorCopies, failures: [String]) {
        let transferred = copies.transferred
        let alreadyLocal = copies.alreadyLocal
        let keptRemoved = plan.keptRemoved
        let createdCollections = prepared.createdCollections
        let candidateCount = sources.candidateCount
        let skipped = syncAll ? 0 : selectedCount - candidateCount
        var summary = "Copied \(transferred) \(transferred == 1 ? "print" : "prints") to \(prepared.host.name)’s Library."
        if alreadyLocal > 0 {
            summary += alreadyLocal == 1 ? " 1 was already here." : " \(alreadyLocal) were already here."
        }
        summary += Self.legacyInputNotice(count: localSaveNotices.count)
        if createdCollections > 0 {
            summary += " Created \(createdCollections) \(createdCollections == 1 ? "collection" : "collections")."
        }
        let unsaved = candidateCount - transferred - alreadyLocal - keptRemoved
        if keptRemoved > 0 { summary += " \(keptRemoved) copies were kept removed from \(prepared.host.name); use Copy to copy them again deliberately." }
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
    /// See `LibrarySyncRecords.links`.
    func syncLinks() -> [PrintID: PrintID] {
        syncRecords.links(localHost: MoldEngine.localHostID,
                          instance: hosts.instanceID(of: MoldEngine.localHostID), authority: syncAuthority)
    }

    /// Conservative hide-only receipt links never enter the ordinary merge graph.
    func syncPrivacyLinks() -> [PrintID: PrintID] {
        guard !hiddenCollectionIDs.isEmpty else { return [:] }
        _ = syncRecords.privacyRevision
        return syncRecords.privacyLinks(localHost: MoldEngine.localHostID,
            instance: hosts.instanceID(of: MoldEngine.localHostID), authority: syncAuthority)
    }

    /// The records' document, now. For quitting.
    func flushSyncRecords() async {
        await syncRecords.flush()
    }
}
