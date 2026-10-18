import CryptoKit
import Foundation
import MoldClient

@MainActor
extension LibraryStore {
    var syncAuthority: [UUID: String] {
        Dictionary(uniqueKeysWithValues: hosts.hosts.map { ($0.id, originContext($0.id)) })
    }

    /// The historical local bucket remains compatible; a remote destination
    /// never adopts local records and is scoped by route and installation.
    func recordScope(_ destination: MoldHost) -> String? {
        let instance = hosts.instanceID(of: destination.id)
        if destination.id == MoldEngine.localHostID { return instance }
        return [destination.id.uuidString, destination.baseURL.absoluteString, instance ?? ""].joined(separator: "\n")
    }

    func reconcileMirrorOrganization(_ copies: MirrorCopies, sources: MirrorSources, prepared: MirrorDestination,
                                     explicitCopy: Bool) async -> MirrorOrganization {
        var result = MirrorOrganization(failedFiles: [], failures: [])
        guard !copies.successful.isEmpty else { return result }
        let destination = prepared.backend
        let local = prepared.host
        let collections: [Collection]
        do { collections = try await destination.collections() }
        catch { return MirrorOrganization(failedFiles: Set(copies.successful.map(\.filename)), failures: ["Destination collections: \(error.sentence)"]) }
        let slugs = Dictionary(collections.map { ($0.id, $0.slug) }, uniquingKeysWith: { first, _ in first })
        let hidden = Set(collections.filter { $0.hidden == true }.map(\.slug))
        publishSyncProgress("Reconciling organization on \(local.name)…", force: true)
        // Fresh authoritative destination rows, after transfer. No per-print GETs.
        do { try await replica(for: local).pull(pullRequest(local, .live), using: destination) }
        catch { return MirrorOrganization(failedFiles: Set(copies.successful.map(\.filename)), failures: ["Destination organization: \(error.sentence)"]) }
        let view = await replica(for: local).mirrorView()
        var current = view.rows.mapValues { print in
            return LibraryOrganizationSnapshot(collections: Set(print.collectionList.compactMap { slugs[$0] }),
                tags: Set(print.tagList), title: print.title ?? "", favorite: print.isFavorite)
        }
        // Privacy memberships can live on a verified backup or Trash owner.
        // They contribute protection without becoming live transfer candidates.
        let components = syncLinks()
        var hiddenByComponent: [PrintID: Set<String>] = [:]
        for copy in copies.successful {
            let sourceID = PrintID(host: copy.target.hostID, filename: copy.target.print.filename)
            let component = components[sourceID] ?? sourceID
            let members = Set(syncRecords.verifiedCopies(of: sourceID, authority: syncAuthority) + [sourceID])
            for id in members {
                guard let sourceHost = sources.hosts[id.host], let row = sourceHost.view.rows[id.filename] else { continue }
                let memberships = row.collectionList.compactMap { sourceHost.collections?[$0] }
                    .filter { $0.hidden == true }.map(\.slug)
                hiddenByComponent[component, default: []].formUnion(memberships)
            }
        }
        var records = syncRecords.records(prepared.recordScope)
        var byFilename: [String: [String: LocalSyncRecord]] = [:]
        for (key, record) in records { byFilename[record.destinationFilename, default: [:]][key] = record }
        var plannedRoots: [String: Set<String>] = [:]
        for copy in copies.successful {
            let root = records[copy.target.key]?.originKey
                ?? syncRecords.verifiedOrigin(of: PrintID(host: copy.target.hostID, filename: copy.target.print.filename), authority: syncAuthority)
                ?? copy.target.key
            plannedRoots[copy.filename, default: []].insert(root)
        }
        for copy in copies.successful.sorted(by: { $0.target.key < $1.target.key }) {
            let target = copy.target, name = copy.filename, key = target.key
            do {
                try Task.checkCancellation()
                guard !syncStopped, target.isCurrent(), hosts.host(local.id) == local,
                      hosts.instanceID(of: local.id) == prepared.instance,
                      let observedDestination = current[name] else { throw MoldClientError.unreachable("The copy or its machine changed while reconciling organization.") }
                var before = observedDestination
                guard target.collections.count == target.print.collectionList.count else { throw MoldClientError.malformedResponse }
                var source = LibraryOrganizationSnapshot(collections: Set(target.collections.map(\.slug)),
                    tags: Set(target.print.tagList), title: target.print.title ?? "", favorite: target.print.isFavorite)
                var record = records[key] ?? LocalSyncRecord(receipt: target.print, from: target.hostID,
                    as: name, destinationInstance: prepared.instance)
                guard var journalRecord = record else { throw MoldClientError.malformedResponse }
                journalRecord.destinationHost = local.id
                if target.privacyOnly {
                    let hiddenMemberships = source.collections.intersection(hidden)
                    source = journalRecord.organization?.observed ?? before
                    source.collections.formUnion(hiddenMemberships)
                }
                let sourceID = PrintID(host: target.hostID, filename: target.print.filename)
                if !target.privacyOnly, let original = syncRecords.organizationAtOrigin(of: sourceID, authority: syncAuthority) {
                    source = original
                }
                source.collections.formUnion(hiddenByComponent[components[sourceID] ?? sourceID] ?? [])
                let root = journalRecord.originKey ?? syncRecords.verifiedOrigin(of: PrintID(host: target.hostID, filename: target.print.filename), authority: syncAuthority) ?? key
                journalRecord.originKey = root
                if journalRecord.organization == nil,
                   let inherited = byFilename[name]?.first(where: { $0.key != key && $0.value.destinationFilename == name
                       && ($0.value.originKey ?? $0.key) == root })?.value {
                    journalRecord.organization = inherited.organization
                }
                if let pending = journalRecord.organizationIntent {
                    try await applyOrganizationIntent(pending, filename: name, destination: destination,
                        collections: prepared.collectionNames, isCurrent: target.isCurrent)
                    try await replica(for: local).pull(pullRequest(local, .live), using: destination)
                    let fresh = await replica(for: local).mirrorView()
                    guard let row = fresh.live(name) else { throw MoldClientError.malformedResponse }
                    before = LibraryOrganizationSnapshot(collections: Set(row.collectionList.compactMap { slugs[$0] }),
                        tags: Set(row.tagList), title: row.title ?? "", favorite: row.isFavorite)
                    // An immutable receipt is replayed before considering a
                    // changed source; do not publish ownership on a mere reply.
                    // Diverging fields are destination edits; the policy
                    // below suppresses missing ordinary memberships and
                    // relinquishes scalar authority instead of overwriting.
                    journalRecord.organization = pending.state
                    journalRecord.organizationIntent = nil
                    try await syncRecords.checkpoint(journalRecord, key: key, instance: prepared.recordScope)
                }
                let contributors = (byFilename[name] ?? [:]).filter { $0.key != key && $0.value.destinationFilename == name
                    && ($0.value.originKey ?? $0.key) != root }
                let claims = LibraryOrganizationSnapshot(
                    collections: Set(contributors.values.flatMap { $0.organization?.observed.collections ?? [] }),
                    tags: Set(contributors.values.flatMap { $0.organization?.observed.tags ?? [] }))
                let roots = Set(contributors.map { $0.value.originKey ?? $0.key })
                // Independent origins do not acquire scalar authority by traversal order.
                let persistedScalarRoots = Set((byFilename[name] ?? [:]).filter { _, record in
                    record.destinationFilename == name && record.provisional != true
                        && (record.organization?.appliedTitle != nil || record.organization?.appliedFavorite != nil)
                }.map { $0.value.originKey ?? $0.key })
                let scalarAuthority = persistedScalarRoots.count == 1
                    ? persistedScalarRoots.contains(root)
                    : (plannedRoots[name]?.count ?? 0) <= 1 && (roots.isEmpty || roots == [root])
                if !scalarAuthority && (source.title != before.title || source.favorite != before.favorite) {
                    result.failures.append("\(name): independent sources disagree on its title or favourite; destination values were preserved.")
                }
                let policy = LibraryOrganizationReconciliation.plan(source: source, destination: before,
                    previous: journalRecord.organization, hidden: hidden, newCopy: copy.newCopy,
                    explicitCopy: explicitCopy, otherClaims: claims, scalarAuthority: scalarAuthority)
                if journalRecord.organizationIntent == nil, journalRecord.organization == policy.state,
                   before == policy.result {
                    if records[key]?.organizationIntent != nil {
                        result.states[key] = policy.state
                        result.expected[key] = policy.result
                    }
                    records[key] = journalRecord
                    byFilename[name, default: [:]][key] = journalRecord
                    continue
                }
                let intent = journalRecord.organizationIntent ?? SyncOrganizationIntent(operation: UUID().uuidString,
                    before: before, after: policy.result, state: policy.state)
                journalRecord.organizationIntent = intent
                // The immutable operation IDs survive ambiguous responses and restart.
                try await syncRecords.checkpoint(journalRecord, key: key, instance: prepared.recordScope)
                try await applyOrganizationIntent(intent, filename: name, destination: destination,
                    collections: prepared.collectionNames, isCurrent: target.isCurrent)
                result.states[key] = intent.state
                result.expected[key] = intent.after
                current[name] = intent.after
                var applied = journalRecord
                applied.organization = intent.state
                record = applied
                records[key] = record
                byFilename[name, default: [:]][key] = record
            } catch {
                result.failedFiles.insert(name)
                result.failures.append("\(name): organization could not be verified: \(error.sentence)")
            }
        }
        return result
    }

    private func applyOrganizationIntent(_ intent: SyncOrganizationIntent, filename: String,
                                         destination: any MoldBackend, collections: [String: String],
                                         isCurrent: @MainActor @Sendable () -> Bool) async throws {
        let old = intent.before, new = intent.after
        var mutations: [(String, GalleryBulkMutation)] = []
        for slug in new.collections.subtracting(old.collections).sorted() {
            guard let name = collections[slug] else { throw MoldClientError.malformedResponse }
            mutations.append(("add-collection-\(slug)", .init(filenames: [filename], addToCollection: .named(name))))
        }
        for slug in old.collections.subtracting(new.collections).sorted() {
            mutations.append(("remove-collection-\(slug)", .init(filenames: [filename], removeFromCollectionSlug: slug)))
        }
        mutations.append(("attributes", .init(filenames: [filename],
            titles: old.title == new.title ? [] : [.init(filename: filename, title: new.title)],
            favorite: old.favorite == new.favorite ? nil : new.favorite,
            addTags: new.tags.subtracting(old.tags).sorted(), removeTags: old.tags.subtracting(new.tags).sorted())))
        for (suffix, mutation) in mutations {
            try Task.checkCancellation()
            guard !syncStopped, isCurrent() else { throw CancellationError() }
            let bound = GalleryBulkMutation(filenames: mutation.filenames, titles: mutation.titles,
                favorite: mutation.favorite, addTags: mutation.addTags, removeTags: mutation.removeTags,
                addToCollection: mutation.addToCollection, removeFromCollectionSlug: mutation.removeFromCollectionSlug,
                operationId: organizationOperationID(intent.operation, suffix: suffix))
            if bound.titles.isEmpty && bound.favorite == nil && bound.addTags.isEmpty && bound.removeTags.isEmpty
                && bound.addToCollection == nil && bound.removeFromCollectionSlug == nil { continue }
            try await destination.mutate(bound)
        }
    }
}

private func organizationOperationID(_ operation: String, suffix: String) -> String {
    let hex = SHA256.hash(data: Data((operation + ":" + suffix).utf8)).prefix(16)
        .map { String(format: "%02x", $0) }.joined()
    let positions = [8, 12, 16, 20]
    var value = ""
    for (index, character) in hex.enumerated() {
        if positions.contains(index) { value.append("-") }
        value.append(character)
    }
    return value
}
