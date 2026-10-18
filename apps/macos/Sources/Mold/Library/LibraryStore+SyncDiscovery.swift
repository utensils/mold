import Foundation
import MoldClient

// Library Sync's first two stages: source discovery and destination
// preparation. Both read the machines' replicas -- already current, and
// refreshed together -- rather than full listings one machine at a time.
@MainActor
extension LibraryStore {
    /// The context a source print's issues and records are bound to.
    func originContext(_ hostID: MoldHost.ID) -> String {
        [hostID.uuidString, hosts.host(hostID)?.baseURL.absoluteString ?? "",
         hosts.instanceID(of: hostID) ?? ""].joined(separator: "\n")
    }

    func discoverMirrorSources(_ selected: [LibraryEntry], local: MoldHost,
                               syncAll: Bool) async -> MirrorDiscovery {
        var failures: [String] = []
        var hostErrors: [MoldHost.ID: String] = [:]
        var sourceHostIDs = syncAll
            ? Set(hosts.hosts.filter { $0.id != local.id }.map(\.id))
            : Set(selected.map(\.hostID))
        if !syncAll {
            for entry in selected {
                sourceHostIDs.formUnion(syncRecords.verifiedCopies(of: entry.id, authority: syncAuthority).map(\.host))
            }
            sourceHostIDs.remove(local.id)
        }
        let bucket = syncAll ? syncRecords.bucket(recordScope(local)) : nil
        var reads: [MoldHost.ID: MirrorSourceRead] = [:]
        // Every source at once: a slow machine never holds the others.
        await withTaskGroup(of: MirrorSourceRead.self) { group in
            for hostID in sourceHostIDs.sorted() {
                guard let sourceHost = hosts.host(hostID), hosts.backend(for: hostID) != nil else {
                    hostErrors[hostID] = "its machine is no longer available"
                    if syncAll {
                        failures.append("\(hosts.name(of: hostID) ?? hostID.uuidString): its machine is no longer available")
                    }
                    continue
                }
                let replica = replica(for: sourceHost)
                let client = hosts.backend(for: sourceHost)
                // Sync All reads both partitions: a tombstone needs to know
                // the source no longer holds a print anywhere.
                let request = pullRequest(sourceHost, .all)
                let context = originContext(hostID)
                let cursor = bucket?.cursors[hostID.uuidString].flatMap { $0.route == context ? $0.cursor : nil }
                let wantsCollections = syncAll || !selected.isEmpty
                group.addTask {
                    await MirrorSourceRead.read(sourceHost, replica: replica, request: request, client: client,
                                                cursor: cursor, candidates: syncAll, collections: wantsCollections)
                }
            }
            for await read in group { reads[read.host.id] = read }
        }
        settleTimelines()

        var discovered: [MoldHost.ID: MirrorSourceHost] = [:]
        for hostID in reads.keys.sorted() {
            guard let read = reads[hostID] else { continue }
            guard hosts.host(hostID) == read.host, let backend = hosts.backend(for: hostID) else {
                hostErrors[hostID] = "The machine route changed while reading its Library."
                if syncAll { failures.append("\(hosts.name(of: hostID) ?? hostID.uuidString): The machine route changed while reading its Library.") }
                continue
            }
            if let error = read.error {
                hostErrors[hostID] = error.sentence
                if syncAll { failures.append("\(hosts.name(of: hostID) ?? hostID.uuidString): \(error.sentence)") }
                continue
            }
            guard let view = read.view else { continue }
            if let error = read.collectionError { failures.append("Collections on \(hostID): \(error.sentence)") }
            discovered[hostID] = MirrorSourceHost(
                host: read.host, backend: backend, originContext: originContext(hostID), view: view,
                candidates: read.candidates, changed: read.changed,
                collections: read.collections.map { Dictionary($0.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first }) })
        }
        if syncStopped { return .stopped(failures) }
        return .ready(MirrorSources(selected: selected, hosts: discovered, hostErrors: hostErrors, failures: failures))
    }

    func prepareMirrorDestination(_ sources: MirrorSources, local: MoldHost, syncAll: Bool,
                                  visibilityGeneration: UUID) async -> MirrorPreparation {
        var failures: [String] = []
        let destination = hosts.backend(for: local)
        let instance = hosts.instanceID(of: local.id)
        // Sync All needs This Mac's Trash too: it must never recreate it.
        let replica = replica(for: local)
        do {
            try await replica.pull(pullRequest(local, syncAll ? .all : .live), using: destination)
        } catch {
            settleTimelines()
            return .unavailable("Couldn’t read \(local.name)’s Library: \(error.sentence)")
        }
        settleTimelines()
        guard hosts.host(local.id) == local, hosts.instanceID(of: local.id) == instance else {
            return .unavailable("Couldn’t read \(local.name)’s Library: The machine changed while reading its Library.")
        }
        let view = await replica.mirrorView()
        let bucket = syncRecords.bucket(recordScope(local))
        let route = local.baseURL.absoluteString + "\n" + (instance ?? "")
        let changed: Set<String>? = if syncAll, let cursor = bucket.destinationCursor, cursor.route == route {
            await replica.changedNames(since: cursor.cursor)
        } else { nil }

        let localCollections: [Collection]
        do {
            localCollections = try await destination.collections()
        } catch {
            failures.append("\(local.name)’s collections: \(error.sentence)")
            localCollections = collectionsPerHost[local.id] ?? []
        }
        var localCollectionBySlug = Dictionary(localCollections.map { ($0.slug, $0) }, uniquingKeysWith: { first, _ in first })
        var localNames = Dictionary(localCollections.map { ($0.slug, $0.name) }, uniquingKeysWith: { first, _ in first })
        var createdCollections = 0
        var failedCollectionSlugs: Set<String> = []
        // Visibility is a shared shelf attribute. Include existing and empty
        // shelves, not just memberships discovered while copying outputs.
        var collectionInventories = collectionsPerHost
        for (host, source) in sources.hosts { if let collections = source.collections { collectionInventories[host] = Array(collections.values) } }
        collectionInventories[local.id] = localCollections
        let desiredShelves = CollectionShelf.merge(collectionInventories)
        var selectedSlugs: Set<String> = Set(sources.selected.flatMap { entry in
            entry.print.collectionList.compactMap { sources.hosts[entry.hostID]?.collections?[$0]?.slug }
        })
        if !syncAll {
            for entry in sources.selected {
                let members = syncRecords.verifiedCopies(of: entry.id, authority: syncAuthority) + [entry.id]
                for id in members {
                    guard let source = sources.hosts[id.host], let row = source.view.rows[id.filename] else { continue }
                    selectedSlugs.formUnion(row.collectionList.compactMap { source.collections?[$0] }
                        .filter { $0.hidden == true }.map(\.slug))
                }
            }
        }
        var shelves: [(Collection, Bool)] = []
        for shelf in desiredShelves {
            if syncStopped { break }
            guard syncAll || selectedSlugs.contains(shelf.slug) else { continue }
            do {
                if let collection = localCollectionBySlug[shelf.slug] {
                    shelves.append((collection, shelf.hidden))
                } else {
                    let description = collectionInventories.values.flatMap { $0 }
                        .first { $0.slug == shelf.slug && $0.description != nil }?.description
                    let created = try await destination.createCollection(name: shelf.name, description: description)
                    guard created.slug == shelf.slug else { throw MoldClientError.malformedResponse }
                    localCollectionBySlug[shelf.slug] = created
                    localNames[shelf.slug] = created.name
                    createdCollections += 1
                    shelves.append((created, shelf.hidden))
                }
            } catch {
                failures.append("Collection “\(shelf.name)”: \(error.sentence)")
                failedCollectionSlugs.insert(shelf.slug)
            }
        }
        // One visibility pass for every shelf, one inventory assignment.
        let visibility = await syncCollectionVisibility(shelves, generation: visibilityGeneration,
                                                        destination: local, backend: destination)
        for (slug, message) in visibility.failures {
            failures.append("Collection “\(localNames[slug] ?? slug)”: \(message)")
            failedCollectionSlugs.insert(slug)
        }
        return .ready(MirrorDestination(
            host: local, backend: destination, instance: instance, recordScope: recordScope(local), view: view, changed: changed, bucket: bucket,
            receipts: MirrorReceipts(records: syncRecords, instance: recordScope(local), destinationInstance: instance, destinationHost: local.id, destinationRoute: local.baseURL.absoluteString),
            collectionNames: localNames, createdCollections: createdCollections,
            failedCollectionSlugs: failedCollectionSlugs, failures: failures))
    }
}

/// One source machine's replica, read off the main actor.
nonisolated struct MirrorSourceRead: Sendable {
    let host: MoldHost
    let view: ReplicaMirrorView?
    let candidates: [String]
    let changed: Set<String>?
    let collections: [Collection]?
    let collectionError: (any Error)?
    let error: (any Error)?

    static func read(_ host: MoldHost, replica: GalleryReplica, request: GalleryReplica.Pull,
                     client: any MoldBackend, cursor: ReplicaCursor?, candidates wantsCandidates: Bool,
                     collections wantsCollections: Bool) async -> MirrorSourceRead {
        async let collections = Self.collections(from: client, wanted: wantsCollections)
        do {
            try await replica.pull(request, using: client)
        } catch {
            _ = await collections
            return MirrorSourceRead(host: host, view: nil, candidates: [], changed: nil, collections: nil,
                                    collectionError: nil, error: error)
        }
        let view = await replica.mirrorView()
        let changed: Set<String>? = if let cursor { await replica.changedNames(since: cursor) } else { nil }
        let candidates = wantsCandidates
            ? view.rows.keys.filter { view.live($0) != nil && LibraryStore.syncableName($0) }.sorted()
            : []
        let (rows, collectionError) = await collections
        return MirrorSourceRead(host: host, view: view, candidates: candidates, changed: changed,
                                collections: rows, collectionError: collectionError, error: nil)
    }

    private static func collections(from client: any MoldBackend, wanted: Bool) async -> ([Collection]?, (any Error)?) {
        guard wanted else { return (nil, nil) }
        do { return (try await client.collections(), nil) } catch { return (nil, error) }
    }
}
