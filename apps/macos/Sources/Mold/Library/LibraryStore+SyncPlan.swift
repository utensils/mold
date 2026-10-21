import Foundation
import MoldClient

// Library Sync's planning stage: which prints to look at, and what each needs.
//
// An explicit Sync All press plans every print, skipping only on matching
// source and destination checkpoint evidence (one checkpoint read per
// machine, cached recipe digests: still nothing per print when nothing
// changed). Keep in Sync's automatic runs consider only what changed since
// each source's persisted mirror cursor -- the replica's change log -- plus the retry set (anything a run did
// not seal), pending organization, and names This Mac's own log says changed.
// Everything else is a sealed copy that nothing has touched since, so a run
// in which nothing changed does no per-print work for ordinary prints.
// Publication assets are re-probed because older receipts did not bind them. A cursor the log
// cannot answer (first run, relaunch, route or instance change) plans every
// print, exactly as before.
@MainActor
extension LibraryStore {
    /// How stale the last full plan may be before a cursor-scoped run plans
    /// everything anyway: what a source-side retained-input change with no
    /// gallery change waits for at most.
    static let fullPlanInterval: TimeInterval = 24 * 60 * 60

    /// Pre-artifact receipts/checkpoints proved the movie and conditioning pins
    /// only. Re-probe these separately owned assets even for a cached movie.
    static func requiresPublicationAssetProbe(_ print: GalleryPrint) -> Bool {
        print.metadata.videoJob?.artifacts != nil
    }

    func planMirrorWork(_ sources: MirrorSources, destination: MirrorDestination,
                        syncAll: Bool, fullPlan requestedFull: Bool = true) async -> MirrorWorkPlan {
        let local = destination.host
        let records = destination.records
        let bucket = destination.bucket
        var failures: [String] = []
        var consideredPairs: [(MoldHost.ID, String)] = []
        var fullHosts: Set<MoldHost.ID> = []
        var unchanged = 0
        var unchangedKeptRemoved = 0
        let reverse = syncAll ? syncRecords.sourceKeys(copiedAs: destination.recordScope) : [:]
        let staleFullPlan = bucket.lastFullPlan.map { Date().timeIntervalSince($0) > Self.fullPlanInterval } ?? true
        let planEverything = requestedFull || staleFullPlan

        if !syncAll {
            for entry in sources.selected {
                if let error = sources.hostErrors[entry.hostID] {
                    failures.append("\(entry.print.filename): \(error)")
                    continue
                }
                consideredPairs.append((entry.hostID, entry.print.filename))
            }
        } else {
            var owed: [MoldHost.ID: Set<String>] = [:]
            let organizationPending = Set(bucket.records.filter { $0.value.organizationIntent != nil }.keys)
            for key in bucket.retry.union(bucket.pending.keys).union(organizationPending) {
                if let source = LocalSyncRecord.source(of: key) { owed[source.host, default: []].insert(source.filename) }
            }
            var kept: [MoldHost.ID: Set<String>] = [:]
            for key in bucket.keptRemoved {
                if let source = LocalSyncRecord.source(of: key) { kept[source.host, default: []].insert(source.filename) }
            }
            for hostID in sources.hosts.keys.sorted() {
                guard let source = sources.hosts[hostID] else { continue }
                guard !planEverything, let changed = source.changed, let localChanged = destination.changed else {
                    fullHosts.insert(hostID)
                    let privacy = source.view.rows.keys.filter { name in
                        source.view.isTrash(name) && syncRecords.verifiedCopy(of: PrintID(host: hostID, filename: name),
                            on: local.id, authority: syncAuthority) != nil
                    }
                    consideredPairs += Set(source.candidates + privacy).sorted().map { (hostID, $0) }
                    continue
                }
                func candidate(_ name: String) -> Bool {
                    Self.syncableName(name) && (source.view.live(name) != nil
                        || (source.view.isTrash(name) && syncRecords.verifiedCopy(of: PrintID(host: hostID, filename: name),
                            on: local.id, authority: syncAuthority) != nil))
                }
                var names = Set(source.candidates.filter { name in
                    source.view.live(name).map(Self.requiresPublicationAssetProbe) == true
                })
                for name in changed where candidate(name) { names.insert(name) }
                for name in owed[hostID] ?? [] where candidate(name) { names.insert(name) }
                for name in localChanged {
                    // A same-named original (a local removal or a Trash
                    // restore) and every copy This Mac's log says changed.
                    if candidate(name) { names.insert(name) }
                    for key in reverse[name] ?? [] {
                        if let origin = LocalSyncRecord.source(of: key), origin.host == hostID, candidate(origin.filename) {
                            names.insert(origin.filename)
                        }
                    }
                }
                consideredPairs += names.sorted().map { (hostID, $0) }
                let keptHere = (kept[hostID] ?? []).filter { !names.contains($0) && candidate($0) }.count
                unchangedKeptRemoved += keptHere
                unchanged += max(0, source.candidates.count - names.count - keptHere)
            }
        }

        if syncAll {
            // A changed backup can add privacy protection even while the
            // original's own row/checkpoints are unchanged.
            var present = Set(consideredPairs.map { LocalSyncRecord.key(hostID: $0.0, filename: $0.1) })
            for (hostID, name) in consideredPairs {
                if let key = syncRecords.verifiedOrigin(of: PrintID(host: hostID, filename: name), authority: syncAuthority),
                   let root = LocalSyncRecord.source(of: key), sources.hosts[root.host]?.view.live(root.filename) != nil,
                   present.insert(key).inserted { consideredPairs.append((root.host, root.filename)) }
            }
        }

        // Targets for the considered prints only.
        var targets: [MirrorTarget] = []
        var unresolvedCollections: Set<String> = []
        for (hostID, name) in consideredPairs {
            guard let source = sources.hosts[hostID], hosts.host(hostID) == source.host,
                  let print = source.view.rows[name] else {
                failures.append("\(name): its recipe is no longer available")
                continue
            }
            let collectionIDs = source.collections ?? [:]
            let names = print.collectionList.compactMap { id -> (slug: String, name: String)? in
                guard let collection = collectionIDs[id] else { return nil }
                return (collection.slug, collection.name)
            }
            if names.count != print.collectionList.count {
                failures.append("\(print.filename): a source collection is no longer available")
                unresolvedCollections.insert(LocalSyncRecord.key(hostID: hostID, filename: print.filename))
            }
            let sourceID = PrintID(host: hostID, filename: print.filename)
            if syncRecords.hasVerifiedIncoming(sourceID, authority: syncAuthority), syncRecords.verifiedOrigin(of: sourceID, authority: syncAuthority) == nil {
                failures.append("\(print.filename): independent original copies disagree on its lineage; copy from an original machine so its organization remains authoritative.")
                unresolvedCollections.insert(LocalSyncRecord.key(hostID: hostID, filename: print.filename))
                continue
            }
            if syncAll, let origin = syncRecords.verifiedOrigin(of: sourceID, authority: syncAuthority),
               let root = LocalSyncRecord.source(of: origin),
               sources.hosts[root.host]?.view.live(root.filename) != nil {
                // The original is already in this run. Forwarded organization
                // must never become a new source contribution or scalar edit.
                unchanged += 1
                continue
            }
            let sourceHost = source.host
            let sourceInstance = hosts.instanceID(of: sourceHost.id)
            let destinationInstance = hosts.instanceID(of: local.id)
            targets.append(MirrorTarget(hostID: hostID, print: print, source: source.backend,
                                        originContext: source.originContext,
                                        isCurrent: { [weak self] in
                                            self?.hosts.host(sourceHost.id) == sourceHost && self?.hosts.host(local.id) == local
                                                && self?.hosts.instanceID(of: sourceHost.id) == sourceInstance
                                                && self?.hosts.instanceID(of: local.id) == destinationInstance
                                        }, collections: names, privacyOnly: source.view.isTrash(name) || (syncAll
                                            && syncRecords.verifiedOrigin(of: sourceID, authority: syncAuthority) != nil
                                            && syncRecords.verifiedCopy(of: sourceID, on: local.id, authority: syncAuthority) != nil)))
        }

        // Checkpoints for exactly what is considered: the whole GET only where
        // a plan is full anyway.
        var sourceCheckpoints: [MoldHost.ID: GallerySyncCheckpoint] = [:]
        var destinationCheckpoint: GallerySyncCheckpoint?
        if !targets.isEmpty {
            publishSyncProgress("Preparing \(targets.count.formatted()) prints…", force: true)
            var namesByHost: [MoldHost.ID: [String]] = [:]
            for target in targets { namesByHost[target.hostID, default: []].append(target.print.filename) }
            let destinationNames = Array(Set(targets.compactMap { records[$0.key]?.destinationFilename }))
            let wholeDestination = !fullHosts.isEmpty || destinationNames.count > 2 * HTTPBackend.boundedCheckpointLimit
            let backend = destination.backend
            async let localCheckpoint: GallerySyncCheckpoint? = wholeDestination
                ? try? await backend.gallerySyncCheckpoint()
                : destinationNames.isEmpty ? nil : try? await backend.gallerySyncCheckpoint(filenames: destinationNames)
            await withTaskGroup(of: (MoldHost.ID, GallerySyncCheckpoint?).self) { group in
                for (hostID, names) in namesByHost {
                    guard let client = sources.hosts[hostID]?.backend else { continue }
                    let whole = fullHosts.contains(hostID) || names.count > 2 * HTTPBackend.boundedCheckpointLimit
                    group.addTask {
                        (hostID, whole ? try? await client.gallerySyncCheckpoint()
                            : try? await client.gallerySyncCheckpoint(filenames: names))
                    }
                }
                for await (hostID, checkpoint) in group { if let checkpoint { sourceCheckpoints[hostID] = checkpoint } }
            }
            destinationCheckpoint = await localCheckpoint
        }

        var work: [MirrorWorkItem] = []
        var cachedCopies: [MirrorCopy] = []
        var claimedOriginals: Set<String> = []
        var keptRemoved = unchangedKeptRemoved
        var keptRemovedKeys: Set<String> = []
        var settled: Set<String> = []
        let localInstance = destination.instance
        for (index, var target) in targets.enumerated() {
            if syncStopped { break }
            let original = target.print.filename
            let sourceKey = target.key
            let sourceID = PrintID(host: target.hostID, filename: original)
            if target.privacyOnly {
                if let verified = syncRecords.verifiedCopy(of: sourceID, on: local.id, authority: syncAuthority),
                   destination.view.live(verified.filename) != nil {
                    cachedCopies.append(MirrorCopy(target: target, filename: verified.filename))
                } else { settled.insert(sourceKey) }
                continue
            }
            if syncAll, let verified = syncRecords.verifiedCopy(of: sourceID, on: local.id, authority: syncAuthority) {
                if destination.view.live(verified.filename) == nil {
                    keptRemoved += 1; keptRemovedKeys.insert(sourceKey); settled.insert(sourceKey)
                    continue
                }
                if syncRecords.verifiedOrigin(of: sourceID, authority: syncAuthority) != nil {
                    // An existing component must never echo output or organization.
                    if Self.requiresPublicationAssetProbe(target.print) {
                        target.assetsOnly = true
                        targets[index] = target
                        work.append(MirrorWorkItem(index:index,target:target,filename:verified.filename,
                                                   occupied:true,cachedFilename:verified.filename))
                    } else { unchanged += 1; settled.insert(sourceKey) }
                    continue
                }
                if records[sourceKey] == nil {
                    work.append(MirrorWorkItem(index: index, target: target, filename: verified.filename,
                        occupied: true, cachedFilename: verified.filename))
                    continue
                }
            }
            let removedLocally = destination.view.rows[original] != nil && destination.view.isTrash(original)
            if syncAll, removedLocally
                || (records[sourceKey].map { $0.destinationInstance == localInstance && destination.view.live($0.destinationFilename) == nil } ?? false) {
                keptRemoved += 1
                keptRemovedKeys.insert(sourceKey)
                settled.insert(sourceKey)
                continue
            }
            let canClaimOriginal = !syncAll || !claimedOriginals.contains(original)
            let requestedName: String
            if let pendingName = bucket.pending[sourceKey] {
                requestedName = pendingName
                claimedOriginals.insert(original)
            } else if canClaimOriginal {
                requestedName = original
                claimedOriginals.insert(original)
            } else {
                requestedName = boundedMirrorName(
                    original, marker: target.hostID.uuidString.lowercased())
            }
            // A receipt proves only the name an import answered with: its
            // copy links through the occupied-name byte comparison.
            if syncAll, let cached = records[sourceKey], cached.provisional == true,
               destination.view.live(cached.destinationFilename) != nil {
                work.append(MirrorWorkItem(index: index, target: target, filename: cached.destinationFilename,
                                           occupied: true, cachedFilename: nil))
                continue
            }
            if syncAll,
               let cached = records[sourceKey],
               let existing = destination.view.live(cached.destinationFilename),
               cached.matches(source: target.print, from: target.hostID, local: existing, on: local.id) {
                let sourceToken = sourceCheckpoints[target.hostID].flatMap { checkpoint in
                    checkpoint.revisions[original].map { checkpoint.instanceId + ":" + $0 }
                }
                let destinationToken = destinationCheckpoint.flatMap { checkpoint in
                    checkpoint.revisions[existing.filename].map { checkpoint.instanceId + ":" + $0 }
                }
                if !Self.requiresPublicationAssetProbe(target.print),
                   bucket.pending[sourceKey] == nil, cached.organizationIntent == nil, cached.provisional != true,
                   target.isCurrent(),
                   let sourceToken, let destinationToken,
                   cached.sourceCheckpoint == sourceToken, cached.destinationCheckpoint == destinationToken,
                   cached.sourceRoute == target.originContext,
                   cached.destinationRoute == local.baseURL.absoluteString,
                   cached.destinationInstance == hosts.instanceID(of: local.id) {
                    cachedCopies.append(MirrorCopy(target: target, filename: existing.filename))
                    settled.insert(sourceKey)
                    continue
                }
                work.append(MirrorWorkItem(index: index, target: target, filename: requestedName, occupied: true, cachedFilename: existing.filename))
                continue
            }
            work.append(MirrorWorkItem(index: index, target: target, filename: requestedName,
                                       occupied: destination.view.live(requestedName) != nil, cachedFilename: nil))
        }

        return MirrorWorkPlan(
            targets: targets, items: work, cachedCopies: cachedCopies, keptRemoved: keptRemoved,
            unchanged: unchanged,
            plannedEverything: syncAll && fullHosts.count == sources.hosts.count && sources.hostErrors.isEmpty,
            considered: Set(targets.map(\.key)).union(unresolvedCollections), settled: settled,
            keptRemovedKeys: keptRemovedKeys, unresolvedCollections: unresolvedCollections,
            sourceCheckpoints: sourceCheckpoints,
            tombstones: syncAll ? tombstones(sources, destination: destination, fullHosts: fullHosts) : [],
            failures: failures)
    }

    /// Records whose source print and This Mac copy are both gone for good --
    /// neither live nor in either Trash. Nothing can bring them back, so the
    /// record (and any retry or pending entry) goes too. A copy removed only
    /// here keeps its record: that is what keeps it removed.
    private func tombstones(_ sources: MirrorSources, destination: MirrorDestination,
                            fullHosts: Set<MoldHost.ID>) -> Set<String> {
        var gone: Set<String> = []
        func check(_ key: String, _ host: MirrorSourceHost, _ name: String) {
            guard host.view.rows[name] == nil, let record = destination.records[key],
                  destination.view.rows[record.destinationFilename] == nil else { return }
            gone.insert(key)
        }
        for (hostID, host) in sources.hosts {
            if fullHosts.contains(hostID) {
                for key in destination.records.keys {
                    guard let origin = LocalSyncRecord.source(of: key), origin.host == hostID else { continue }
                    check(key, host, origin.filename)
                }
            } else {
                for name in host.changed ?? [] { check(LocalSyncRecord.key(hostID: hostID, filename: name), host, name) }
            }
        }
        return gone
    }
}
