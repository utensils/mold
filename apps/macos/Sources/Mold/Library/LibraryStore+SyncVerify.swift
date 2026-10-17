import Foundation
import MoldClient

// Library Sync's final checkpoint verification, and the run's bookkeeping:
// records, the retry set, tombstones and each mirror cursor, written once.
@MainActor
extension LibraryStore {
    static func destinationRoute(_ local: MoldHost, instance: String?) -> String {
        local.baseURL.absoluteString + "\n" + (instance ?? "")
    }

    /// Only the prints this run touched are verified, through the bounded
    /// checkpoint: evidence first, then bindings, then evidence again.
    func verifyMirrorCopies(sources: MirrorSources, prepared: MirrorDestination,
                            plan: MirrorWorkPlan, copies: MirrorCopies,
                            organization: MirrorOrganization) async -> [String] {
        let destination = prepared.backend
        let local = prepared.host
        let instance = prepared.instance
        let checkedKeys = Set(plan.items.map(\.target.key))
        let touched = copies.successful.filter { checkedKeys.contains($0.target.key) }
        var failures: [String] = []
        var outcome = LibrarySyncRecords.RunOutcome()
        var sealedDestination: Set<String> = []
        let localReplica = replica(for: local)
        let refreshed = await localReplica.mirrorView()

        if !touched.isEmpty {
            publishSyncProgress("Verifying copied prints and retained inputs…", force: true)
            let destinationNames = Array(Set(touched.map(\.filename)))
            let verifiedDestinationCheckpoint = try? await destination.gallerySyncCheckpoint(filenames: destinationNames)
            var checkpointFailures: Set<String> = []
            if verifiedDestinationCheckpoint != nil {
                // Capture evidence BEFORE checking bindings. A foreign writer
                // changing a binding after copy must not seal a stale success.
                await withTaskGroup(of: (MirrorCopy, MirrorResult).self) { group in
                    var next = 0
                    func refill(stopped: Bool) {
                        while next < touched.count {
                            let copy = touched[next]
                            next += 1
                            if stopped {
                                checkpointFailures.insert(copy.filename)
                                continue
                            }
                            group.addTask { (copy, await repairCachedSources(copy.target, to: destination, as: copy.filename)) }
                            return
                        }
                    }
                    for _ in 0..<min(3, touched.count) { refill(stopped: syncStopped) }
                    for await (copy, result) in group {
                        refill(stopped: syncStopped)
                        if result.filename == nil {
                            checkpointFailures.insert(copy.filename)
                            failures.append("\(copy.target.print.filename): \(result.error ?? "copy verification failed")")
                        }
                    }
                }
            }
            let finalDestinationCheckpoint = try? await destination.gallerySyncCheckpoint(filenames: destinationNames)
            var namesByHost: [MoldHost.ID: [String]] = [:]
            for copy in touched where plan.sourceCheckpoints[copy.target.hostID] != nil {
                namesByHost[copy.target.hostID, default: []].append(copy.target.print.filename)
            }
            var stableSourceCheckpoints: [MoldHost.ID: GallerySyncCheckpoint] = [:]
            await withTaskGroup(of: (MoldHost.ID, GallerySyncCheckpoint?).self) { group in
                for (hostID, names) in namesByHost {
                    guard let backend = sources.hosts[hostID]?.backend else { continue }
                    group.addTask { (hostID, try? await backend.gallerySyncCheckpoint(filenames: names)) }
                }
                for await (hostID, after) in group {
                    if let after, after.instanceId == plan.sourceCheckpoints[hostID]?.instanceId {
                        stableSourceCheckpoints[hostID] = after
                    }
                }
            }
            for copy in touched {
                let target = copy.target, filename = copy.filename
                let sourceKey = target.key
                guard !organization.failedFiles.contains(filename),
                      !plan.unresolvedCollections.contains(sourceKey),
                      !target.collections.contains(where: {
                          prepared.failedCollectionSlugs.contains($0.slug)
                      }) else { continue }
                guard let localPrint = refreshed.live(filename),
                      var record = LocalSyncRecord(source: target.print, from: target.hostID, local: localPrint,
                                                   on: local.id, destinationInstance: instance)
                else { continue }
                if let before = plan.sourceCheckpoints[target.hostID],
                   let after = stableSourceCheckpoints[target.hostID],
                   let token = before.revisions[target.print.filename],
                   after.revisions[target.print.filename] == token {
                    record.sourceCheckpoint = before.instanceId + ":" + token
                }
                if !checkpointFailures.contains(filename),
                   target.isCurrent(),
                   let checkpoint = finalDestinationCheckpoint,
                   let verified = verifiedDestinationCheckpoint,
                   checkpoint.instanceId == verified.instanceId,
                   let token = checkpoint.revisions[filename],
                   verified.revisions[filename] == token {
                    record.destinationCheckpoint = checkpoint.instanceId + ":" + token
                }
                record.sourceRoute = target.originContext
                record.destinationRoute = local.baseURL.absoluteString
                outcome.records[sourceKey] = record
                outcome.clearedPending.insert(sourceKey)
                if record.sealed {
                    outcome.settled.insert(sourceKey)
                    sealedDestination.insert(filename)
                }
            }
        }

        outcome.settled.formUnion(plan.settled)
        outcome.retried = plan.considered.subtracting(outcome.settled)
        outcome.removed = plan.tombstones
        outcome.keptRemoved = plan.keptRemovedKeys
        outcome.reconsidered = plan.considered
        if !syncStopped {
            for (hostID, source) in sources.hosts {
                outcome.sourceCursors[hostID.uuidString] = .init(cursor: source.view.cursor, route: source.originContext)
            }
            let route = Self.destinationRoute(local, instance: instance)
            // This run's own writes moved This Mac's log past the cursor it
            // planned from. Advance over them, but carry every name in that
            // stretch it did not just seal -- a foreign change among them is
            // still owed a look.
            if hosts.instanceID(of: local.id) == instance,
               let between = await localReplica.changedNames(since: prepared.view.cursor) {
                let reverse = syncRecords.sourceKeys(copiedAs: instance)
                for name in between where !sealedDestination.contains(name) {
                    outcome.retried.formUnion(reverse[name] ?? [])
                    for (hostID, source) in sources.hosts where source.view.live(name) != nil {
                        outcome.retried.insert(LocalSyncRecord.key(hostID: hostID, filename: name))
                    }
                }
                outcome.retried.subtract(outcome.removed)
                outcome.destinationCursor = .init(cursor: refreshed.cursor, route: route)
            } else {
                outcome.destinationCursor = .init(cursor: prepared.view.cursor, route: route)
            }
        }
        await syncRecords.finishRun(outcome, instance: instance)
        return failures
    }
}
