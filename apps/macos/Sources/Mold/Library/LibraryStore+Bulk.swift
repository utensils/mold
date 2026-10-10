import Foundation
import MoldClient

/// Bounded lifecycle requests: an HTTP failure can leave part of a batch
/// committed, so only successful answers advance the confirmed count.
@MainActor
extension LibraryStore {
    enum BulkRemoval: String {
        case trash = "Moving to Trash", restore = "Putting Back", delete = "Deleting Permanently"
    }

    static let bulkBatchSize = 16

    func beginBulkActivity(_ message: String) -> UUID {
        let id = UUID()
        bulkActivities[id] = message
        return id
    }

    func updateBulkActivity(_ id: UUID, _ message: String) { bulkActivities[id] = message }
    func endBulkActivity(_ id: UUID) { bulkActivities[id] = nil }

    func runBulk(_ action: BulkRemoval, entries: [LibraryEntry]) async {
        guard !isBulkBusy, !entries.isEmpty else { return }
        bulkRunning = true
        bulkStopRequested = false
        bulkResult = nil
        defer {
            bulkRunning = false
            bulkProgress = nil
            bulkTargets.removeAll()
        }
        let copies = withCopies(entries)
        let groups = Dictionary(grouping: copies, by: \.hostID)
        let targets = hosts.hosts.compactMap { host -> (MoldHost, any MoldBackend, [LibraryEntry])? in
            guard let rows = groups[host.id] else { return nil }
            return (host, hosts.backend(for: host), rows)
        }
        let total = copies.count
        var reconciliationFailed = false
        var failureDetails: [String] = []
        var completed = 0
        var failed = total - targets.reduce(0) { $0 + $1.2.count }
        bulkTargets = Set(copies.map(\.id))
        bulkProgress = "\(action.rawValue) — 0 of \(total.formatted()) copies"
        for (host, client, rows) in targets {
            guard !bulkStopRequested, !Task.isCancelled else { break }
            var attempted = false
            for start in stride(from: 0, to: rows.count, by: Self.bulkBatchSize) {
                guard !bulkStopRequested, !Task.isCancelled else { break }
                guard hosts.host(host.id) == host else {
                    failed += rows.count - start
                    break
                }
                let batch = Array(rows[start..<min(start + Self.bulkBatchSize, rows.count)])
                let names = batch.map(\.print.filename)
                bulkProgress = "\(action.rawValue) — \(completed.formatted()) of \(total.formatted()) copies · \(host.name)"
                attempted = true
                do {
                    switch action {
                    case .trash: try await client.trash(names)
                    case .restore: try await client.restoreFromTrash(names)
                    case .delete:
                        let trashed = batch.filter { $0.print.trashedAt != nil }.map(\.print.filename)
                        let live = batch.filter { $0.print.trashedAt == nil }.map(\.print.filename)
                        // Trash deletion must not destroy a concurrently restored
                        // live print on the same host.
                        if !trashed.isEmpty { try await client.deleteTrashed(trashed) }
                        if !live.isEmpty { try await client.deleteForever(live) }
                    }
                    guard hosts.host(host.id) == host else {
                        failed += rows.count - start
                        break
                    }
                    completed += batch.count
                    let names = Set(names)
                    if action != .restore {
                        perHost[host.id] = (perHost[host.id] ?? []).filter { !names.contains($0.print.filename) }
                        rebuild()
                    }
                    if action != .trash {
                        trashPerHost[host.id] = (trashPerHost[host.id] ?? []).filter { !names.contains($0.print.filename) }
                        rebuildTrash()
                    }
                    hosts.succeeded(on: host.id)
                } catch {
                    failed += rows.count - start
                    guard hosts.host(host.id) == host else { break }
                    // No stale-snapshot rollback: earlier names in a failed
                    // request may already have moved on the host.
                    failureDetails.append("\(host.name): \(UserFacingError.describe(error))")
                    hosts.report(error, on: host.id, doing: action.rawValue.lowercased())
                    break
                }
            }
            guard attempted, hosts.host(host.id) == host else { continue }
            etags[host.id] = nil
            trashEtags[host.id] = nil
            bulkProgress = "Checking \(host.name) — \(completed.formatted()) of \(total.formatted()) copies confirmed"
            if !(await reconcileBulk(host: host, client: client)) { reconciliationFailed = true }
        }
        let ending = reconciliationFailed ? "Some listings could not be refreshed; check the machine and refresh again."
            : failed > 0 ? "Some requests could not be confirmed; the Library was refreshed."
            : (completed < total ? "Stopped after the current batch." : "Finished.")
        // Routine trash succeeds silently; interrupted or uncertain work remains visible.
        if action == .trash, !reconciliationFailed, failed == 0, completed == total {
            bulkResult = nil
            return
        }
        bulkResult = "\(action.rawValue): \(completed.formatted()) of \(total.formatted()) copies confirmed. \(ending)"
            + (failureDetails.isEmpty ? "" : " " + failureDetails.joined(separator: " · "))
    }

    /// Empty Trash must use the trash-only server operation. Enumerating
    /// names then calling deleteForever could destroy a concurrently restored
    /// live print; that endpoint deliberately supports live deletion too.
    func runEmptyTrash(on hostIDs: Set<MoldHost.ID>? = nil) async {
        guard !isBulkBusy else { return }
        bulkRunning = true
        bulkEmptying = true
        bulkStopRequested = false
        bulkResult = nil
        defer {
            bulkRunning = false
            bulkEmptying = false
            bulkProgress = nil
        }
        let destinations = hosts.hosts.filter { hostIDs?.contains($0.id) ?? true }
            .map { ($0, hosts.backend(for: $0)) }
        var completed = 0
        var failed = false
        var failureDetails: [String] = []
        for (host, client) in destinations {
            guard !bulkStopRequested, !Task.isCancelled else { break }
            guard hosts.host(host.id) == host else { failed = true; continue }
            bulkProgress = "Emptying Trash — \(completed) of \(destinations.count) machines · \(host.name)"
            do {
                try await client.emptyTrash()
                guard hosts.host(host.id) == host else { failed = true; continue }
                completed += 1
                hosts.succeeded(on: host.id)
            } catch {
                failed = true
                if hosts.host(host.id) == host {
                    failureDetails.append("\(host.name): \(UserFacingError.describe(error))")
                    hosts.report(error, on: host.id, doing: "empty the trash")
                }
            }
            guard hosts.host(host.id) == host else { continue }
            etags[host.id] = nil
            trashEtags[host.id] = nil
            bulkProgress = "Checking \(host.name) after emptying its Trash…"
            if !(await reconcileBulk(host: host, client: client)) { failed = true }
        }
        let ending = failed ? "Some results could not be confirmed; check the machine."
            : completed < destinations.count ? "Stopped after the current machine." : "Finished."
        bulkResult = "Emptying Trash: \(completed) of \(destinations.count) machines confirmed. \(ending)"
            + (failureDetails.isEmpty ? "" : " " + failureDetails.joined(separator: " · "))
    }

    private func reconcileBulk(host: MoldHost, client: any MoldBackend) async -> Bool {
        var succeeded = true
        do {
            let live = try await client.gallery(etag: nil)
            guard hosts.host(host.id) == host else { return false }
            if case let .fresh(prints, etag) = live {
                perHost[host.id] = prints.map { LibraryEntry(host: host, print: $0) }
                etags[host.id] = etag
                mutations.replayPending(on: host.id, in: self)
                rebuild()
            }
        } catch {
            succeeded = false
            if hosts.host(host.id) == host { hosts.report(error, on: host.id, doing: "list its prints") }
        }
        do {
            let trash = try await client.trashedPrints(etag: nil)
            guard hosts.host(host.id) == host else { return false }
            if case let .fresh(prints, etag) = trash {
                trashPerHost[host.id] = prints.map { LibraryEntry(host: host, print: $0) }
                trashEtags[host.id] = etag
                rebuildTrash()
            }
        } catch {
            succeeded = false
            if hosts.host(host.id) == host { hosts.report(error, on: host.id, doing: "list its trash") }
        }
        return succeeded
    }
}
