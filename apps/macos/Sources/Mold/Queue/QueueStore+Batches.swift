import Foundation
import MoldClient

/// The typed half of a queue row, which the queue endpoint does not carry.
///
/// `error_code` and `revision` exist only on the batch child, and `error_code`
/// only while it is held (`routes.rs:2951-2956`). One call per machine
/// hydrates every batch the listing mentions -- `POST /api/generation-batches/status`
/// is a READ (`routes.rs:3411`, `spawn_queue_read`) and takes up to
/// `QueueBatchStatusLimit.identities` ids, so a pane full of batches is one
/// request, not one per batch.
@MainActor
extension QueueStore {

    /// Reads every batch this machine's current listing mentions, merges each
    /// child against what was already known, and drops a batch the machine
    /// answers `missing` for rather than keeping a stale copy.
    func hydrate(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        let ids = Array(Set(entries(on: host).compactMap(\.batchId)))
        guard !ids.isEmpty else {
            children[host] = [:]
            return
        }

        // Seeded from what is already known, restricted to batches still on
        // screen -- a chunk whose call fails keeps its last view rather than
        // losing it.
        var merged = (children[host] ?? [:]).filter { ids.contains($0.key) }
        for chunk in ids.chunked(by: QueueBatchStatusLimit.identities) {
            do {
                let listing = try await client.batchStatuses(batchIds: chunk)
                for status in listing.batches {
                    merged[status.id] = merge(status.children, into: merged[status.id])
                }
                for goneId in listing.missing.batchIds { merged.removeValue(forKey: goneId) }
                hosts.succeeded(on: host, doing: "list its batches")
            } catch {
                hosts.report(error, on: host, doing: "list its batches")
            }
        }
        children[host] = merged
    }

    /// The newer view of each child, by `supersedes(_:)` -- never the newest
    /// CALL, since a delayed answer can still arrive after a fresher one
    /// already landed.
    private func merge(_ incoming: [BatchChild], into existing: [BatchChild]?) -> [BatchChild] {
        var byJobId: [String: BatchChild] = [:]
        for child in existing ?? [] { byJobId[child.jobId] = child }
        for child in incoming where byJobId[child.jobId].map({ child.supersedes($0) }) ?? true {
            byJobId[child.jobId] = child
        }
        return Array(byJobId.values)
    }

    /// The typed cause behind a held row, when this machine has answered for
    /// its batch. A `nil` child -- a host that answered no batch status, or a
    /// row with no batch at all -- still resolves from the row's own
    /// sentence, the state the pane has shipped in since M1.
    func hold(for entry: QueueEntry, on host: MoldHost.ID) -> QueueHold? {
        let child = entry.batchId
            .flatMap { children[host]?[$0] }
            .flatMap { rows in rows.first { $0.jobId == entry.id } }
        return QueueHold.resolve(entry: entry, child: child)
    }

    /// This machine's queue as the pane draws it: a flat row, or a batch and
    /// its children.
    func groups(on host: MoldHost.ID) -> [QueueGroup] {
        QueueGroup.build(entries(on: host), children: children[host] ?? [:])
    }

    /// One action reaching every LIVE row of a group, serialized, then a
    /// single re-read -- the same "N calls, one re-read" shape reorder's
    /// batch move takes (design decision 4), reused here for a group's own
    /// Pause/Resume/Cancel rather than one re-read per child. A settled row
    /// (complete, failed, cancelled) is left alone even when it rides along
    /// in `group.rows`.
    func act(_ action: QueueRow.Action, onLiveChildrenOf group: QueueGroup, host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        for entry in group.rows where entry.state.isLive {
            do {
                switch action {
                case .cancel: try await client.cancelJob(id: entry.id)
                case .pause: try await client.pauseJob(id: entry.id)
                case .resume: try await client.resumeJob(id: entry.id)
                // Retry needs a `QueueAuthority` per row, not a bare id, and
                // belongs to `QueueHoldRow` -- not a group-wide action.
                case .retry: continue
                }
                hosts.succeeded(on: host)
            } catch {
                hosts.report(error, on: host, doing: groupVerb(action))
            }
        }
        await poll(host)
    }

    private func groupVerb(_ action: QueueRow.Action) -> String {
        switch action {
        case .cancel: "cancel that job"
        case .pause: "pause that job"
        case .resume: "resume that job"
        case .retry: "retry that job"
        }
    }
}

private extension Array {
    /// Splits into pieces of at most `size`. `QueueBatchStatusLimit` is more
    /// batches than a queue pane will ever show at once, but this chunks
    /// anyway rather than trust that forever.
    func chunked(by size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { start in
            Array(self[start ..< Swift.min(start + size, count)])
        }
    }
}
