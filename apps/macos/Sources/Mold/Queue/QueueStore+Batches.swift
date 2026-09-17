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
    ///
    /// ONE at a time per machine. `refresh(on:)` has several concurrent
    /// callers -- the SSE coalescer, `QueuePane.load()`, `MachinesPane`'s
    /// `.task(id:)` and its Refresh -- and the body below reads
    /// `children[host]` before its first `await` and writes it back only at
    /// the end. Two overlapping runs therefore both computed `before` from
    /// the same snapshot, both saw the same `held → failed` transition, and
    /// both called `onOutcome`: two "Failed on plato" banners for one job,
    /// because a failure notification is deliberately never coalesced
    /// (`MoldNotifications.swift:123-127`). Pressing ⌘R while a
    /// `job_state_committed` frame was in flight was enough.
    ///
    /// Serialized rather than coalesced: the second caller wants a FRESH
    /// read, it just must not take its `before` from a world the first one
    /// has already moved on from.
    func hydrate(on host: MoldHost.ID) async {
        let previous = hydrations[host]
        // No `await` between the read above and the write below, so on the
        // main actor this claim is atomic.
        let mine = Task { [weak self] in
            await previous?.value
            await self?.hydrateNow(on: host)
        }
        hydrations[host] = mine
        await mine.value
        if hydrations[host] == mine { hydrations[host] = nil }
    }

    private func hydrateNow(on host: MoldHost.ID) async {
        guard !isSeeded, let client = hosts.backend(for: host) else { return }
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
                    let before = merged[status.id] ?? []
                    let after = merge(status.children, into: merged[status.id])
                    merged[status.id] = after
                    reportOutcomes(before: before, after: after, on: host)
                }
                for goneId in listing.missing.batchIds { merged.removeValue(forKey: goneId) }
                hosts.succeeded(on: host, doing: "list its batches")
            } catch {
                hosts.report(error, on: host, doing: "list its batches")
            }
        }
        children[host] = merged
    }

    /// `onOutcome`'s only caller. A child not seen before is not a
    /// TRANSITION this app watched happen, so it stays quiet -- only a state
    /// this store already recorded, now different, fires. A resolvable hold
    /// (missing model, or `retryable: true`) is not an outcome: it still
    /// offers a button.
    private func reportOutcomes(before: [BatchChild], after: [BatchChild], on host: MoldHost.ID) {
        guard let onOutcome else { return }
        let previously = Dictionary(uniqueKeysWithValues: before.map { ($0.jobId, $0) })
        for child in after {
            guard let was = previously[child.jobId], was.state != child.state else { continue }
            guard let entry = entries(on: host).first(where: { $0.id == child.jobId }) else { continue }
            if child.state == .failed {
                onOutcome(host, entry, child.error ?? entry.error ?? "The job failed.")
            } else if case let .prose(sentence, retryable: false)? = QueueHold.resolve(entry: entry, child: child) {
                onOutcome(host, entry, sentence)
            }
        }
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
