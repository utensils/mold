import BackgroundTasks
import Foundation
import MoldClient

/// Background refresh (DESIGN.md §A): while machines are paired, iOS can wake
/// the app after a requested 15-minute minimum to ask each
/// machine how it went -- notifications for what settled, the Live Activity
/// for what did not, and a fresh widget snapshot.
extension CompanionStores {
    static let refreshTask = "io.utensils.mold.companion.refresh"

    func scheduleRefresh() {
        guard !hosts.hosts.isEmpty else { return }
        let request = BGAppRefreshTaskRequest(identifier: Self.refreshTask)
        request.earliestBeginDate = .now.addingTimeInterval(15 * 60)
        try? BGTaskScheduler.shared.submit(request)
    }

    func backgroundRefresh() async {
        await library.restoreSaved()
        await hosts.refreshAll()
        for batch in generate.ledger.batches {
            await reconcile(batch)
        }
        // The widgets show what the machines hold NOW, not what the app saw
        // before it was suspended.
        async let library: Void = library.reload()
        async let queue: Void = queue.reload()
        _ = await (library, queue)
        notifier.iconBadge.update(self.library.unreadCount, allowPrompt: false)
        await notifier.iconBadge.flush()
        await widgets.refresh()
        scheduleRefresh()
    }

    /// A render older than this is not waited for any more.
    static let ledgerLimit: TimeInterval = 24 * 60 * 60

    /// One pending batch, as its machine reports it now.
    func reconcile(_ batch: ActiveBatch) async {
        // The machine was removed, or the render is a day old: stop asking.
        guard let host = hosts.host(batch.host), Date.now.timeIntervalSince(batch.startedAt) < Self.ledgerLimit else {
            activities.end(batch.clientBatchId, with: nil)
            return generate.ledger.remove(batch.clientBatchId)
        }
        guard hosts.isUp(host) else { return }
        let client = hosts.backend(for: host)
        let status: BatchStatus
        do {
            status = try await client.batchStatus(id: batch.id)
        } catch {
            // A machine that restarted has no record of it: nothing to wait for.
            if TransferPlan.isNotFound(error) {
                activities.end(batch.clientBatchId, with: nil)
                generate.ledger.remove(batch.clientBatchId)
            }
            return
        }
        let held = status.children.first { $0.state == .held }
        if let held, status.isAtRest {
            notifier.post(.held(BatchOutcome.heldSentence(held.error)), batch: batch, machine: host.name, print: nil)
            activities.end(batch.clientBatchId, with: nil)
            generate.ledger.remove(batch.clientBatchId)
        } else if let outcome = BatchOutcome(settling: status) {
            if outcome.results.isEmpty {
                let reason = outcome.failures.first ?? String(localized: "The render didn't finish.")
                notifier.post(.failed(reason), batch: batch, machine: host.name, print: nil)
            } else {
                let first = outcome.results.first?.filename.map { PrintID(host: batch.host, filename: $0) }
                notifier.post(.finished(count: outcome.results.count), batch: batch, machine: host.name, print: first)
            }
            activities.end(batch.clientBatchId, with: ActivityProjection.state(
                for: outcome.results.isEmpty ? .failed(outcome.failures.first ?? "") : .finished(outcome, host: batch.host),
                machine: host.name, waiting: 0, remaining: nil, preview: "\(batch.clientBatchId).jpg"))
            generate.ledger.remove(batch.clientBatchId)
        } else {
            let job = status.children.first { $0.state.isLive }?.jobId
            let progress = if let job { try? await client.jobPreview(jobId: job) } else { JobProgress?.none }
            activities.reflect(batch, status: status, progress: progress ?? nil)
        }
    }
}
