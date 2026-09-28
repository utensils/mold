import Foundation
import MoldClient

/// Every machine's queue (DESIGN.md §5.3), live from its event stream: a batch
/// as one parent row with its children, held rows that ask in words, and the
/// actions each machine says it can take. A dropped stream reconciles by
/// asking again, never by trusting the deltas it missed.
@Observable
final class QueueStore {
    private(set) var listings: [MoldHost.ID: [QueueEntry]] = [:]
    private(set) var children: [MoldHost.ID: [String: [BatchChild]]] = [:]
    /// The whole-queue gate, from the stream's edge events; `nil` hands the
    /// question back to `/api/status.queue_paused`.
    private(set) var gate: [MoldHost.ID: Bool] = [:]
    /// Each running row's latest step and denoise preview, by job id.
    private(set) var progress: [String: JobProgress] = [:]
    /// A one-line result a person should see ("Sent to hal9000."), in place.
    var summary: String?

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored private var pending: [MoldHost.ID: Task<Void, Never>] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
        hosts.listen { [weak self] id, event in
            switch event {
            case .queue(.paused): self?.gate[id] = true
            case .queue(.resumed): self?.gate[id] = false
            case .resyncRequired: self?.gate[id] = nil; self?.schedule(id)
            case .job, .deviceStateChanged: self?.schedule(id)
            default: break
            }
        }
    }

    func reload() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.upHosts { group.addTask { await self.poll(host.id) } }
        }
        for id in Set(listings.keys).subtracting(hosts.hosts.map(\.id)) {
            listings[id] = nil; children[id] = nil; gate[id] = nil
        }
    }

    func poll(_ id: MoldHost.ID) async {
        guard let host = hosts.host(id), hosts.isUp(host) else { listings[id] = nil; return }
        let client = hosts.backend(for: host)
        do {
            let entries = try await client.queue().merged.filter(\.state.isLive)
            listings[id] = entries
            let batches = Array(Set(entries.compactMap(\.batchId))).sorted()
            if !batches.isEmpty, let listing = try? await client.batchStatuses(batchIds: batches) {
                children[id] = Dictionary(uniqueKeysWithValues: listing.batches.map { ($0.id, $0.children) })
            } else {
                children[id] = [:]
            }
        } catch {
            hosts.report(host, doing: String(localized: "list its queue"), error)
        }
    }

    /// Coalesced: a burst of job events is one re-read.
    private func schedule(_ id: MoldHost.ID) {
        pending[id]?.cancel()
        pending[id] = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            await self?.poll(id)
        }
    }

    /// While the Queue is on screen: each running row's step and preview,
    /// about once a second. Stops with the view (`.task`).
    func followRunning() async {
        while !Task.isCancelled {
            var fresh: [String: JobProgress] = [:]
            for host in hosts.upHosts {
                for entry in listings[host.id] ?? [] where entry.state == .running {
                    if let value = try? await hosts.backend(for: host).jobPreview(jobId: entry.id) { fresh[entry.id] = value }
                }
            }
            progress = fresh
            try? await Task.sleep(for: .seconds(1))
        }
    }

    // MARK: - Reading

    func isQueuePaused(_ id: MoldHost.ID) -> Bool {
        if let known = gate[id] { return known }
        guard let host = hosts.host(id), case let .up(status) = hosts.reachability(of: host) else { return false }
        return status.queuePaused ?? false
    }

    /// Machines whose queue can be paused, in list order.
    var gateMachines: [MoldHost] {
        hosts.upHosts.filter { hosts.capabilities[$0.id]?.canPauseQueue == true }
    }

    func groups(for id: MoldHost.ID) -> [QueueGroup] {
        QueueGroup.build(listings[id] ?? [], children: children[id] ?? [:])
    }

    func child(for entry: QueueEntry, on id: MoldHost.ID) -> BatchChild? {
        entry.batchId.flatMap { children[id]?[$0] }?.first { $0.jobId == entry.id }
    }

    func hold(for entry: QueueEntry, on id: MoldHost.ID) -> QueueHold? {
        QueueHold.resolve(entry: entry, child: child(for: entry, on: id))
    }

    /// The Queue tab's badge: what is being made or held, fleet-wide.
    var badge: Int {
        listings.values.joined().filter { $0.state == .running || $0.state == .held }.count
    }

    var isEmpty: Bool { listings.values.allSatisfy(\.isEmpty) }

    /// Whether this row can be cancelled at all: a job already rendering on a
    /// machine that cannot stop at a safe point has nothing to press.
    func canCancel(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard entry.state == .running else { return entry.state.isLive }
        return hosts.capabilities[id]?.canCancelRunningJob == true
    }

    func canPause(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        hosts.capabilities[id]?.canPauseOneJob == true && (entry.state == .queued || entry.state == .paused)
    }

    func canReorder(on id: MoldHost.ID) -> Bool { hosts.capabilities[id]?.canReorderQueue == true }

    // MARK: - Acting

    func cancel(_ entry: QueueEntry, on id: MoldHost.ID) async {
        await act(id, String(localized: "cancel that job")) { client in
            if entry.state == .held { _ = try await client.cancelHeldJob(id: entry.id) } else {
                try await client.cancelJob(id: entry.id)
            }
        }
    }

    func setPaused(_ paused: Bool, _ entry: QueueEntry, on id: MoldHost.ID) async {
        await act(id, paused ? String(localized: "pause that job") : String(localized: "resume that job")) { client in
            if paused { try await client.pauseJob(id: entry.id) } else { try await client.resumeJob(id: entry.id) }
        }
    }

    /// Up or down one place, where the machine will actually put it.
    func move(_ entry: QueueEntry, up: Bool, on id: MoldHost.ID) async {
        let waiting = (listings[id] ?? []).filter(\.state.isReorderable)
        guard let index = waiting.firstIndex(where: { $0.id == entry.id }),
              up ? index > 0 : index + 1 < waiting.count else { return }
        let neighbour: String? = up ? (index >= 2 ? waiting[index - 2].id : nil) : waiting[index + 1].id
        await moveGroup([entry.id], after: neighbour, on: id)
    }

    /// Rows moved as a unit to sit after `neighbour` (`nil`: the front), as
    /// the PATCHes `QueueOrder` plans, in order.
    func moveGroup(_ ids: [String], after neighbour: String?, on id: MoldHost.ID) async {
        let plan = QueueOrder.moves(ids, after: neighbour, in: listings[id] ?? [])
        guard !plan.isEmpty else { return }
        await act(id, String(localized: "move that job")) { client in
            for step in plan { try await client.reorderJob(id: step.id, position: step.position) }
        }
    }

    func retry(_ entry: QueueEntry, on id: MoldHost.ID) async {
        guard let instance = hosts.instanceID(of: id), let authority = entry.authority(instanceId: instance) else { return }
        await act(id, String(localized: "try that job again")) { try await $0.retryJob(authority) }
    }

    func setQueuePaused(_ paused: Bool, on ids: [MoldHost.ID]) async {
        for id in ids where hosts.capabilities[id]?.canPauseQueue == true {
            await act(id, paused ? String(localized: "pause its queue") : String(localized: "resume its queue")) { client in
                let state = paused ? try await client.pauseQueue() : try await client.resumeQueue()
                self.gate[id] = state.paused
            }
        }
    }

    /// Everything waiting or held, on these machines. Anything rendering
    /// keeps going; held rows are cleared one by one (the bulk route leaves
    /// them), as on the Mac.
    func empty(_ ids: [MoldHost.ID]) async {
        for id in ids {
            await act(id, String(localized: "empty its queue")) { client in
                if self.hosts.capabilities[id]?.canCancelAllQueued == true { _ = try await client.cancelAllQueued() }
                for held in (self.listings[id] ?? []) where held.state == .held {
                    _ = try await client.cancelHeldJob(id: held.id)
                }
            }
        }
    }

    private func act(_ id: MoldHost.ID, _ verb: String, _ body: (any MoldBackend) async throws -> Void) async {
        guard let host = hosts.host(id) else { return }
        do { try await body(hosts.backend(for: host)) } catch { hosts.report(host, doing: verb, error) }
        await poll(id)
    }
}
