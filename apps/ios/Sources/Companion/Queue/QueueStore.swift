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
    private(set) var acting: Set<String> = []
    private var sourcePrompts: [String: String] = [:]
    private var inputFailures: Set<String> = []
    private var sourceThumbnails: [String: [QueueInputPreview]] = [:]
    @ObservationIgnored private var thumbnailAttempts: Set<String> = []
    @ObservationIgnored private var detailThumbnailAttempts: Set<String> = []
    @ObservationIgnored private var thumbnailInFlight: Set<String> = []

    private func thumbnailKey(_ entry: QueueEntry, _ id: MoldHost.ID) -> String {
        "\(id)|\(hosts.instanceID(of: id) ?? "unknown")|\(entry.id)"
    }

    func prompt(for entry: QueueEntry, on id: MoldHost.ID) -> String? {
        entry.metadata?.prompt ?? sourcePrompts[thumbnailKey(entry, id)]
    }

    func sourceThumbnail(for entry: QueueEntry, on id: MoldHost.ID) -> Data? {
        inputPreviews(for: entry, on: id).first(where: { $0.bytes != nil })?.bytes
    }

    func inputPreviews(for entry: QueueEntry, on id: MoldHost.ID) -> [QueueInputPreview] {
        sourceThumbnails[thumbnailKey(entry, id)] ?? []
    }

    func inputLoadFailed(for entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        inputFailures.contains(thumbnailKey(entry, id))
    }

    func loadSourceThumbnail(for entry: QueueEntry, on id: MoldHost.ID, detailed: Bool = false, retry: Bool = false) async {
        let key = thumbnailKey(entry, id)
        let flightKey = key + (detailed ? "|detail" : "|row")
        guard let host = hosts.host(id), hosts.isUp(host), !thumbnailInFlight.contains(flightKey) else { return }
        if detailed {
            guard retry || detailThumbnailAttempts.insert(key).inserted else { return }
        } else { guard thumbnailAttempts.insert(key).inserted else { return } }
        thumbnailInFlight.insert(flightKey)
        defer { thumbnailInFlight.remove(flightKey) }
        let client = hosts.backend(for: host)
        if !detailed, let detail = try? await client.queueJob(id: entry.id), canStorePreview(entry, on: id, key: key) {
            sourcePrompts[key] = detail.job.metadata?.prompt
        }
        do {
            let previews = try await client.queueInputPreviews(id: entry.id, firstOnly: !detailed, cached: inputPreviews(for: entry, on: id))
            guard !Task.isCancelled else { thumbnailAttempts.remove(key); detailThumbnailAttempts.remove(key); return }
            guard canStorePreview(entry, on: id, key: key), previews.allSatisfy({ ($0.bytes?.count ?? 0) <= 2 * 1024 * 1024 }) else { thumbnailAttempts.remove(key); detailThumbnailAttempts.remove(key); return }
            inputFailures.remove(key)
            let existing = sourceThumbnails[key] ?? []
            sourceThumbnails[key] = previews.map { preview in
                QueueInputPreview(input: preview.input, bytes: preview.bytes ?? existing.first(where: { $0.input == preview.input })?.bytes)
            }
            if detailed && previews.contains(where: { $0.input.preview && $0.bytes == nil }) { detailThumbnailAttempts.remove(key) }
            if !detailed && previews.contains(where: { $0.input.preview }) && !previews.contains(where: { $0.bytes != nil }) { thumbnailAttempts.remove(key) }
        } catch is CancellationError {
            thumbnailAttempts.remove(key); detailThumbnailAttempts.remove(key)
        } catch {
            if detailed && canStorePreview(entry, on: id, key: key) { inputFailures.insert(key) }
            detailThumbnailAttempts.remove(key)
            // Older hosts and jobs without a source image simply have no preview.
            if let issue = error as? MoldClientError, case .http(status: 404, code: _, message: _) = issue {
                return
            }
            thumbnailAttempts.remove(key)
        }
    }

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
        pruneSourcePreviews()
    }

    func poll(_ id: MoldHost.ID) async {
        guard let host = hosts.host(id), hosts.isUp(host) else { listings[id] = nil; return }
        let client = hosts.backend(for: host)
        do {
            let entries = try await client.queue().merged.filter(\.state.isLive)
            listings[id] = entries
            pruneSourcePreviews()
            let batches = Array(Set(entries.compactMap(\.batchId))).sorted()
            if !batches.isEmpty, let listing = try? await client.batchStatuses(batchIds: batches) {
                children[id] = Dictionary(uniqueKeysWithValues: listing.batches.map { ($0.id, $0.children) })
            } else {
                children[id] = [:]
            }
        } catch is CancellationError {
            return
        } catch {
            guard !Task.isCancelled else { return }
            listings[id] = nil
            hosts.report(host, doing: String(localized: "list its queue"), error)
        }
    }

    private func canStorePreview(_ entry: QueueEntry, on id: MoldHost.ID, key: String) -> Bool {
        !Task.isCancelled && hosts.host(id) != nil && key == thumbnailKey(entry, id)
            && listings[id]?.contains(where: { $0.id == entry.id }) == true
    }

    private func pruneSourcePreviews() {
        let retained = Set(hosts.hosts.flatMap { host in
            (listings[host.id] ?? []).map { thumbnailKey($0, host.id) }
        })
        sourceThumbnails = sourceThumbnails.filter { retained.contains($0.key) }
        sourcePrompts = sourcePrompts.filter { retained.contains($0.key) }
        inputFailures.formIntersection(retained)
        thumbnailAttempts.formIntersection(retained)
        detailThumbnailAttempts.formIntersection(retained)
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

    /// An unanswered queue cannot establish that nothing is waiting.
    var unavailableMachines: [MoldHost] {
        hosts.hosts.filter { !hosts.isUp($0) || listings[$0.id] == nil }
    }

    var isEmpty: Bool { listings.values.allSatisfy(\.isEmpty) }

    var heldCount: Int { listings.values.joined().filter { $0.state == .held }.count }

    /// Whether this row can be cancelled at all: a job already rendering on a
    /// machine that cannot stop at a safe point has nothing to press.
    func current(_ entry: QueueEntry, on id: MoldHost.ID) -> QueueEntry? {
        listings[id]?.first { $0.id == entry.id }
    }

    func headline(for entry: QueueEntry, on id: MoldHost.ID) -> String {
        hosts.models[id]?.first { $0.name == entry.model }?.headline ?? entry.modelHeadline
    }

    func isActing(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        acting.contains("\(id)|\(entry.id)")
    }

    private func actionable(_ entry: QueueEntry, on id: MoldHost.ID) -> QueueEntry? {
        guard let host = hosts.host(id), hosts.isUp(host), !isActing(entry, on: id) else { return nil }
        return current(entry, on: id)
    }

    func canCancel(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard let row = actionable(entry, on: id) else { return false }
        switch row.state {
        case .queued, .paused, .held: return true
        case .running: return hosts.capabilities[id]?.canCancelRunningJob == true
        default: return false
        }
    }

    func canPause(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard let row = actionable(entry, on: id) else { return false }
        return hosts.capabilities[id]?.canPauseOneJob == true && (row.state == .queued || row.state == .paused)
    }

    func canRetry(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard let row = actionable(entry, on: id), row.state == .held,
              row.retryable != false, let instance = hosts.instanceID(of: id),
              row.authority(instanceId: instance) != nil else { return false }
        if case .prose(_, retryable: false) = hold(for: row, on: id) { return false }
        return true
    }

    func canTransfer(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard let row = actionable(entry, on: id), row.state == .held,
              let instance = hosts.instanceID(of: id) else { return false }
        return row.authority(instanceId: instance) != nil
    }

    func canReorder(on id: MoldHost.ID) -> Bool {
        guard let host = hosts.host(id), hosts.isUp(host) else { return false }
        return hosts.capabilities[id]?.canReorderQueue == true
    }

    func canMove(_ entry: QueueEntry, on id: MoldHost.ID) -> Bool {
        guard canReorder(on: id), entry.state.isReorderable,
              let row = actionable(entry, on: id) else { return false }
        return row.state.isReorderable
    }

    // MARK: - Acting

    func cancel(_ entry: QueueEntry, on id: MoldHost.ID) async {
        guard canCancel(entry, on: id), let row = current(entry, on: id), row.state == entry.state else { return }
        await actOn(row, on: id, String(localized: "cancel that job")) { client in
            if row.state == .held { _ = try await client.cancelHeldJob(id: entry.id) } else {
                try await client.cancelJob(id: entry.id)
            }
        }
    }

    func setPaused(_ paused: Bool, _ entry: QueueEntry, on id: MoldHost.ID) async {
        guard canPause(entry, on: id), let row = current(entry, on: id),
              row.state == (paused ? .queued : .paused) else { return }
        await actOn(row, on: id, paused ? String(localized: "pause that job") : String(localized: "resume that job")) { client in
            if paused { try await client.pauseJob(id: entry.id) } else { try await client.resumeJob(id: entry.id) }
        }
    }

    /// Up or down one place, where the machine will actually put it.
    func move(_ entry: QueueEntry, up: Bool, on id: MoldHost.ID) async {
        guard canMove(entry, on: id) else { return }
        let waiting = (listings[id] ?? []).filter(\.state.isReorderable)
        guard let index = waiting.firstIndex(where: { $0.id == entry.id }),
              up ? index > 0 : index + 1 < waiting.count else { return }
        let neighbour: String? = up ? (index >= 2 ? waiting[index - 2].id : nil) : waiting[index + 1].id
        await moveGroup([entry.id], after: neighbour, on: id)
    }

    /// Where a dragged group lands, as the machine's reorder route reads it:
    /// the nearest row above that the machine can reorder. A held, paused or
    /// running row is not in its index space (`QueueOrder`), and naming one
    /// would send the job to the front -- as the Mac's pane learned.
    static func neighbour(above landing: Int, in groups: [QueueGroup]) -> String? {
        groups[..<min(max(landing, 0), groups.count)].reversed()
            .compactMap { $0.rows.last(where: \.state.isReorderable)?.id }.first
    }

    /// Rows moved as a unit to sit after `neighbour` (`nil`: the front), as
    /// the PATCHes `QueueOrder` plans, in order.
    func moveGroup(_ ids: [String], after neighbour: String?, on id: MoldHost.ID) async {
        let plan = QueueOrder.moves(ids, after: neighbour, in: listings[id] ?? [])
        let rows = plan.compactMap { step in listings[id]?.first { $0.id == step.id } }
        guard !plan.isEmpty, rows.count == plan.count,
              rows.allSatisfy({ canMove($0, on: id) }), let sourceHost = hosts.host(id) else { return }
        let instance = hosts.instanceID(of: id)
        await actOnIDs(plan.map(\.id), on: id, String(localized: "move that job")) { client in
            for step in plan {
                guard self.hosts.instanceID(of: id) == instance,
                      let host = self.hosts.host(id), host == sourceHost, self.hosts.isUp(host),
                      self.listings[id]?.first(where: { $0.id == step.id })?.state.isReorderable == true else { return }
                try await client.reorderJob(id: step.id, position: step.position)
            }
        }
    }

    func retry(_ entry: QueueEntry, on id: MoldHost.ID) async {
        guard canRetry(entry, on: id), let row = current(entry, on: id),
              row.batchId == entry.batchId, row.clientBatchId == entry.clientBatchId,
              let instance = hosts.instanceID(of: id), let authority = row.authority(instanceId: instance) else { return }
        await actOn(row, on: id, String(localized: "try that job again")) { try await $0.retryJob(authority) }
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
            guard let host = hosts.host(id), hosts.isUp(host) else { continue }
            let cancelQueued = hosts.capabilities[id]?.canCancelAllQueued == true
            let affected = (listings[id] ?? []).filter { $0.state == .held || (cancelQueued && ($0.state == .queued || $0.state == .paused)) }
            let instance = hosts.instanceID(of: id)
            await actOnIDs(affected.map(\.id), on: id, String(localized: "empty its queue")) { client in
                if cancelQueued { _ = try await client.cancelAllQueued() }
                for held in affected where held.state == .held {
                    guard self.hosts.instanceID(of: id) == instance,
                          let currentHost = self.hosts.host(id), currentHost == host, self.hosts.isUp(currentHost) else { return }
                    guard let row = self.current(held, on: id), row.state == .held,
                          row.batchId == held.batchId, row.clientBatchId == held.clientBatchId else { continue }
                    _ = try await client.cancelHeldJob(id: held.id)
                }
            }
        }
    }

    private func actOn(_ entry: QueueEntry, on id: MoldHost.ID, _ verb: String,
                       _ body: (any MoldBackend) async throws -> Void) async {
        await actOnIDs([entry.id], on: id, verb, body)
    }

    /// Reserve the exact jobs a single or grouped request affects until its
    /// refreshed listing arrives, so another menu or gesture cannot overlap it.
    private func actOnIDs(_ jobIDs: [String], on id: MoldHost.ID, _ verb: String,
                          _ body: (any MoldBackend) async throws -> Void) async {
        let keys = Set(jobIDs.map { "\(id)|\($0)" })
        guard acting.isDisjoint(with: keys) else { return }
        acting.formUnion(keys)
        defer { acting.subtract(keys) }
        await act(id, verb, body)
    }

    private func act(_ id: MoldHost.ID, _ verb: String, _ body: (any MoldBackend) async throws -> Void) async {
        guard let host = hosts.host(id) else { return }
        do { try await body(hosts.backend(for: host)) } catch { hosts.report(host, doing: verb, error) }
        await poll(id)
    }
}
