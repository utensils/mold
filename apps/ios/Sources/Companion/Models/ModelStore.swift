import Foundation
import MoldClient

/// What each machine has installed and is fetching (DESIGN.md §5.5): the
/// download board from `/api/downloads/stream` (reduced by the shared
/// `DownloadBoard`), licences, and the Load / Unload / Delete actions. The
/// installed list itself is `HostStore.models`, re-read after every change.
@Observable
final class ModelStore {
    struct PendingLicense: Identifiable {
        let refusal: LicenseRefusal
        let mismatch: Bool
        let host: MoldHost.ID
        let retry: () async -> Void
        var id: String { refusal.id }
    }

    private(set) var active: [MoldHost.ID: [String: DownloadProgress]] = [:]
    private(set) var finished: [MoldHost.ID: [DownloadJob]] = [:]
    private(set) var licences: [MoldHost.ID: [ThirdPartyLicense]] = [:]
    var pendingLicense: PendingLicense?
    /// One line after a delete: "Removed … and freed 6.8 GB."
    var summary: String?

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let queue: QueueStore
    @ObservationIgnored private var streams: [MoldHost.ID: Task<Void, Never>] = [:]
    @ObservationIgnored private var samples: [String: (bytes: Int64, at: Date, rate: Double?)] = [:]
    @ObservationIgnored private var foreground = false

    init(hosts: HostStore, queue: QueueStore) {
        self.hosts = hosts
        self.queue = queue
    }

    // MARK: - Reading

    func progress(for model: String, on host: MoldHost.ID) -> (id: String, row: DownloadProgress)? {
        active[host]?.first { $0.value.model == model }.map { ($0.key, $0.value) }
    }

    func isBusy(_ model: String, on host: MoldHost.ID) -> Bool { progress(for: model, on: host) != nil }

    /// Bytes per second, smoothed, once two frames have been seen.
    func rate(of jobID: String) -> Double? { samples[jobID]?.rate }

    func licence(gating model: String, on host: MoldHost.ID) -> ThirdPartyLicense? {
        licences[host]?.first { $0.requiredBy.contains(model) }
    }

    /// Installed generators first by family, then name -- what the list draws.
    func installed(on host: MoldHost.ID) -> [(family: String, models: [Model])] {
        let rows = (hosts.models[host] ?? []).filter { $0.downloaded == true }
        return Dictionary(grouping: rows, by: \.family)
            .map { ($0.key, $0.value.sorted { $0.headline.localizedStandardCompare($1.headline) == .orderedAscending }) }
            .sorted { $0.family < $1.family }
    }

    /// No answer is not an empty inventory. Say what can actually be known.
    func emptyInventoryMessage(on host: MoldHost) -> String {
        if hosts.models[host.id] == nil {
            switch hosts.reachability(of: host) {
            case .unknown, .checking:
                return String(localized: "Checking installed models on \(host.name)…")
            case .needsKey:
                return String(localized: "Add an API key for \(host.name) to see its installed models.")
            case .down:
                return String(localized: "Connect to \(host.name) to see its installed models.")
            case .up:
                return String(localized: "The installed models on \(host.name) couldn't be read. Pull to refresh.")
            }
        }
        return String(localized: "Nothing installed on \(host.name) yet. Discover has models to fetch.")
    }

    // MARK: - Lifecycle

    /// Foreground: learn what each machine is fetching, and follow it.
    func resume() async {
        foreground = true
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.upHosts { group.addTask { await self.refresh(on: host.id) } }
        }
    }

    /// Background: no streams (`ConnectionSupervisor`).
    func stop() {
        foreground = false
        for task in streams.values { task.cancel() }
        streams = [:]
    }

    func refresh(on id: MoldHost.ID) async {
        guard let host = hosts.host(id) else { return }
        let client = hosts.backend(for: host)
        do {
            adopt(try await client.downloads(), on: id)
        } catch {
            hosts.report(host, doing: String(localized: "list its downloads"), error)
        }
        if hosts.capabilities[id]?.hasLicenses == true, let list = try? await client.licenses() {
            licences[id] = list
        }
        reconcile()
    }

    private func adopt(_ listing: DownloadsListing, on id: MoldHost.ID) {
        let board = DownloadBoard.adopt(listing)
        active[id] = board.isEmpty ? nil : board
    }

    /// A stream per machine with something fetching, while in the foreground.
    private func reconcile() {
        let wanted = foreground ? Set(active.keys).intersection(hosts.hosts.map(\.id)) : []
        for id in streams.keys where !wanted.contains(id) { streams.removeValue(forKey: id)?.cancel() }
        for id in wanted where streams[id] == nil {
            guard let host = hosts.host(id) else { continue }
            let client = hosts.backend(for: host)
            streams[id] = Task { [weak self] in
                do {
                    for try await event in client.downloadEvents() { self?.apply(event, on: id) }
                } catch {}
                if !Task.isCancelled { self?.streams[id] = nil }
            }
        }
    }

    func apply(_ event: DownloadEvent, on id: MoldHost.ID) {
        if case .snapshot = event.effect, let listing = event.listing {
            adopt(listing, on: id)
            return reconcile()
        }
        var board = active[id] ?? [:]
        if let settled = DownloadBoard.apply(event, to: &board) {
            samples[settled.id] = nil
            finished[id] = Array(([settled] + (finished[id] ?? [])).prefix(16))
            // Installed now (or not): the list says so.
            if let host = hosts.host(id) { Task { await hosts.refresh(host) } }
        }
        for (job, row) in board { sample(job, row) }
        active[id] = board.isEmpty ? nil : board
        reconcile()
    }

    private func sample(_ job: String, _ row: DownloadProgress) {
        guard let bytes = row.bytesDone else { return }
        let now = Date.now
        guard let last = samples[job] else { samples[job] = (bytes, now, nil); return }
        let seconds = now.timeIntervalSince(last.at)
        guard seconds >= 0.5 else { return }
        let instant = Double(bytes - last.bytes) / seconds
        let rate = last.rate.map { $0 * 0.7 + instant * 0.3 } ?? instant
        samples[job] = (bytes, now, max(rate, 0))
    }

    // MARK: - Acting

    /// Get, or Repair: the catalog install for a catalog id, a plain pull
    /// otherwise. A gated model asks for its licence first, then tries again.
    func install(_ name: String, on id: MoldHost.ID) async {
        guard let host = hosts.host(id) else { return }
        let client = hosts.backend(for: host)
        let verb = String(localized: "start that download")
        do {
            let jobs: [String] = if Model.isCatalogName(name) {
                try await client.installCatalogEntry(id: name).jobIDs
            } else {
                [try await client.startDownload(DownloadRequest(model: name)).id]
            }
            var board = active[id] ?? [:]
            for job in jobs where board[job] == nil { board[job] = DownloadProgress(model: name) }
            active[id] = board
            hosts.clearFailures(for: id, doing: verb)
            reconcile()
        } catch let MoldClientError.licenseRequired(refusal, mismatch) {
            pendingLicense = PendingLicense(refusal: refusal, mismatch: mismatch, host: id) { [weak self] in
                await self?.install(name, on: id)
            }
        } catch {
            hosts.report(host, doing: verb, error)
        }
    }

    func accept(_ pending: PendingLicense) async {
        guard let host = hosts.host(pending.host) else { return }
        do {
            licences[pending.host] = try await hosts.backend(for: host).acceptLicenses([pending.refusal.acceptance])
            pendingLicense = nil
            await pending.retry()
        } catch {
            hosts.report(host, doing: String(localized: "accept the licence for \(pending.refusal.name)"), error)
        }
    }

    func cancel(job: String, on id: MoldHost.ID) async {
        guard let host = hosts.host(id) else { return }
        do { try await hosts.backend(for: host).cancelDownload(id: job) } catch {
            hosts.report(host, doing: String(localized: "cancel that download"), error)
        }
        active[id]?.removeValue(forKey: job)
        if active[id]?.isEmpty == true { active[id] = nil }
        reconcile()
    }

    /// The held row's Pull: fetch the model, and only once that download
    /// settles as a success, try the job again -- never on a cancelled or
    /// failed one, where it would just hold again (the Mac's rule).
    func pullThenRetry(_ model: String, entry: QueueEntry, on id: MoldHost.ID) {
        Task {
            if !isBusy(model, on: id) { await install(model, on: id) }
            guard await settles(model, on: id) else { return }
            await queue.retry(entry, on: id)
        }
    }

    private func settles(_ model: String, on id: MoldHost.ID) async -> Bool {
        let deadline = Date.now.addingTimeInterval(60 * 60)
        while isBusy(model, on: id) || pendingLicense?.host == id {
            guard Date.now < deadline else { return false }
            do { try await Task.sleep(for: .milliseconds(250)) } catch { return false }
        }
        return finished[id]?.first { $0.model == model }?.status == .completed
    }

    func load(_ model: Model, on id: MoldHost.ID) async {
        await act(id, String(localized: "load \(model.headline)")) { try await $0.loadModel(model.name, gpu: nil) }
    }

    func unload(_ model: Model, on id: MoldHost.ID) async {
        await act(id, String(localized: "unload \(model.headline)")) { try await $0.unloadModel(model: model.name, gpu: nil) }
    }

    func delete(_ model: Model, on id: MoldHost.ID) async {
        await act(id, String(localized: "delete \(model.headline)")) { client in
            let removal = try await client.deleteModel(model.name)
            self.summary = removal.summary(headline: model.headline)
        }
    }

    func components(of model: Model, on id: MoldHost.ID) async -> [ModelComponentStatus] {
        guard let host = hosts.host(id) else { return [] }
        do { return try await hosts.backend(for: host).modelComponents(model.name).components } catch {
            hosts.report(host, doing: String(localized: "list the files \(model.headline) needs"), error)
            return []
        }
    }

    private func act(_ id: MoldHost.ID, _ verb: String, _ body: (any MoldBackend) async throws -> Void) async {
        guard let host = hosts.host(id) else { return }
        do {
            try await body(hosts.backend(for: host))
            hosts.clearFailures(for: id, doing: verb)
        } catch {
            hosts.report(host, doing: verb, error)
        }
        await hosts.refresh(host)
    }
}
