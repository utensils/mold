import Foundation
import MoldClient

/// Model fetches in progress, per machine.
@MainActor
@Observable
final class DownloadStore {
    struct Progress: Hashable {
        var model: String
        var fraction: Double?
        var bytesDone: Int64?
        var bytesTotal: Int64?
        var currentFile: String?
        var failed: String?
    }

    private let hosts: HostStore
    /// Keyed by host then by the host's job id.
    private(set) var active: [MoldHost.ID: [String: Progress]] = [:]
    private(set) var streams: [MoldHost.ID: Task<Void, Never>] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func progress(for model: String, on host: MoldHost.ID) -> Progress? {
        active[host]?.values.first { $0.model == model }
    }

    func isBusy(_ model: String, on host: MoldHost.ID) -> Bool {
        progress(for: model, on: host) != nil
    }

    /// Asks a machine to fetch a model.
    ///
    /// A 409 means it is already queued there, which is the outcome the click
    /// wanted -- the client treats it as success and starts watching.
    func install(_ model: Model, on host: MoldHost) async {
        let client = hosts.backend(for: host)
        do {
            let ticket = try await client.startDownload(DownloadRequest(model: model.name))
            var forHost = active[host.id] ?? [:]
            forHost[ticket.id] = Progress(model: model.name)
            active[host.id] = forHost
            hosts.succeeded(on: host.id)
            reconcile()
        } catch {
            hosts.report(error, on: host.id, doing: "start that download")
        }
    }

    func cancel(jobID: String, on host: MoldHost) async {
        do {
            try await hosts.backend(for: host).cancelDownload(id: jobID)
            hosts.succeeded(on: host.id)
        } catch {
            hosts.report(error, on: host.id, doing: "cancel that download")
        }
        active[host.id]?.removeValue(forKey: jobID)
        reconcile()
    }

    /// One stream per machine with something in flight, and none for a
    /// machine that is gone. The same rule `HostStore` reconciles its event
    /// watchers by, rather than a second mechanism -- and the reason nothing
    /// has to remember to STOP a stream: the method that did had no callers,
    /// so a removed machine kept a live connection for the rest of the launch.
    ///
    /// Called after every change to `active`, and by the root when the machine
    /// list changes.
    func reconcile() {
        let wanted = Set(active.keys).intersection(hosts.hosts.map(\.id))
        for id in streams.keys where !wanted.contains(id) {
            streams.removeValue(forKey: id)?.cancel()
        }
        for id in wanted where streams[id] == nil {
            guard let host = hosts.host(id) else { continue }
            streams[id] = watch(host: host)
        }
    }

    private func watch(host: MoldHost) -> Task<Void, Never> {
        Task { [weak self] in
            // A cancelled task was already taken out of `streams` by
            // `reconcile`, which may have replaced it -- clearing the entry
            // here would take the successor's. Any other exit is a dropped
            // connection nobody knows about yet, so it says so.
            defer { if !Task.isCancelled { self?.streams[host.id] = nil } }
            // A dropped stream just stops the live figures; the download
            // itself belongs to the host and carries on.
            guard let backend = self?.hosts.backend(for: host) else { return }
            do {
                for try await event in backend.downloadEvents() {
                    self?.apply(event, on: host.id)
                }
            } catch {
                self?.active[host.id] = nil
            }
        }
    }

    private func apply(_ event: DownloadEvent, on host: MoldHost.ID) {
        guard let id = event.id else { return }
        var forHost = active[host] ?? [:]

        if event.isTerminal {
            forHost.removeValue(forKey: id)
        } else {
            var progress = forHost[id] ?? Progress(model: event.model ?? "")
            if let model = event.model { progress.model = model }
            progress.fraction = event.fraction ?? progress.fraction
            progress.bytesDone = event.bytesDone ?? progress.bytesDone
            progress.bytesTotal = event.bytesTotal ?? progress.bytesTotal
            progress.currentFile = event.currentFile ?? progress.currentFile
            forHost[id] = progress
        }
        active[host] = forHost.isEmpty ? nil : forHost
        reconcile()
    }

}
