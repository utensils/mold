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

    /// Keyed by host then by the host's job id.
    private(set) var active: [MoldHost.ID: [String: Progress]] = [:]
    private(set) var failure: String?
    private var streams: [MoldHost.ID: Task<Void, Never>] = [:]

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
    func install(_ model: Model, on host: MoldHost, backend: any MoldBackend) async {
        guard let client = backend as? HTTPBackend else { return }
        do {
            let ticket = try await client.startDownload(DownloadRequest(model: model.name))
            var forHost = active[host.id] ?? [:]
            forHost[ticket.id] = Progress(model: model.name)
            active[host.id] = forHost
            watch(host: host, backend: client)
        } catch {
            failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    func cancel(jobID: String, on host: MoldHost, backend: any MoldBackend) async {
        guard let client = backend as? HTTPBackend else { return }
        try? await client.cancelDownload(id: jobID)
        active[host.id]?.removeValue(forKey: jobID)
    }

    /// One stream per machine, however many models are being fetched on it.
    private func watch(host: MoldHost, backend: HTTPBackend) {
        guard streams[host.id] == nil else { return }
        streams[host.id] = Task { [weak self] in
            defer { self?.streams[host.id] = nil }
            // A dropped stream just stops the live figures; the download
            // itself belongs to the host and carries on.
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
    }

    /// Stops every stream. Called when the pane goes away so a background
    /// connection per machine does not outlive the screen that wanted it.
    func stopWatching() {
        streams.values.forEach { $0.cancel() }
        streams.removeAll()
    }
}
