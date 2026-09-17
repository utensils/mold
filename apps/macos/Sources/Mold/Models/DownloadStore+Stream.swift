import Foundation
import MoldClient

// Watching one machine's `GET /api/downloads/stream` and folding each frame
// into `active` and `finished`. Split from `DownloadStore.swift` for size.
extension DownloadStore {
    // Not `private`: `reconcile()`, in the core file, is what starts one of
    // these, and `private` does not cross a file boundary even within one
    // type.
    func watch(host: MoldHost) -> Task<Void, Never> {
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

    /// What each frame is allowed to do is `DownloadEvent.effect`'s answer,
    /// not this method's -- "has an id and is not terminal" was the old test,
    /// and `catalog_ready` passes it while naming a CATALOG entry rather than
    /// a job. See that type for what that invented.
    func apply(_ event: DownloadEvent, on host: MoldHost.ID) {
        // The FIRST frame on every connection -- how this store learns about
        // a job it never started, from a `mold pull` at a terminal or from
        // the web app on the same machine.
        if case .snapshot = event.effect, let listing = event.listing {
            adopt(listing, on: host)
            reconcile()
            return
        }
        guard let id = event.id else { return }
        var forHost = active[host] ?? [:]

        switch event.effect {
        case .settle:
            let last = forHost.removeValue(forKey: id)
            remember(id: id, event: event, last: last, on: host)
        case .forget:
            forHost.removeValue(forKey: id)
        case .introduce:
            forHost[id] = moved(forHost[id] ?? Progress(model: event.model ?? ""), by: event)
        case .update:
            // Only a row already known. A frame about a job this client has
            // never seen is not a reason to invent one (`downloads.ts:96-97`).
            guard let known = forHost[id] else { return }
            forHost[id] = moved(known, by: event)
        case .snapshot, .ignore:
            return
        }
        active[host] = forHost.isEmpty ? nil : forHost
        reconcile()
    }

    /// One row, plus whatever this frame actually carried. Every field is
    /// optional on the wire and absent means "unchanged", never zero.
    private func moved(_ row: Progress, by event: DownloadEvent) -> Progress {
        var progress = row
        if let model = event.model { progress.model = model }
        progress.fraction = event.fraction ?? progress.fraction
        progress.bytesDone = event.bytesDone ?? progress.bytesDone
        progress.bytesTotal = event.bytesTotal ?? progress.bytesTotal
        progress.currentFile = event.currentFile ?? progress.currentFile
        return progress
    }

    private func remember(id: String, event: DownloadEvent, last: Progress?, on host: MoldHost.ID) {
        let status: JobStatus =
            switch event.type {
            case "job_done": .completed
            case "job_cancelled": .cancelled
            default: .failed
            }
        let job = DownloadJob(
            id: id, model: event.model ?? last?.model ?? "", status: status,
            bytesDone: event.bytesDone ?? last?.bytesDone ?? 0,
            bytesTotal: event.bytesTotal ?? last?.bytesTotal ?? 0,
            currentFile: event.currentFile ?? last?.currentFile, error: event.error ?? last?.failed)
        var list = finished[host] ?? []
        list.insert(job, at: 0)
        finished[host] = Array(list.prefix(16))
    }
}
