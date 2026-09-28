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
        // The arms themselves are `DownloadBoard.apply` (MoldClient), shared
        // with the iPhone app: only `enqueued`/`started` create a row, and a
        // frame about a job this client never saw changes nothing
        // (`downloads.ts:96-97`).
        let before = active[host] ?? [:]
        var forHost = before
        let settled = DownloadBoard.apply(event, to: &forHost)
        guard settled != nil || forHost != before else { return }
        if let settled { remember(settled, on: host) }
        active[host] = forHost.isEmpty ? nil : forHost
        reconcile()
    }

    private func remember(_ job: DownloadJob, on host: MoldHost.ID) {
        var list = finished[host] ?? []
        list.insert(job, at: 0)
        finished[host] = Array(list.prefix(16))
    }
}
