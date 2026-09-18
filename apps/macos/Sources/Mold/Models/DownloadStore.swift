import Foundation
import MoldClient

/// Model fetches in progress, per machine -- and, briefly, what just
/// finished. Installing and licence recovery are `DownloadStore+Install.swift`,
/// and watching the live stream is `DownloadStore+Stream.swift`, both split
/// out for size; the stored properties live here because an extension cannot
/// add one.
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

    /// A machine won't fetch a gated model until somebody accepts its
    /// terms. Held rather than reported through the usual funnel, because it
    /// is the one failure the app can resolve on the spot: a sheet shows the
    /// terms and `accepted(_:)` retries the SAME install.
    struct PendingLicense: Identifiable {
        let refusal: LicenseRefusal
        let mismatch: Bool
        let host: MoldHost.ID
        let retry: () async -> Void
        var id: String { refusal.id }
    }

    let hosts: HostStore
    let licenses: LicenseStore
    /// Keyed by host then by the host's job id. Not `private(set)`:
    /// `DownloadStore+Install.swift` writes it too, and `private` does not
    /// cross a file boundary even within one type.
    internal(set) var active: [MoldHost.ID: [String: Progress]] = [:]
    /// Jobs this app watched go terminal, newest first, 16 per machine --
    /// the popover's own record, since workstation retains no server-side history
    /// for it to re-read. Cleared by `clearFinished(on:)`; written from
    /// `DownloadStore+Stream.swift`.
    internal(set) var finished: [MoldHost.ID: [DownloadJob]] = [:]
    /// Written from `DownloadStore+Stream.swift` too.
    internal(set) var streams: [MoldHost.ID: Task<Void, Never>] = [:]
    /// Held rather than reported -- see `PendingLicense`. Written from
    /// `DownloadStore+Install.swift`.
    internal(set) var pendingLicense: PendingLicense?

    init(hosts: HostStore, licenses: LicenseStore) {
        self.hosts = hosts
        self.licenses = licenses
    }

    func progress(for model: String, on host: MoldHost.ID) -> Progress? {
        active[host]?.values.first { $0.model == model }
    }

    func isBusy(_ model: String, on host: MoldHost.ID) -> Bool {
        progress(for: model, on: host) != nil
    }

    func clearFinished(on host: MoldHost.ID) {
        finished[host] = nil
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

    /// Replaces this host's active rows wholesale, from whichever door
    /// answered a full listing: `refresh(on:)`'s GET, or a stream's own
    /// `snapshot` frame. Keyed by job id either way, so a `mold pull` at a
    /// terminal lands on the same row a later frame updates.
    func adopt(_ listing: DownloadsListing, on host: MoldHost.ID) {
        var forHost: [String: Progress] = [:]
        for job in listing.activeJobs + listing.queued {
            forHost[job.id] = Progress(
                model: job.model,
                fraction: job.bytesTotal > 0 ? Double(job.bytesDone) / Double(job.bytesTotal) : nil,
                bytesDone: job.bytesDone, bytesTotal: job.bytesTotal,
                currentFile: job.currentFile, failed: job.error)
        }
        active[host] = forHost.isEmpty ? nil : forHost
    }

    /// One stream per machine with something in flight, and none for a
    /// machine that is gone. The same rule `HostStore` reconciles its event
    /// watchers by, rather than a second mechanism -- and the reason nothing
    /// has to remember to STOP a stream: the method that did had no callers,
    /// so a removed machine kept a live connection for the rest of the launch.
    /// `finished` never keeps a stream open on its own: a host with rows
    /// there and none in `active` is already outside `wanted`.
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
}
