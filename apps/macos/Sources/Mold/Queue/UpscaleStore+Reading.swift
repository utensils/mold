import Foundation
import MoldClient

// Reading the store: what a machine offers, what it is doing, and what this
// app is in the middle of. Split from `UpscaleStore.swift` past the
// file-size advisory; the state and the lifecycle stay there.
@MainActor
extension UpscaleStore {
    /// The upscalers this machine has, as the default-picking policy needs to
    /// see them (`UpscalePlan.defaultUpscaler`).
    func upscalers(on host: MoldHost.ID) -> [UpscalerChoice] {
        models.all(on: host)
            .filter(\.isUpscaler)
            .map { UpscalerChoice(name: $0.name, isDownloaded: $0.isReady) }
    }

    /// Whether this print can be made bigger where it lives.
    ///
    /// Two different answers from one block: a clip needs `video_upscale`
    /// itself, a still additionally needs `gallery_image`, because only that
    /// route publishes the bigger still into the machine's own Library. A
    /// mesh is neither. Absence anywhere is a definitive no -- the action is
    /// then ABSENT from the menu, never present and inert.
    func canUpscale(_ entry: LibraryEntry) -> Bool {
        guard entry.print.trashedAt == nil,
              let capabilities = hosts.capabilities[entry.hostID] else { return false }
        switch entry.print.kind {
        case .clip: return capabilities.canUpscaleClips
        case .picture: return capabilities.canUpscaleStills
        case .mesh: return false
        }
    }

    /// The clip jobs worth drawing, newest machine-name order aside: every
    /// print this app is following, settled or not.
    var live: [(key: Key, job: VideoUpscaleJob)] {
        jobs.map { (key: $0.key, job: $0.value) }
            .sorted { $0.key.filename < $1.key.filename }
    }

    /// The still upscales worth drawing, in a stable order.
    var liveStills: [(key: Key, state: StillUpscale)] {
        stills.map { (key: $0.key, state: $0.value) }
            .sorted { $0.key.filename < $1.key.filename }
    }

    func job(for entry: LibraryEntry) -> VideoUpscaleJob? {
        jobs[Key(host: entry.hostID, filename: entry.print.filename)]
    }

    /// Whether this app is in the middle of making this print bigger --
    /// a request in flight, a clip job still moving, or a still still
    /// running. What the Library's own offer is hidden behind, so the action
    /// cannot be pressed twice from the menu at all.
    func isBusy(with entry: LibraryEntry) -> Bool {
        let key = Key(host: entry.hostID, filename: entry.print.filename)
        return working.contains(key) || UpscalePlan.shouldPoll(jobs[key])
            || stills[key] == .working
    }

    /// Forgets a settled still, so its row leaves the pane.
    func forgetStill(_ key: Key) {
        guard stills[key] != .working else { return }
        stills[key] = nil
    }
}
