import Foundation
import MoldClient

@MainActor extension LibraryStore {
    /// A successful timeline patch preserves every untouched logical group.
    /// Hidden collections still require the full cross-copy privacy projection.
    func refreshUnreadMedia(changed: [LibraryEntry], removed: [PrintID]) {
        if localSaveRunning { scheduleUnreadRefresh(); return }
        guard hiddenCollectionIDs.isEmpty else { refreshUnreadMedia(); return }
        let previousCount = unreadMedia.count
        var seen = Set<PrintID>()
        let changedTiles = changed.compactMap { liveTimeline.tile(containing: $0.id) }
            .filter { seen.insert($0.id).inserted }
        let changed = unreadMedia.observeChanges(entries: changedTiles, visible: changedTiles, removed: removed,
            loadedHosts: Set(perHost.keys), presentHosts: Set(hosts.hosts.map(\.id)))
        if changed { scheduleUnreadSave() }
        if unreadMedia.count != previousCount { unreadCountChanged?(unreadCount) }
    }

    /// One save per second during a burst, always of the latest ledger. Viewer
    /// reads remain synchronous and quit flushes pending arrival observations.
    private func scheduleUnreadSave() {
        guard unreadSaveTask == nil else { return }
        unreadSaveTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(1)) } catch { return }
            guard let self else { return }
            self.unreadSaveTask = nil
            self.unreadMedia.save(to: self.readDefaults)
        }
    }

    func prepareUnreadInteraction() {
        if unreadRefreshTask != nil {
            unreadRefreshTask?.cancel()
            unreadRefreshTask = nil
            refreshUnreadMediaNow()
        }
        unreadSaveTask?.cancel()
        unreadSaveTask = nil
    }

    func flushUnreadMedia() {
        prepareUnreadInteraction()
        unreadMedia.save(to: readDefaults)
    }

    /// Sync can publish many replicas per display frame. Read the latest state
    /// once per second while transferring, without postponing it until sync ends.
    func refreshUnreadMedia() {
        if localSaveRunning { scheduleUnreadRefresh(); return }
        unreadRefreshTask?.cancel()
        unreadRefreshTask = nil
        refreshUnreadMediaNow()
    }

    private func scheduleUnreadRefresh() {
        guard unreadRefreshTask == nil else { return }
        unreadRefreshTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(1)) } catch { return }
            guard let self else { return }
            self.unreadRefreshTask = nil
            self.refreshUnreadMediaNow()
        }
    }

    private func refreshUnreadMediaNow() {
        let previous = unreadMedia
        let query = LibraryScope.all.resolve(LibraryQuery(), shelves: shelves, hiddenCollectionIDs: hiddenCollectionIDs, privacyEvidence: trashed, privacyLinks: syncPrivacyLinks())
        unreadMedia.observe(entries: items, visible: query.apply(to: items),
                            loadedHosts: Set(perHost.keys), presentHosts: Set(hosts.hosts.map(\.id)))
        if unreadMedia != previous { scheduleUnreadSave() }
        if unreadMedia.count != previous.count { unreadCountChanged?(unreadCount) }
    }
}
