import Foundation

/// Library Sync runs when asked. "Sync All to This Mac" is one run; the
/// opt-in "Keep This Mac in Sync" mode mirrors again whenever a remote
/// machine announces a gallery change (and once when it is turned on or a
/// machine reconnects). There is no timer: an idle fleet costs nothing.
@MainActor
@Observable
final class LibrarySyncSession {
    /// The persisted "Keep This Mac in Sync" choice. Off by default.
    private(set) var keepInSync: Bool
    /// A change arrived while a run was in flight; one more run follows it.
    private(set) var rerunPending = false
    private var debounceTask: Task<Void, Never>?
    private let defaults: UserDefaults
    private let debounce: Duration
    static let keepInSyncKey = "library.keepInSync.v1"
    private var acknowledged: Set<String>
    private static let issueKey = "library.acknowledgedSyncIssues.v1"

    init(defaults: UserDefaults = AppStorageSuite.defaults, debounce: Duration = .seconds(2)) {
        self.defaults = defaults
        self.debounce = debounce
        keepInSync = defaults.bool(forKey: Self.keepInSyncKey)
        // Older releases persisted multiple source hashes as one colon-joined
        // token. Expand those receipts without forgetting acknowledged sources.
        acknowledged = Set((defaults.stringArray(forKey: Self.issueKey) ?? [])
            .flatMap { $0.split(separator: ":").map(String.init) })
    }

    /// One run now. A run already in flight is never overlapped.
    func syncNow(in library: LibraryStore) {
        guard library.localSaveTask == nil, !library.localSaveRunning else { return }
        library.localSaveTask = Task { [weak self, weak library] in
            await library?.syncAllLocally()
            guard let self, let library else { return }
            if self.rerunPending, self.keepInSync {
                self.rerunPending = false
                self.syncNow(in: library)
            }
        }
    }

    func setKeepInSync(_ enabled: Bool, in library: LibraryStore) {
        keepInSync = enabled
        defaults.set(enabled, forKey: Self.keepInSyncKey)
        if enabled { syncNow(in: library) } else {
            debounceTask?.cancel()
            debounceTask = nil
            rerunPending = false
        }
    }

    /// A remote machine announced a gallery change (or reconnected). Bursts
    /// coalesce into one run after `debounce`; a change landing mid-run queues
    /// exactly one follow-up run.
    func noteRemoteChange(in library: LibraryStore) {
        guard keepInSync else { return }
        debounceTask?.cancel()
        debounceTask = Task { [weak self, weak library, debounce] in
            do { try await Task.sleep(for: debounce) } catch { return }
            guard let self, let library, self.keepInSync else { return }
            self.debounceTask = nil
            if library.localSaveTask != nil || library.localSaveRunning {
                self.rerunPending = true
            } else {
                self.syncNow(in: library)
            }
        }
    }

    /// Stops the current run and anything queued behind it. "Keep This Mac in
    /// Sync" stays as chosen; the next remote change starts a fresh run.
    func stop(in library: LibraryStore) {
        debounceTask?.cancel()
        debounceTask = nil
        rerunPending = false
        if library.localSaveRunning || library.localSaveTask != nil {
            library.localSaveStopRequested = true
        }
    }

    /// Quit: nothing may keep transferring while the engine drains.
    func cancelForQuit(in library: LibraryStore) {
        stop(in: library)
        library.localSaveTask?.cancel()
    }

    func hasNewIssues(_ keys: [String]) -> Bool {
        keys.contains { !acknowledged.contains($0) }
    }

    func acknowledge(_ keys: [String]) {
        acknowledged.formUnion(keys)
        defaults.set(Array(acknowledged).sorted(), forKey: Self.issueKey)
    }

    func unacknowledge(_ keys: [String]) {
        acknowledged.subtract(keys)
        defaults.set(Array(acknowledged).sorted(), forKey: Self.issueKey)
    }

    func resetAcknowledgments() {
        acknowledged.removeAll()
        defaults.removeObject(forKey: Self.issueKey)
    }
}

@MainActor
extension LibraryStore {
    static func legacyInputNotice(count: Int) -> String {
        guard count > 0 else { return "" }
        return count == 1
            ? " Original inputs are unavailable on the source machine for 1 legacy print; its finished file is saved."
            : " Original inputs are unavailable on the source machine for \(count) legacy prints; their finished files are saved."
    }

    func isUnacknowledgedSyncIssue(_ message: String) -> Bool {
        guard let keys = localSaveIssueKeys[message] else { return true }
        return syncSession.hasNewIssues(keys.split(separator: ":").map(String.init))
    }

    var syncHasUnacknowledgedIssues: Bool {
        (localSaveFailures + localSaveNotices).contains(where: isUnacknowledgedSyncIssue)
    }

    /// Keep the complete report for Details; suppress only exact acknowledged
    /// source notices in the recurring status. Connectivity failures lack keys.
    var syncBannerReport: String {
        var report = localSaveReport
        if !localSaveNotices.isEmpty {
            report = report.replacingOccurrences(of: Self.legacyInputNotice(count: localSaveNotices.count),
                with: Self.legacyInputNotice(count: localSaveNotices.filter(isUnacknowledgedSyncIssue).count))
        }
        if !localSaveFailures.isEmpty {
            let count = localSaveFailures.filter(isUnacknowledgedSyncIssue).count
            report = report.replacingOccurrences(of: " \(localSaveFailures.count) issues need attention.",
                with: count == 0 ? "" : " \(count) issues need attention.")
        }
        return report
    }

    /// The report control reflects saved state, including when reopened after
    /// an acknowledged issue recurs. New eligible issues leave it unchecked.
    var syncIssueAcknowledgment: Bool {
        get {
            let keys = localSaveIssueKeys.values.flatMap { $0.split(separator: ":").map(String.init) }
            return !keys.isEmpty && !syncSession.hasNewIssues(keys)
        }
        set {
            let keys = localSaveIssueKeys.values.flatMap { $0.split(separator: ":").map(String.init) }
            if newValue { syncSession.acknowledge(keys) }
            else { syncSession.unacknowledge(keys) }
        }
    }
}
