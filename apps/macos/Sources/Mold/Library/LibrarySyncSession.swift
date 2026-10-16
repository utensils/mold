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
    /// An automatic run is waiting for This Mac's engine to connect.
    private(set) var waitingForLocalEngine = false
    private var terminating = false
    private let defaults: UserDefaults
    private let debounce: Duration
    static let keepInSyncKey = "library.keepInSync.v1"
    static let retiredIntervalKey = "library.syncIntervalMinutes.v1"
    private var acknowledged: Set<String>
    private static let issueKey = "library.acknowledgedSyncIssues.v1"

    init(defaults: UserDefaults = AppStorageSuite.defaults, debounce: Duration = .seconds(2)) {
        self.defaults = defaults
        self.debounce = debounce
        keepInSync = defaults.bool(forKey: Self.keepInSyncKey)
        // Retired with the timer; nothing reads it any more.
        defaults.removeObject(forKey: Self.retiredIntervalKey)
        // Older releases persisted multiple source hashes as one colon-joined
        // token. Expand those receipts without forgetting acknowledged sources.
        acknowledged = Set((defaults.stringArray(forKey: Self.issueKey) ?? [])
            .flatMap { $0.split(separator: ":").map(String.init) })
    }

    /// One run now. A run already in flight is never overlapped: with Keep in
    /// Sync on, the press queues one follow-up instead of being dropped.
    func syncNow(in library: LibraryStore) {
        guard !terminating else { return }
        guard library.localSaveTask == nil, !library.localSaveRunning else {
            if keepInSync { rerunPending = true }
            return
        }
        // A Stop that landed after an earlier run's last check would
        // otherwise end this one before it starts.
        library.localSaveStopRequested = false
        library.localSaveTask = Task { [weak library] in
            await library?.syncAllLocally()
        }
    }

    func setKeepInSync(_ enabled: Bool, in library: LibraryStore) {
        keepInSync = enabled
        defaults.set(enabled, forKey: Self.keepInSyncKey)
        if enabled { noteRemoteChange(in: library, after: .zero) } else {
            debounceTask?.cancel()
            debounceTask = nil
            rerunPending = false
            waitingForLocalEngine = false
        }
    }

    /// A remote machine announced a gallery change (or connected). Bursts
    /// coalesce into one run after `debounce`; a change landing mid-run queues
    /// exactly one follow-up. An automatic run never raises the "start This
    /// Mac's engine" alert: it waits for the engine to connect instead.
    func noteRemoteChange(in library: LibraryStore) {
        noteRemoteChange(in: library, after: debounce)
    }

    private func noteRemoteChange(in library: LibraryStore, after delay: Duration) {
        guard keepInSync, !terminating else { return }
        debounceTask?.cancel()
        debounceTask = Task { [weak self, weak library] in
            if delay > .zero { do { try await Task.sleep(for: delay) } catch { return } }
            guard let self, let library, self.keepInSync, !self.terminating else { return }
            self.debounceTask = nil
            guard let local = library.hosts.host(MoldEngine.localHostID), library.hosts.isUp(local) else {
                self.waitingForLocalEngine = true
                return
            }
            self.syncNow(in: library)
        }
    }

    /// This Mac's engine connected: release a run that was waiting for it.
    func localEngineAvailable(in library: LibraryStore) {
        guard waitingForLocalEngine else { return }
        waitingForLocalEngine = false
        noteRemoteChange(in: library)
    }

    /// Every finished copy run -- Sync or an explicit Save -- ends here, so a
    /// change that arrived while it ran is never stranded.
    func runFinished(in library: LibraryStore) {
        guard rerunPending else { return }
        rerunPending = false
        guard keepInSync, !terminating else { return }
        syncNow(in: library)
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

    /// Quit: nothing may keep transferring -- or start -- while the engine
    /// drains.
    func cancelForQuit(in library: LibraryStore) {
        terminating = true
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
