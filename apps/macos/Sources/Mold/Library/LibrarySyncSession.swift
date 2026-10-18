import Foundation
import MoldClient

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
    /// The queued follow-up was a press: it plans every print.
    private var rerunFull = false
    private struct PendingRun {
        let destination: MoldHost
        let instance: String?
        var fullPlan: Bool
        var selection: [LibraryEntry]?
        var automatic: Bool
    }
    private var queued: [PendingRun] = []
    private var destinations: Set<String>
    private var localIdentity: String?
    private static let localIdentityKey = "library.keepInSync.localIdentity.v2"
    private static let destinationsKey = "library.keepInSync.destinations.v2"
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
        destinations = Set(defaults.stringArray(forKey: Self.destinationsKey) ?? [])
        localIdentity = defaults.string(forKey: Self.localIdentityKey)
        // Retired with the timer; nothing reads it any more.
        defaults.removeObject(forKey: Self.retiredIntervalKey)
        // Older releases persisted multiple source hashes as one colon-joined
        // token. Expand those receipts without forgetting acknowledged sources.
        acknowledged = Set((defaults.stringArray(forKey: Self.issueKey) ?? [])
            .flatMap { $0.split(separator: ":").map(String.init) })
    }

    /// One run now. A run already in flight is never overlapped: with Keep in
    /// Sync on, the press queues one follow-up instead of being dropped.
    /// A press plans every print; Keep in Sync's own runs plan from cursors.
    func syncNow(in library: LibraryStore) {
        guard let local = library.hosts.host(MoldEngine.localHostID) else { return }
        start(in: library, destination: local, fullPlan: true)
    }

    func syncNow(to destination: MoldHost, in library: LibraryStore) {
        start(in: library, destination: destination, fullPlan: true)
    }

    func copy(_ selection: [LibraryEntry], to destination: MoldHost, in library: LibraryStore) {
        start(in: library, destination: destination, fullPlan: true, selection: selection)
    }

    func keepInSync(to destination: MoldHost, in library: LibraryStore) -> Bool {
        if destination.id == MoldEngine.localHostID {
            if keepInSync && localIdentity == nil, library.hosts.instanceID(of: destination.id) != nil {
                localIdentity = identity(destination, in: library)
                defaults.set(localIdentity, forKey: Self.localIdentityKey)
            }
            return keepInSync && (localIdentity == nil || localIdentity == identity(destination, in: library))
        }
        return destinations.contains(identity(destination, in: library))
    }

    private func identity(_ destination: MoldHost, in library: LibraryStore) -> String {
        [destination.id.uuidString, destination.baseURL.absoluteString,
         library.hosts.instanceID(of: destination.id) ?? ""].joined(separator: "\n")
    }

    func setKeepInSync(_ enabled: Bool, to destination: MoldHost, in library: LibraryStore) {
        if destination.id == MoldEngine.localHostID { setKeepInSync(enabled, in: library); return }
        let key = identity(destination, in: library)
        if enabled { destinations.insert(key) } else { destinations.remove(key) }
        defaults.set(Array(destinations).sorted(), forKey: Self.destinationsKey)
        if enabled { start(in: library, destination: destination, fullPlan: false, automatic: true) }
        else { queued.removeAll { $0.destination.id == destination.id && $0.automatic } }
    }

    private func start(in library: LibraryStore, destination: MoldHost, fullPlan: Bool,
                       selection: [LibraryEntry]? = nil, automatic: Bool = false) {
        guard !terminating, library.hosts.host(destination.id) == destination else { return }
        let run = PendingRun(destination: destination, instance: library.hosts.instanceID(of: destination.id),
            fullPlan: fullPlan, selection: selection, automatic: automatic)
        guard library.localSaveTask == nil, !library.localSaveRunning else {
            if let index = queued.firstIndex(where: { $0.destination == destination && $0.instance == run.instance
                && ($0.selection == nil) == (selection == nil) }) {
                queued[index].fullPlan = queued[index].fullPlan || fullPlan
                queued[index].automatic = queued[index].automatic && automatic
                if let selection {
                    var ids = Set(queued[index].selection?.map(\.id) ?? [])
                    queued[index].selection?.append(contentsOf: selection.filter { ids.insert($0.id).inserted })
                }
            } else { queued.append(run) }
            rerunPending = true
            rerunFull = rerunFull || fullPlan
            return
        }
        guard !automatic || library.hosts.isUp(destination) else {
            if destination.id == MoldEngine.localHostID { waitingForLocalEngine = true }
            return
        }
        library.localSaveStopRequested = false
        library.localSaveTask = Task { [weak library] in
            guard let library else { return }
            if let selection { await library.copy(selection, to: destination) }
            else { await library.syncAll(to: destination, fullPlan: fullPlan, automatic: automatic) }
        }
    }

    func setKeepInSync(_ enabled: Bool, in library: LibraryStore) {
        keepInSync = enabled
        if enabled, let local = library.hosts.host(MoldEngine.localHostID) {
            localIdentity = identity(local, in: library)
            defaults.set(localIdentity, forKey: Self.localIdentityKey)
        }
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
        guard (keepInSync || !destinations.isEmpty), !terminating else { return }
        debounceTask?.cancel()
        debounceTask = Task { [weak self, weak library] in
            if delay > .zero { do { try await Task.sleep(for: delay) } catch { return } }
            guard let self, let library, (self.keepInSync || !self.destinations.isEmpty), !self.terminating else { return }
            self.debounceTask = nil
            if self.keepInSync {
                if let local = library.hosts.host(MoldEngine.localHostID), library.hosts.isUp(local), self.keepInSync(to: local, in: library) {
                    self.start(in: library, destination: local, fullPlan: false, automatic: true)
                } else { self.waitingForLocalEngine = true }
            }
            for host in library.hosts.hosts where host.id != MoldEngine.localHostID && self.keepInSync(to: host, in: library) {
                self.start(in: library, destination: host, fullPlan: false, automatic: true)
            }
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
        rerunPending = false
        rerunFull = false
        guard !terminating else { queued.removeAll(); return }
        while !queued.isEmpty {
            let run = queued.removeFirst()
            guard library.hosts.host(run.destination.id) == run.destination,
                  library.hosts.instanceID(of: run.destination.id) == run.instance else { continue }
            if run.automatic && !keepInSync(to: run.destination, in: library) { continue }
            rerunPending = !queued.isEmpty
            start(in: library, destination: run.destination, fullPlan: run.fullPlan,
                selection: run.selection, automatic: run.automatic)
            return
        }
    }

    /// Stops the current run and anything queued behind it. "Keep This Mac in
    /// Sync" stays as chosen; the next remote change starts a fresh run.
    func stop(in library: LibraryStore) {
        debounceTask?.cancel()
        debounceTask = nil
        rerunPending = false
        rerunFull = false
        queued.removeAll()
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
