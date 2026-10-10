import Foundation

/// Copying is opt-in for the current application session. The timer belongs to
/// the store, so changing destinations does not silently stop it.
@MainActor
@Observable
final class LibrarySyncSession {
    private(set) var isEnabled = false
    private(set) var nextRun: Date?
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private let defaults: UserDefaults
    private var interval: TimeInterval
    private var waitTask: Task<Void, Never>?
    private static let intervalKey = "library.syncIntervalMinutes.v1"
    var intervalMinutes: Int {
        get { Int((interval / 60).rounded()) }
        set {
            interval = Double(min(1440, max(1, newValue))) * 60
            defaults.set(intervalMinutes, forKey: Self.intervalKey)
            if nextRun != nil {
                nextRun = Date().addingTimeInterval(interval)
                waitTask?.cancel()
            }
        }
    }
    private var acknowledged: Set<String>
    private static let issueKey = "library.acknowledgedSyncIssues.v1"

    init(defaults: UserDefaults = AppStorageSuite.defaults, interval: TimeInterval? = nil) {
        self.defaults = defaults
        let persisted = defaults.object(forKey: Self.intervalKey) as? Int ?? 5
        self.interval = interval ?? Double(min(1440, max(1, persisted))) * 60
        // Older releases persisted multiple source hashes as one colon-joined
        // token. Expand those receipts without forgetting acknowledged sources.
        acknowledged = Set((defaults.stringArray(forKey: Self.issueKey) ?? [])
            .flatMap { $0.split(separator: ":").map(String.init) })
    }

    func start(in library: LibraryStore) {
        guard !isEnabled, library.localSaveTask == nil, !library.localSaveRunning else { return }
        isEnabled = true
        generation = UUID()
        let current = generation
        task = Task { [weak self, weak library] in
            while !Task.isCancelled, self?.isEnabled == true, self?.generation == current {
                // A selected Save can be running when the timer wakes. Wait
                // for the following cycle rather than overlap transfer runs.
                if library?.localSaveRunning == false, library?.localSaveTask == nil {
                    await library?.syncAllLocally()
                }
                guard let interval = self?.interval, self?.isEnabled == true, self?.generation == current else { break }
                self?.nextRun = Date().addingTimeInterval(interval)
                repeat {
                    guard let self, let deadline = self.nextRun else { break }
                    let wait = Task<Void, Never> { do { try await Task.sleep(for: .seconds(max(0, deadline.timeIntervalSinceNow))) } catch {} }
                    self.waitTask = wait
                    await wait.value
                    guard self.generation == current else { return }
                    self.waitTask = nil
                } while !Task.isCancelled && self?.isEnabled == true && (self?.nextRun?.timeIntervalSinceNow ?? 0) > 0
                guard self?.generation == current else { return }
                self?.nextRun = nil
            }
            if self?.generation == current {
                self?.nextRun = nil
                self?.task = nil
            }
        }
    }

    func stop(in library: LibraryStore) {
        isEnabled = false
        nextRun = nil
        waitTask?.cancel()
        if library.localSaveRunning { library.localSaveStopRequested = true }
        else { task?.cancel(); task = nil }
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
