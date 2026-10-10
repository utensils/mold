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
        acknowledged = Set(defaults.stringArray(forKey: Self.issueKey) ?? [])
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
    /// The report control reflects saved state, including when reopened after
    /// an acknowledged issue recurs. New eligible issues leave it unchecked.
    var syncIssueAcknowledgment: Bool {
        get {
            let keys = Array(localSaveIssueKeys.values)
            return !keys.isEmpty && !syncSession.hasNewIssues(keys)
        }
        set {
            let keys = Array(localSaveIssueKeys.values)
            if newValue { syncSession.acknowledge(keys) }
            else { syncSession.unacknowledge(keys) }
        }
    }
}
