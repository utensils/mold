import Foundation
import OSLog

/// No Observable state or main-thread timers: disabled builds schedule no work.
/// All shared mutable state is guarded by `lock`; callbacks retain this service weakly.
nonisolated final class PerformanceDiagnostics: @unchecked Sendable {
    static let shared = PerformanceDiagnostics()
    private let logger = Logger(subsystem: "io.mold.native", category: "Performance")
    private let signposter = OSSignposter(subsystem: "io.mold.native", category: "Performance")
    private let lock = NSLock()
    private var heartbeat = PerformanceHeartbeat()
    private var policy = PerformancePolicy()
    private var timer: DispatchSourceTimer?
    private var active = false
    private var generation: UInt64 = 0
    let isEnabled: Bool

    struct Interval {
        fileprivate let operation: PerformanceOperation
        fileprivate let started: TimeInterval
        fileprivate let state: OSSignpostIntervalState
    }

    init(environment: [String: String] = ProcessInfo.processInfo.environment,
         version: String = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "") {
        isEnabled = PerformancePolicy.enabled(environment: environment, version: version)
    }

    /// Start once from the composition root, after the embedded engine bootstraps.
    func start() {
        guard isEnabled else { return }
        lock.lock()
        guard timer == nil else { lock.unlock(); return }
        let source = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "io.mold.performance", qos: .utility))
        timer = source
        lock.unlock()
        source.schedule(deadline: .now() + 1, repeating: 1, leeway: .milliseconds(250))
        source.setEventHandler { [weak self] in self?.tick() }
        source.resume()
        logger.notice("diagnostics_started heartbeat_seconds=1 hang_threshold_seconds=2 report_interval_seconds=30")
    }

    /// Activation is an AppKit lifecycle fact, never SwiftUI view state.
    func setActive(_ active: Bool) {
        lock.lock()
        self.active = active
        generation &+= 1
        heartbeat = PerformanceHeartbeat()
        lock.unlock()
    }

    func begin(_ operation: PerformanceOperation) -> Interval? {
        guard isEnabled else { return nil }
        let state = signposter.beginInterval("NativeOperation", id: signposter.makeSignpostID(), "operation=\(operation.rawValue, privacy: .public)")
        return Interval(operation: operation, started: ProcessInfo.processInfo.systemUptime, state: state)
    }

    func end(_ interval: Interval?) {
        guard let interval else { return }
        signposter.endInterval("NativeOperation", interval.state)
        let now = ProcessInfo.processInfo.systemUptime
        let elapsed = now - interval.started
        lock.lock()
        let report = policy.shouldReport(interval.operation, seconds: elapsed, now: now)
        lock.unlock()
        if report {
            logger.notice("slow_operation operation=\(interval.operation.rawValue, privacy: .public) elapsed_ms=\(Int(elapsed * 1000), privacy: .public)")
        }
    }

    private func tick() {
        lock.lock()
        guard active else { lock.unlock(); return }
        let expectedGeneration = generation
        let action = heartbeat.tick(now: ProcessInfo.processInfo.systemUptime)
        lock.unlock()
        switch action {
        case .none: break
        case .enqueue:
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.lock.lock()
                if self.generation == expectedGeneration { self.heartbeat.acknowledge() }
                self.lock.unlock()
            }
        case .stalled(let seconds):
            logger.warning("main_thread_unresponsive pending_ms=\(Int(seconds * 1000), privacy: .public)")
        }
    }

    deinit { timer?.cancel() }
}
