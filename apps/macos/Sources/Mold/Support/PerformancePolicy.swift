import Foundation

/// A fixed vocabulary prevents instrumentation from accidentally recording user data.
nonisolated enum PerformanceOperation: String, CaseIterable, Sendable {
    case libraryCounts
    case sourceInventory
    case librarySync
    case libraryProjection
}

nonisolated struct PerformancePolicy: Sendable {
    private var lastReport: [PerformanceOperation: TimeInterval] = [:]

    static func enabled(environment: [String: String], version: String) -> Bool {
        if let override = environment["MOLD_PERFORMANCE_DIAGNOSTICS"] { return override == "1" }
        return version.contains("nightly")
    }

    mutating func shouldReport(_ operation: PerformanceOperation, seconds: TimeInterval, now: TimeInterval) -> Bool {
        guard seconds >= 0.1 else { return false }
        if let last = lastReport[operation], now - last < 30 { return false }
        lastReport[operation] = now
        return true
    }
}

/// At most one queued main-thread ping, even during a prolonged hang.
nonisolated struct PerformanceHeartbeat: Sendable {
    enum Action: Equatable { case none, enqueue, stalled(seconds: TimeInterval) }
    private var pendingSince: TimeInterval?
    private var lastTick: TimeInterval?
    private var lastReport: TimeInterval?

    mutating func tick(now: TimeInterval) -> Action {
        defer { lastTick = now }
        if let lastTick, now - lastTick > 10 {
            // Sleep/wake and suspension are not evidence of main-thread work.
            if pendingSince != nil { pendingSince = now }
            return .none
        }
        guard let pendingSince else {
            pendingSince = now
            return .enqueue
        }
        let delay = now - pendingSince
        guard delay >= 2, lastReport.map({ now - $0 >= 30 }) ?? true else { return .none }
        lastReport = now
        return .stalled(seconds: delay)
    }

    mutating func acknowledge() { pendingSince = nil }
}
