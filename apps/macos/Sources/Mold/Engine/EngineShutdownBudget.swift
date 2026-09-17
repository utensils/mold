import Foundation

/// How long the engine is given to drain when Mold quits.
///
/// It is the SERVER's budget, not one this app invented: `run_server` joins
/// its GPU owner threads for `resolve_shutdown_abort_secs()` and releases the
/// gallery writer leases on the line after that join
/// (`crates/mold-server/src/lib.rs:1594-1620`). The app used to wait 8 s and
/// discard the answer, so quitting during a render cut the drain and left the
/// lease behind — while `Info.plist`'s `NSSupportsSuddenTermination=false`
/// promised macOS would wait (review 05-M7).
///
/// `EngineShutdownBudgetTests` reads the Rust constant, so the two cannot
/// drift apart silently.
enum EngineShutdownBudget {
    /// `DEFAULT_SHUTDOWN_ABORT_SECS`.
    static let defaultSeconds: UInt64 = 45
    /// `SHUTDOWN_ABORT_SECS_ENV`, applied with the server's own rule: at
    /// least one second, anything unparseable ignored.
    static let environmentName = "MOLD_SHUTDOWN_ABORT_SECS"

    static var seconds: UInt64 {
        resolve(ProcessInfo.processInfo.environment[environmentName])
    }

    static var milliseconds: UInt64 { seconds * 1_000 }

    static func resolve(_ raw: String?) -> UInt64 {
        guard let parsed = UInt64(raw?.trimmingCharacters(in: .whitespaces) ?? ""), parsed >= 1
        else { return defaultSeconds }
        return parsed
    }
}
