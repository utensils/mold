import Foundation

/// How long the engine is given to drain when Mold quits, and the ONE number
/// the app tells anyone about it.
///
/// It is the SERVER's budget plus what sits outside it, not one this app
/// invented: `run_server` joins its GPU owner threads for
/// `resolve_shutdown_abort_secs()` and releases the gallery writer leases on
/// the line AFTER that join (`crates/mold-server/src/lib.rs:1594-1620`), with
/// the HTTP drain ahead of it and `drop(runtime)` behind it. Waiting exactly
/// the server's figure therefore expires while the engine is still writing,
/// which is how the first version of this came to strip a live writer's lease.
/// The app used to allow 8 s and discard the join's answer entirely, while
/// `Info.plist`'s `NSSupportsSuddenTermination=false` promised macOS would
/// wait (review 05-M7, F1).
///
/// `EngineShutdownBudgetTests` reads the Rust constant, so the two cannot
/// drift apart.
enum EngineShutdownBudget {
    /// `DEFAULT_SHUTDOWN_ABORT_SECS`.
    static let defaultSeconds: UInt64 = 45
    /// `SHUTDOWN_ABORT_SECS_ENV`, applied with the server's own rule: at
    /// least one second, anything unparseable ignored.
    static let environmentName = "MOLD_SHUTDOWN_ABORT_SECS"
    /// `HTTP_DRAIN_GRACE` in the FFI (2 s) plus margin for everything the
    /// server does either side of the join it does budget.
    static let graceSeconds: UInt64 = 10
    /// `POST /api/shutdown`'s own timeout in `stop()`.
    static let shutdownRequestSeconds: UInt64 = 5

    /// The server's own figure.
    static var serverSeconds: UInt64 {
        resolve(ProcessInfo.processInfo.environment[environmentName])
    }

    /// What `mold_engine_join` is given.
    static var joinSeconds: UInt64 { serverSeconds + graceSeconds }
    static var joinMilliseconds: UInt64 { joinSeconds * 1_000 }

    /// Everything quitting can wait for, and so the only figure shown to
    /// anyone. The panel used to say 45 while the wait could reach 50.
    static var totalSeconds: UInt64 { shutdownRequestSeconds + joinSeconds }

    static func resolve(_ raw: String?) -> UInt64 {
        guard let parsed = UInt64(raw?.trimmingCharacters(in: .whitespaces) ?? ""), parsed >= 1
        else { return defaultSeconds }
        return parsed
    }
}
