import AppKit
import Dispatch

/// SIGTERM, once the engine is running.
///
/// `run_server` installs a tokio unix SIGTERM handler
/// (`crates/mold-server/src/lib.rs:1247-1261`), which replaces `SIG_DFL`
/// PROCESS-wide. Once the engine had started, `kill` on Mold no longer
/// terminated it: the engine began a graceful shutdown of ITSELF while AppKit
/// carried on, and since `allow_hard_shutdown_exit()` is deliberately not
/// called, nothing ended the process. `pkill`, a script or a crash reporter's
/// terminate left a zombie GUI app with a half-shut-down engine
/// (review 05-M8).
///
/// This puts the app back in charge: SIGTERM becomes `NSApp.terminate`, which
/// runs `applicationShouldTerminate` and so reaches the engine through the
/// same drain quitting from the menu does.
enum EngineSignals {
    /// The source has to outlive this call or it is cancelled on dealloc.
    private nonisolated(unsafe) static var source: DispatchSourceSignal?

    /// Installed AFTER the engine answers, so it replaces tokio's handler
    /// rather than being replaced by it. The handler is registered from a
    /// spawned task inside `run_server`, which the runtime has certainly
    /// polled by the time `/api/status` answers; that ordering is the whole
    /// guarantee, and it is an ordering rather than a lock.
    static func forwardTerminationToTheApp() {
        guard source == nil else { return }
        // `DispatchSourceSignal` observes delivery through kqueue and does
        // NOT change the disposition, so without this the default (or
        // tokio's) action still runs.
        signal(SIGTERM, SIG_IGN)
        let installed = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        installed.setEventHandler {
            MainActor.assumeIsolated { NSApplication.shared.terminate(nil) }
        }
        installed.resume()
        source = installed
    }
}
