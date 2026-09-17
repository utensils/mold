import Foundation
import MoldClient

/// mold's own Rust engine, running inside this process.
///
/// The app talks to it over HTTP on loopback — the SAME wire contract it uses
/// for a machine across the network. That is the whole point of the
/// arrangement: `EmbeddedBackend` is `HTTPBackend` pointed at 127.0.0.1, so
/// every screen already works against it, and nothing about generation has to
/// cross the FFI boundary.
@MainActor
@Observable
final class MoldEngine {
    enum State: Equatable {
        case unavailable(String)
        case stopped
        case starting
        case running(port: UInt16)
        case failed(String)
    }

    private(set) var state: State = MoldEngine.isLinked ? .stopped : .unavailable(
        "This build has no local engine. Run `make engine` and rebuild."
    )

    /// True when the staticlib was linked in.
    static var isLinked: Bool {
        #if MOLD_EMBEDDED_ENGINE
        true
        #else
        false
        #endif
    }

    var host: MoldHost? {
        guard case let .running(port) = state,
              let url = URL(string: "http://127.0.0.1:\(port)")
        else { return nil }
        return MoldHost(id: Self.localHostID, name: "This Mac", baseURL: url)
    }

    /// A fixed id so the local engine keeps its identity across launches and
    /// its prints stay attributable to this Mac in the merged library.
    static let localHostID = UUID(uuidString: "00000000-0000-4000-A000-000000000001")!

    func start() {
        #if MOLD_EMBEDDED_ENGINE
        guard case .stopped = state else { return }
        state = .starting

        // Resolved the way every other mold on this Mac resolves it. An app
        // launched from Finder is handed no environment at all, so reading
        // MOLD_HOME alone meant the engine ran against `~/.mold` while the CLI
        // and the Tauri app used the home someone had actually chosen.
        let resolved = MoldHome.resolve()
        if let reason = resolved.unavailableReason {
            state = .failed(reason)
            return
        }
        let home = resolved.url.path(percentEncoded: false)
        let logs = MoldEngine.logDirectory
        Task.detached(priority: .userInitiated) {
            // The engine starts at most ONCE per process: the models-dir
            // override is a process-lifetime OnceLock and tracing installs a
            // global subscriber, so changing either means relaunching.
            let bootstrapped = home.withCString { homePtr in
                logs.withCString { logPtr in
                    mold_engine_bootstrap(homePtr, nil, logPtr)
                }
            }
            guard bootstrapped == 0 else {
                await MainActor.run { self.state = .failed("The engine couldn't start.") }
                return
            }
            let port = mold_engine_alloc_port()
            guard port != 0 else {
                await MainActor.run { self.state = .failed("No free port on this Mac.") }
                return
            }
            let started = "127.0.0.1".withCString { bind in
                mold_engine_start(bind, port, nil)
            }
            await MainActor.run {
                self.state = started == 0
                    ? .running(port: port)
                    : .failed("The engine couldn't start.")
            }
        }
        #endif
    }

    /// Stops the engine by asking it to, which is the only way an embedder can:
    /// `run_server`'s shutdown trigger is reachable through `POST
    /// /api/shutdown` and nothing else.
    func stop() async {
        #if MOLD_EMBEDDED_ENGINE
        guard case let .running(port) = state else { return }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/shutdown")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 5
        _ = try? await URLSession.shared.data(for: request)
        // Off the main thread: this blocks for up to 8s, and the shutdown
        // request above already yielded, so nothing here needs to run on
        // the actor.
        _ = await Task.detached { mold_engine_join(8_000) }.value
        // The engine bootstraps at most once per process (`OnceLock`, a
        // global tracing subscriber) -- `.stopped` is what `start()` accepts,
        // and accepting it again here would be a second bootstrap this
        // process can't actually do.
        state = .unavailable("The engine starts once per launch. Relaunch Mold to start it again.")
        #endif
    }

    private static var logDirectory: String {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Logs/Mold").path
    }
}
