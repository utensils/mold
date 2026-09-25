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
        /// Draining. The app has asked the engine to stop and is waiting out
        /// the server's own budget. The sentence is carried because a drain
        /// that OVERRUNS that budget is a different thing to say: the engine
        /// thread is still writing, and this process can never start another
        /// (review F1b).
        case stopping(String)
        case failed(Failure)
    }

    /// A refusal, and whether pressing Start again could possibly help.
    struct Failure: Equatable, Sendable {
        let reason: String
        /// The engine bootstraps ONCE per process — the models-dir override
        /// is a `OnceLock`, tracing installs a global subscriber, and
        /// `mold_engine_start` keeps a finished thread's handle in its slot —
        /// so once it has run, nothing short of a relaunch starts another.
        /// Start used to be offered for every failure and was simply inert
        /// (review 05-M2); now it is offered only where it can work, and a
        /// relaunch is offered by name where it cannot.
        let relaunchNeeded: Bool
    }

    private(set) var state: State = MoldEngine.isLinked ? .stopped : .unavailable(
        "This build has no local engine. Run `make engine` and rebuild."
    )

    /// The ONE writer, so a view can never move the engine and a `grep` for
    /// `transition(to:)` finds every place it does move. `MoldEngine+Lifecycle`
    /// is the only caller.
    func transition(to next: State) { state = next }

    /// What this launch resolved, kept because the API key the engine was
    /// STARTED with is the one "This Mac" has to present back to it.
    private(set) var launch: EngineLaunch?

    /// Written once, by `prepared()`, for the same reason `transition(to:)`
    /// exists: one writer, and a greppable one.
    func record(_ resolved: EngineLaunch) { launch = resolved }

    /// Polls the engine's liveness while it is running, so a panic or a
    /// `run_server` that returned `Err` becomes `.failed` rather than a
    /// machine list entry pointing at a closed port (review 05-M1).
    var watchdog: Task<Void, Never>?

    /// Called when a running engine stops or dies, so the machine list can
    /// drop "This Mac". Set by the composition root, which owns both.
    var onEngineGone: (() -> Void)?

    /// Called once the engine ANSWERS, with the machine-list entry for it.
    /// Whoever started it -- the launch or Settings' Start button -- gets
    /// "This Mac" in the list without polling for it.
    var onEngineReady: ((MoldHost) -> Void)?

    /// What is true but not a refusal — today, only that something else is
    /// already publishing into this home (`EngineInterlock`).
    var advisory: String?

    /// True when the staticlib was linked in.
    static var isLinked: Bool {
        #if MOLD_EMBEDDED_ENGINE
        true
        #else
        false
        #endif
    }

    var host: MoldHost? {
        guard case let .running(port) = state, let launch else { return nil }
        return Self.localHost(port: port, apiKey: launch.apiKey)
    }

    /// The machine-list entry for an engine listening on `port`.
    ///
    /// It carries the key, which is the whole of review 05-H1 on this side:
    /// the engine is started with `MOLD_API_KEY` set, so every route it serves
    /// is authenticated and the app is the only caller that holds the answer.
    static func localHost(port: UInt16, apiKey: String) -> MoldHost? {
        guard let url = URL(string: "http://127.0.0.1:\(port)") else { return nil }
        return MoldHost(id: localHostID, name: "This Mac", baseURL: url, apiKey: apiKey)
    }

    /// A fixed id so the local engine keeps its identity across launches and
    /// its prints stay attributable to this Mac in the merged library.
    static let localHostID = UUID(uuidString: "00000000-0000-4000-A000-000000000001")!

    /// Whether a phone could ever be paired with this machine.
    ///
    /// False for this Mac's own engine, whatever it advertises: it binds
    /// 127.0.0.1, so a pairing issued there encodes a URL that resolves, on
    /// the phone, to the phone. It became reachable at all only because the
    /// engine is started WITH a key now, which makes `auth_required` true
    /// (review F4).
    static func isPairable(_ host: MoldHost) -> Bool { host.id != localHostID }

    /// Prepares this process for an engine, as early in the launch as the app
    /// has a main actor.
    ///
    /// `mold_engine_bootstrap` writes `MOLD_HOME`, `MOLD_API_KEY` and
    /// `MOLD_CORS_ORIGIN` with `setenv`, which reallocates `environ` and is
    /// not safe beside a concurrent `getenv` — CFNetwork reads proxy
    /// variables, among others. It used to run from a detached `Task` in a
    /// fully launched app, with URLSession's threads and `HostStore` already
    /// polling (review 05-M4). `@main`'s `init`, before a single store is
    /// built, is the earliest point this app owns. Honestly: AppKit and the
    /// Swift runtime have started threads of their own before `init` runs, so
    /// this NARROWS the window rather than closing it. Closing it means
    /// `run_server` taking the home and the key as parameters instead of
    /// through the environment — a change to the engine, not to the embedder.
    func bootstrapAtLaunch() { _ = prepared() }

    static var logDirectory: String {
        FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Logs/Mold").path
    }
}
