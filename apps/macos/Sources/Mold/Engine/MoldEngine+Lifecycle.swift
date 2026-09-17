import Foundation
import MoldClient

extension MoldEngine {
    /// Built and prepared before anything else in the app exists, because the
    /// preamble writes `MOLD_HOME`, `MOLD_API_KEY` and `MOLD_CORS_ORIGIN`
    /// with `setenv` (review 05-M4).
    static func bootstrapped() -> MoldEngine {
        let engine = MoldEngine()
        engine.bootstrapAtLaunch()
        return engine
    }

    /// A dead or stopped engine must leave the machine list, or "This Mac"
    /// keeps pointing at a closed port. `MoldEngine` cannot reach `HostStore`,
    /// so the composition root hands it this.
    func dropsItsMachine(from hosts: HostStore) {
        onEngineGone = { [weak hosts] in hosts?.dropLocalEngine() }
    }

    /// Whether Start can do anything. `.failed` counts only where nothing
    /// one-shot has been consumed yet.
    var canStart: Bool { MoldEngine.canStart(state) }

    static func canStart(_ state: State) -> Bool {
        switch state {
        case .stopped: true
        case let .failed(failure): !failure.relaunchNeeded
        default: false
        }
    }

    func start() {
        #if MOLD_EMBEDDED_ENGINE
        guard canStart else { return }
        transition(to: .starting)
        guard let launch = prepared() else { return }
        Task { await bring(up: launch) }
        #endif
    }

    /// Stops the engine by asking it to, which is the only way an embedder
    /// can: `run_server`'s shutdown trigger is reachable through `POST
    /// /api/shutdown` and nothing else.
    ///
    /// The join gets the server's OWN budget. It used to get 8 s, and the
    /// result was discarded — so quitting during a render cut the drain and
    /// left the gallery writer lease behind (review 05-M7).
    func stop() async {
        #if MOLD_EMBEDDED_ENGINE
        guard case let .running(port) = state else { return }
        watchdog?.cancel()
        transition(to: .stopping)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/shutdown")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 5
        if let key = launch?.apiKey { request.setValue(key, forHTTPHeaderField: "X-Api-Key") }
        _ = try? await URLSession.shared.data(for: request)
        // Off the main thread: this blocks for the whole budget, and the
        // shutdown request above already yielded.
        let budget = EngineShutdownBudget.milliseconds
        let joined = await Task.detached { mold_engine_join(budget) }.value
        // The engine bootstraps at most once per process, so `.stopped` would
        // be a lie: there is nothing left to start.
        transition(to: .unavailable(
            joined
                ? "The engine starts once per launch. Relaunch Mold to start it again."
                : "The engine was still finishing when Mold quit. Relaunch Mold to start it again."
        ))
        onEngineGone?()
        #endif
    }

    #if MOLD_EMBEDDED_ENGINE
    private func bring(up launch: EngineLaunch) async {
        if let refusal = await EngineInterlock.otherServer() {
            transition(to: .failed(MoldEngine.Failure(reason: refusal, relaunchNeeded: false)))
            return
        }
        let port = await Task.detached { mold_engine_alloc_port() }.value
        guard port != 0 else {
            transition(to: .failed(MoldEngine.Failure(
                reason: "No free port on this Mac.", relaunchNeeded: false)))
            return
        }
        let started = await Task.detached {
            "127.0.0.1".withCString { bind in mold_engine_start(bind, port, nil) }
        }.value
        guard started == 0 else {
            transition(to: .failed(MoldEngine.Failure(
                reason: "The engine couldn't start.", relaunchNeeded: false)))
            return
        }
        // `mold_engine_start` returns the instant the THREAD is spawned —
        // before the runtime exists, before the DB migration, before gallery
        // authority recovery, and before the TCP bind. Publishing `.running`
        // there had every store polling a closed port (review 05-M3).
        switch await EngineProbe.answer(port: port, apiKey: launch.apiKey) {
        case .answered:
            transition(to: .running(port: port))
            // After the engine is listening, so `run_server`'s own tokio
            // SIGTERM handler is already in place and this replaces it.
            EngineSignals.forwardTerminationToTheApp()
            watch()
        case let .refused(reason):
            transition(to: .failed(MoldEngine.Failure(reason: reason, relaunchNeeded: true)))
        }
    }

    /// Asks the engine thread whether it is still there. Nothing did, so a
    /// dead engine stayed `.running` forever.
    private func watch() {
        watchdog?.cancel()
        watchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                guard let self, case .running = state, !Task.isCancelled else { return }
                if await Task.detached(operation: { mold_engine_is_alive() }).value { continue }
                transition(to: .failed(MoldEngine.Failure(
                    reason: "The engine stopped on its own. Relaunch Mold to start it again — "
                        + "Mold's log in ~/Library/Logs/Mold has the detail.",
                    relaunchNeeded: true)))
                onEngineGone?()
                return
            }
        }
    }
    #endif
}
