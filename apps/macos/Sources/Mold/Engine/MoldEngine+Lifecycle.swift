import Foundation
import MoldClient

extension MoldEngine {
    /// A dead or stopped engine must leave the machine list, or "This Mac"
    /// keeps pointing at a closed port. `MoldEngine` cannot reach `HostStore`,
    /// so the composition root hands it this.
    func dropsItsMachine(from hosts: HostStore) {
        onEngineGone = { [weak hosts] in hosts?.dropLocalEngine() }
    }

    /// The other half: the engine answering puts "This Mac" in the list,
    /// whoever started it.
    func adoptsItsMachine(into hosts: HostStore) {
        onEngineReady = { [weak hosts] host in hosts?.adoptLocalEngine(host) }
    }

    /// Resolves the launch and runs the one-shot preamble, at most once.
    /// `nil` means the refusal is already on `state`.
    @discardableResult
    func prepared() -> EngineLaunch? {
        #if MOLD_EMBEDDED_ENGINE
        if let launch { return launch }
        // Startup is the sole owner of credential IO and the one-shot preamble.
        // Never retry either on the main actor after stores have begun networking.
        transition(to: .failed(Failure(
            reason: "The engine hasn't been prepared. Quit and relaunch Mold to try again.",
            relaunchNeeded: true)))
        return nil
        #else
        return nil
        #endif
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

    #if MOLD_EMBEDDED_ENGINE
    func bring(up launch: EngineLaunch) async {
        // Nonblocking flock does not bound external-volume path resolution.
        let writer = await Task.detached { EngineInterlock.homeWriter() }.value
        advisory = EngineInterlock.advisory(for: writer)
        let port = await Task.detached { mold_engine_configured_port() }.value
        guard port != 0 else {
            transition(to: .failed(MoldEngine.Failure(
                reason: "This Mac's configured port is unavailable. Set a free port with "
                    + "mold config set server_port <port>, then try again.",
                relaunchNeeded: false)))
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
            guard mold_engine_is_alive() else {
                transition(to: .failed(MoldEngine.Failure(
                    reason: "The engine stopped before it could answer on the configured port. "
                        + "Mold's log in ~/Library/Logs/Mold has the detail.",
                    relaunchNeeded: true)))
                return
            }
            transition(to: .running(port: port))
            if let host { onEngineReady?(host) }
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
    func watch() {
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
