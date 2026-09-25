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
        let resolved: EngineLaunch
        do {
            resolved = try EngineLaunchPlan.resolve(
                home: MoldHome.resolve(), secrets: .shared, logDirectory: Self.logDirectory)
        } catch {
            // Nothing one-shot has been consumed, so Start can be pressed
            // again once the drive is back or the store is writable.
            transition(to: .failed(Failure(
                reason: (error as? EngineLaunchRefusal)?.reason ?? "The engine couldn't start.",
                relaunchNeeded: false)))
            return nil
        }
        let code = resolved.home.withCString { home in
            resolved.apiKey.withCString { key in
                resolved.logDirectory.withCString { logs in
                    mold_engine_bootstrap(home, key, logs)
                }
            }
        }
        guard code == 0 else {
            transition(to: .failed(Failure(
                reason: "The engine couldn't prepare itself. Relaunch Mold to try again — "
                    + "Mold's log in ~/Library/Logs/Mold has the detail.",
                relaunchNeeded: true)))
            return nil
        }
        record(resolved)
        if case .failed = state { transition(to: .stopped) }
        return resolved
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
        // On the actor: one open(2) and one non-blocking flock on a local
        // path, which is microseconds and cannot wait on a lock.
        advisory = EngineInterlock.advisory(for: EngineInterlock.homeWriter())
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
