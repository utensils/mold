import Foundation

extension MoldEngine {
    /// Stops the engine by asking it to, which is the only way an embedder
    /// can: `run_server`'s shutdown trigger is reachable through `POST
    /// /api/shutdown` and nothing else.
    func stop() async {
        #if MOLD_EMBEDDED_ENGINE
        guard case let .running(port) = state else { return }
        watchdog?.cancel()
        transition(to: .stopping("Finishing this Mac's renders and closing the library."))
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/shutdown")!)
        request.httpMethod = "POST"
        request.timeoutInterval = TimeInterval(EngineShutdownBudget.shutdownRequestSeconds)
        if let key = launch?.apiKey { request.setValue(key, forHTTPHeaderField: "X-Api-Key") }
        _ = try? await URLSession.shared.data(for: request)
        // Off the main thread: this blocks for the whole budget, and the
        // shutdown request above already yielded.
        let budget = EngineShutdownBudget.joinMilliseconds
        let joined = await Task.detached { mold_engine_join(budget) }.value
        // The listener is gone either way, so the machine leaves the list.
        onEngineGone?()
        guard joined else {
            // It is STILL WRITING. Reporting "stopped" here was a lie that
            // left a zombie engine thread holding the port, the DB and the
            // GPU with nothing watching it (review F1b). `canStart` refuses
            // `.stopping`, so this process can never start a second one.
            transition(to: .stopping(
                "The engine is taking longer than \(EngineShutdownBudget.joinSeconds) seconds to "
                    + "finish a render. It is still writing; Mold will say when it is done."))
            watchUntilFinished()
            return
        }
        // The engine bootstraps at most once per process, so `.stopped` would
        // be a lie: there is nothing left to start.
        transition(to: .unavailable(
            "The engine starts once per launch. Relaunch Mold to start it again."))
        #endif
    }

    /// Everything quitting has to wait for, whatever the engine is doing.
    ///
    /// `applicationShouldTerminate` used to answer `.terminateNow` for every
    /// state but `.running`, so ⌘Q during startup hard-killed the engine
    /// inside `recover_storage` or the one-time v2→v3 upgrade, and ⌘Q after
    /// Stop Engine cut the remaining drain (review F2).
    func finishForQuit() async {
        #if MOLD_EMBEDDED_ENGINE
        // A startup cannot be asked to stop -- there is no listener yet -- so
        // it is waited out, bounded, and then stopped properly if it arrives.
        if case .starting = state {
            let deadline = ContinuousClock.now + .seconds(Int64(EngineShutdownBudget.joinSeconds))
            while case .starting = state, ContinuousClock.now < deadline {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        if case .running = state {
            await stop()
            return
        }
        // Mid-drain already, from Stop Engine: wait on the same join rather
        // than starting a second one.
        if case .stopping = state {
            let deadline = ContinuousClock.now + .seconds(Int64(EngineShutdownBudget.joinSeconds))
            while case .stopping = state, ContinuousClock.now < deadline {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        #endif
    }

    /// Whether quitting has to wait at all: an engine thread exists.
    var isDraining: Bool { MoldEngine.isDraining(state) }

    static func isDraining(_ state: State) -> Bool {
        switch state {
        case .running, .starting, .stopping: true
        case .unavailable, .stopped, .failed: false
        }
    }

    #if MOLD_EMBEDDED_ENGINE
    /// Keeps watching an overrunning drain, so "still finishing" becomes
    /// "finished" rather than staying on screen forever.
    private func watchUntilFinished() {
        watchdog?.cancel()
        watchdog = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                guard let self, case .stopping = state, !Task.isCancelled else { return }
                if await Task.detached(operation: { mold_engine_is_alive() }).value { continue }
                transition(to: .unavailable(
                    "The engine has finished. It starts once per launch, so relaunch Mold to "
                        + "start it again."))
                return
            }
        }
    }
    #endif
}
