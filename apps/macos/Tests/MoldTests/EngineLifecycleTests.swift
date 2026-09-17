import Foundation
import MoldClient
import Testing

@testable import Mold

/// Starting, watching and stopping the in-process engine.
@MainActor
struct EngineLifecycleTests {
    /// **Fails today**: `start()` guarded on `.stopped` alone while
    /// `LocalEngineSettings` offered Start for `.failed` too, so every
    /// recoverable failure left an enabled button that did nothing.
    @Test func startIsOfferedOnlyWhereItCanWork() {
        #expect(MoldEngine.canStart(.stopped))
        // A refusal that consumed nothing one-shot: reconnect the drive and
        // press Start.
        #expect(MoldEngine.canStart(
            .failed(.init(reason: "drive is away", relaunchNeeded: false))))
        // A refusal that did: the process has had its one bootstrap.
        #expect(!MoldEngine.canStart(
            .failed(.init(reason: "the engine stopped", relaunchNeeded: true))))
        #expect(!MoldEngine.canStart(.running(port: 61_440)))
        #expect(!MoldEngine.canStart(.unavailable("no engine in this build")))
        // Including a drain that OVERRAN its budget: the engine thread is
        // still writing, and a second one in this process is not a thing that
        // can exist (review F1b).
        #expect(!MoldEngine.canStart(.stopping("still finishing a render")))
    }

    /// **Fails today**: `applicationShouldTerminate` answered `.terminateNow`
    /// for every state but `.running`, so ⌘Q during startup hard-killed the
    /// engine inside `recover_storage` or the one-time v2→v3 authority
    /// upgrade, and ⌘Q after Stop Engine cut the rest of the drain.
    @Test func quittingWaitsForEveryStateWithAnEngineThreadInIt() {
        let engine = MoldEngine()
        #expect(MoldEngine.isDraining(.running(port: 61_440)))
        #expect(MoldEngine.isDraining(.starting))
        #expect(MoldEngine.isDraining(.stopping("finishing")))
        #expect(!MoldEngine.isDraining(.stopped))
        #expect(!MoldEngine.isDraining(.failed(.init(reason: "x", relaunchNeeded: true))))
        #expect(!MoldEngine.isDraining(.unavailable("no engine in this build")))
        // An engine nobody started has no thread in EITHER build -- `.stopped`
        // when one is linked, `.unavailable` when none is -- so quitting never
        // waits on it. (This read `== isLinked` when it was written in a
        // build with no engine, where that is `false == false`.)
        #expect(!engine.isDraining)
    }

    /// **Fails today**: the probe counted 480 ATTEMPTS, each costing its own
    /// 2 s URL timeout plus the sleep, so an engine that bound and then
    /// stalled pinned `.starting` for ~18 minutes rather than the 2 it
    /// advertised.
    @Test func theProbeGivesUpOnTheClockRatherThanAfterNAttempts() async {
        var clock = ContinuousClock.now
        var asked = 0
        let answer = await EngineProbe.answer(
            port: 61_440, apiKey: "k", budget: .seconds(10), interval: .milliseconds(1),
            now: {
                // Every question costs its whole URL timeout, which is what
                // an attempt count cannot see.
                defer { clock = clock.advanced(by: .seconds(3)) }
                return clock
            },
            ask: { _, _ in
                asked += 1
                return nil
            })
        guard case .refused = answer else {
            Issue.record("a silent port must not read as a running engine")
            return
        }
        // 10 s of budget at 3 s a question: a handful, not 480.
        #expect(asked <= 5)
    }

    /// **Fails today**: `.running` was published the instant the thread was
    /// spawned, so every store began polling a port nothing had bound.
    @Test func runningWaitsForTheEngineToAnswer() async {
        var asked = 0
        let answer = await EngineProbe.answer(
            port: 61_440, apiKey: "k", budget: .seconds(4), interval: .milliseconds(1),
            ask: { _, _ in
                asked += 1
                return asked < 3 ? nil : 200
            })
        #expect(answer == .answered)
        #expect(asked == 3)
    }

    @Test func aPortHeldBySomethingElseIsNamedRatherThanWaitedOut() async {
        let answer = await EngineProbe.answer(
            port: 61_440, apiKey: "k", budget: .seconds(4), interval: .milliseconds(1),
            ask: { _, _ in 401 })
        guard case let .refused(reason) = answer else {
            Issue.record("a 401 must refuse, not retry")
            return
        }
        #expect(reason.contains("61440"))
    }

    @Test func anEngineThatNeverAnswersGivesUpWithinItsBudget() async {
        let answer = await EngineProbe.answer(
            port: 61_440, apiKey: "k", budget: .milliseconds(6), interval: .milliseconds(1),
            ask: { _, _ in nil })
        guard case .refused = answer else {
            Issue.record("a silent port must not read as a running engine")
            return
        }
    }

    /// **Fails today**: the check asked `127.0.0.1:7680`, which this app never
    /// binds and Mold Desktop only PREFERS — so it detected nothing, and the
    /// test injected the probe and could not see that. The DETECTION now lives
    /// in the FFI against mold's own gallery writer lease and is tested there
    /// (`a_held_lease_names_the_writer_and_is_never_taken_from_it`); what is
    /// left here is the reading of its answer.
    ///
    /// Note the deliberate change of policy from a refusal to an advisory —
    /// `EngineInterlock`'s own doc cites the three places in mold's code that
    /// decide it.
    @Test func anotherMoldPublishingIntoThisHomeIsSaidOutLoud() {
        #expect(EngineInterlock.homeWriter(probe: { 4242 }) == .live(pid: 4242))
        #expect(EngineInterlock.advisory(for: .live(pid: 4242))?.contains("4242") == true)
        // Stale, absent, and "could not tell" are all silence: absence of an
        // answer is not evidence.
        #expect(EngineInterlock.homeWriter(probe: { 0 }) == .none)
        #expect(EngineInterlock.homeWriter(probe: { -1 }) == .unknown)
        #expect(EngineInterlock.advisory(for: .none) == nil)
        #expect(EngineInterlock.advisory(for: .unknown) == nil)
        // A live writer whose body could not be read is still said, unnamed.
        #expect(EngineInterlock.homeWriter(probe: { -2 }) == .live(pid: nil))
        #expect(EngineInterlock.advisory(for: .live(pid: nil)) != nil)
    }

    /// **Fails today**: the engine is started with a key now, so
    /// `auth_required` is true for "This Mac" and the Machines pane drew a
    /// live "Pair a Phone…" whose QR encodes `http://127.0.0.1:<ephemeral>` —
    /// a credential for the phone's own loopback, persisted server-side.
    @Test func thisMacsEngineIsNotSomethingAPhonePairsWith() {
        let url = URL(string: "http://127.0.0.1:61440")!
        let local = MoldEngine.localHost(port: 61_440, apiKey: "minted")
        #expect(local.map(MoldEngine.isPairable) == false)
        // Every other machine is unaffected, keyed or not.
        #expect(MoldEngine.isPairable(MoldHost(name: "plato", baseURL: url, apiKey: "k")))
        #expect(MoldEngine.isPairable(MoldHost(name: "hal9000", baseURL: url)))
    }

    /// The drain budget is the SERVER's, read from the server's own source so
    /// the two cannot drift.
    @Test func theQuitBudgetIsTheServersOwn() throws {
        let lib = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .deletingLastPathComponent() // apps
            .deletingLastPathComponent() // <repo>
            .appending(path: "crates/mold-server/src/lib.rs")
        let rust = try String(contentsOf: lib, encoding: .utf8)
        let declared = rust
            .split(separator: "\n")
            .first { $0.contains("pub const DEFAULT_SHUTDOWN_ABORT_SECS") }
            .flatMap { $0.split(separator: "=").last }
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: " ;")) }
        #expect(declared == String(EngineShutdownBudget.defaultSeconds))
        #expect(rust.contains("\"\(EngineShutdownBudget.environmentName)\""))

        // The join waits LONGER than the server's own figure, because that
        // figure covers the GPU-owner join only -- the HTTP drain is ahead of
        // it and `drop(runtime)` behind it. Waiting exactly 45 s is what made
        // the first version expire while the engine was still writing.
        #expect(EngineShutdownBudget.joinSeconds > EngineShutdownBudget.serverSeconds)
        // And the figure shown to anyone covers the whole wait, shutdown POST
        // included: the panel used to say 45 while the wait could reach 50.
        #expect(EngineShutdownBudget.totalSeconds
            == EngineShutdownBudget.joinSeconds + EngineShutdownBudget.shutdownRequestSeconds)

        // The server's own rule for the override: at least one second,
        // anything unparseable ignored.
        #expect(EngineShutdownBudget.resolve("90") == 90)
        #expect(EngineShutdownBudget.resolve("0") == EngineShutdownBudget.defaultSeconds)
        #expect(EngineShutdownBudget.resolve("soon") == EngineShutdownBudget.defaultSeconds)
        #expect(EngineShutdownBudget.resolve(nil) == EngineShutdownBudget.defaultSeconds)
    }
}
