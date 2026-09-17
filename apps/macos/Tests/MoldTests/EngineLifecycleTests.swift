import Foundation
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
        #expect(!MoldEngine.canStart(.stopping))
        #expect(!MoldEngine.canStart(.unavailable("no engine in this build")))
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

    /// **Fails today**: nothing checked, so starting this engine beside Mold
    /// Desktop put two `run_server` on one home and stranded the other app's
    /// queued work.
    @Test func aSecondEngineOnOneHomeIsRefusedByName() async {
        let refusal = await EngineInterlock.otherServer(probe: { _ in true })
        #expect(refusal?.contains("127.0.0.1:7680") == true)
        #expect(await EngineInterlock.otherServer(probe: { _ in false }) == nil)
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

        // The server's own rule for the override: at least one second,
        // anything unparseable ignored.
        #expect(EngineShutdownBudget.resolve("90") == 90)
        #expect(EngineShutdownBudget.resolve("0") == EngineShutdownBudget.defaultSeconds)
        #expect(EngineShutdownBudget.resolve("soon") == EngineShutdownBudget.defaultSeconds)
        #expect(EngineShutdownBudget.resolve(nil) == EngineShutdownBudget.defaultSeconds)
    }
}
