import Foundation
import MoldClient
import Testing

@testable import Mold

/// The engine starting by itself at launch.
@MainActor
struct EngineAutostartTests {
    /// ON for everyone who has not turned it off -- a new install and an
    /// existing one alike, because an absent key means "never chose".
    @Test func anAbsentPreferenceStartsTheEngine() {
        #expect(EngineAutostart.shouldStart(isLinked: true, preference: nil, isRunningTests: false))
    }

    @Test func turningItOffIsRespected() {
        #expect(!EngineAutostart.shouldStart(isLinked: true, preference: false, isRunningTests: false))
    }

    /// The tests run inside the app; an engine there would open this Mac's
    /// real home.
    @Test func neverInsideATestRun() {
        #expect(!EngineAutostart.shouldStart(isLinked: true, preference: true, isRunningTests: true))
    }

    @Test func aRemoteOnlyBuildHasNothingToStart() {
        #expect(!EngineAutostart.shouldStart(isLinked: false, preference: true, isRunningTests: false))
    }

    @Test func itReadsTheAppsOwnSuite() throws {
        let defaults = try #require(UserDefaults(suiteName: "EngineAutostartTests"))
        defer { defaults.removePersistentDomain(forName: "EngineAutostartTests") }
        defaults.set(false, forKey: EngineAutostart.startsAtLaunchKey)
        #expect(!EngineAutostart.atLaunch(defaults: defaults, environment: [:]))
        #expect(!EngineAutostart.atLaunch(
            defaults: UserDefaults(suiteName: "EngineAutostartTests.empty")!,
            environment: ["XCTestConfigurationFilePath": "/x"]))
    }

    /// Whoever started it, the engine answering puts "This Mac" in the list
    /// -- the launch has no Settings pane polling for it.
    @Test func anAnsweringEngineAdoptsThisMac() throws {
        let hosts = HostStore(hosts: []) { FakeBackend(host: $0) }
        let engine = MoldEngine()
        engine.adoptsItsMachine(into: hosts)
        let local = try #require(MoldEngine.localHost(port: 7680, apiKey: "k"))

        engine.onEngineReady?(local)

        #expect(hosts.hosts.map(\.id) == [MoldEngine.localHostID])
    }
}
