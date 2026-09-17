import Foundation
import MoldClient
import Testing

@testable import Mold

/// What the in-process engine is started with.
///
/// **Fails today**: `MoldEngine.start()` passed `nil` for the FFI's `api_key`,
/// so the engine ran with `AuthState = None` and every route open — and the
/// local `MoldHost` carried no key either (review 05-H1).
@MainActor
struct EngineLaunchTests {
    private func scratch() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "engine-launch-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func home(_ url: URL) -> MoldHome {
        MoldHome.resolve(environment: ["MOLD_HOME": url.path(percentEncoded: false)])
    }

    @Test func theEngineIsStartedWithAKeyThatSurvivesTheLaunch() throws {
        let directory = scratch()
        let secrets = SecretStore(directory: directory)
        let first = try EngineLaunchPlan.resolve(
            home: home(directory), secrets: secrets, logDirectory: "/tmp", environment: [:])
        #expect(!first.apiKey.isEmpty)

        // A second launch -- a second process, so a cold store -- must present
        // the same key, or "This Mac" holds one the engine does not accept.
        let second = try EngineLaunchPlan.resolve(
            home: home(directory), secrets: SecretStore(directory: directory),
            logDirectory: "/tmp", environment: [:])
        #expect(second.apiKey == first.apiKey)
    }

    @Test func anOperatorsOwnKeyWins() throws {
        let directory = scratch()
        let launch = try EngineLaunchPlan.resolve(
            home: home(directory), secrets: SecretStore(directory: directory),
            logDirectory: "/tmp", environment: ["MOLD_API_KEY": "operator-key"])
        #expect(launch.apiKey == "operator-key")
    }

    /// A store that cannot be written must stop the engine, not start it
    /// keyless: keyless is the hole.
    @Test func aKeyThatCannotBePersistedRefusesTheLaunch() {
        // A path whose parent is a FILE, so creating the directory fails.
        let blocker = FileManager.default.temporaryDirectory
            .appending(path: "engine-launch-blocker-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: blocker.path(percentEncoded: false), contents: Data())
        let secrets = SecretStore(directory: blocker.appending(path: "inside"))

        #expect(throws: EngineLaunchRefusal.self) {
            try EngineLaunchPlan.resolve(
                home: home(scratch()), secrets: secrets, logDirectory: "/tmp", environment: [:])
        }
    }

    @Test func aHomeThatIsNotThereRefusesBeforeAnyKeyIsMinted() {
        let missing = FileManager.default.temporaryDirectory
            .appending(path: "engine-home-\(UUID().uuidString)")
        let pointer = FileManager.default.temporaryDirectory
            .appending(path: "engine-pointer-\(UUID().uuidString)")
        try? missing.path(percentEncoded: false).write(
            to: pointer, atomically: true, encoding: .utf8)
        let home = MoldHome.resolve(
            environment: ["MOLD_HOME_POINTER_PATH": pointer.path(percentEncoded: false)])

        #expect(throws: EngineLaunchRefusal.self) {
            try EngineLaunchPlan.resolve(
                home: home, secrets: SecretStore(directory: scratch()),
                logDirectory: "/tmp", environment: [:])
        }
    }

    @Test func theLocalMachineCarriesTheEnginesKey() {
        let host = MoldEngine.localHost(port: 61_440, apiKey: "minted")
        #expect(host?.apiKey == "minted")
        #expect(host?.id == MoldEngine.localHostID)
        #expect(host?.baseURL.absoluteString == "http://127.0.0.1:61440")
    }
}
