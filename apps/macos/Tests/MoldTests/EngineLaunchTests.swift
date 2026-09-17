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

    /// An environment naming an explicit home, so nothing here reads this
    /// Mac's real pointer file or its real secrets.
    private func environment(_ url: URL, _ extra: [String: String] = [:]) -> [String: String] {
        extra.merging(["MOLD_HOME": url.path(percentEncoded: false)]) { current, _ in current }
    }

    @Test func theEngineIsStartedWithAKeyThatSurvivesTheLaunch() throws {
        let directory = scratch()
        let env = environment(directory)
        let first = try EngineLaunchPlan.resolve(
            home: MoldHome.resolve(environment: env), secrets: SecretStore(directory: directory),
            logDirectory: "/tmp", environment: env)
        #expect(!first.apiKey.isEmpty)

        // A second launch -- a second process, so a cold store -- must present
        // the same key, or "This Mac" holds one the engine does not accept.
        let second = try EngineLaunchPlan.resolve(
            home: MoldHome.resolve(environment: env), secrets: SecretStore(directory: directory),
            logDirectory: "/tmp", environment: env)
        #expect(second.apiKey == first.apiKey)
    }

    @Test func anOperatorsOwnKeyWins() throws {
        let directory = scratch()
        let env = environment(directory, ["MOLD_API_KEY": "operator-key"])
        let launch = try EngineLaunchPlan.resolve(
            home: MoldHome.resolve(environment: env), secrets: SecretStore(directory: directory),
            logDirectory: "/tmp", environment: env)
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
        let env = environment(scratch())

        #expect(throws: EngineLaunchRefusal.self) {
            try EngineLaunchPlan.resolve(
                home: MoldHome.resolve(environment: env), secrets: secrets,
                logDirectory: "/tmp", environment: env)
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

    /// The engine forces its answer into `MOLD_HOME`, so mold's own
    /// fail-closed guard can never fire -- this side has to.
    @Test func aDamagedHomePointerRefusesRatherThanBuildingANewLibrary() {
        let pointer = FileManager.default.temporaryDirectory
            .appending(path: "engine-damaged-pointer-\(UUID().uuidString)")
        try? "not/absolute".write(to: pointer, atomically: true, encoding: .utf8)
        let env = ["MOLD_HOME_POINTER_PATH": pointer.path(percentEncoded: false)]

        #expect(throws: EngineLaunchRefusal.self) {
            try EngineLaunchPlan.resolve(
                home: MoldHome.resolve(environment: env), secrets: SecretStore(directory: scratch()),
                logDirectory: "/tmp", environment: env)
        }
    }

    @Test func theLocalMachineCarriesTheEnginesKey() {
        let host = MoldEngine.localHost(port: 61_440, apiKey: "minted")
        #expect(host?.apiKey == "minted")
        #expect(host?.id == MoldEngine.localHostID)
        #expect(host?.baseURL.absoluteString == "http://127.0.0.1:61440")
    }
}
