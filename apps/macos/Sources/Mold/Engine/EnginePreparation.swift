import Foundation
import MoldClient

/// Only the startup worker resolves private storage and mutates engine process state.
nonisolated enum EnginePreparation {
    static func resolve(onHomeResolved: @Sendable (URL) -> Void = { _ in }) throws -> EngineLaunch? {
        #if MOLD_EMBEDDED_ENGINE
        // Test hosts must never touch the user's private home.
        guard ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil else { return nil }
        let home = MoldHome.resolve()
        onHomeResolved(home.url)
        return try EngineLaunchPlan.resolve(
            home: home, secrets: .shared,
            logDirectory: FileManager.default.homeDirectoryForCurrentUser
                .appending(path: "Library/Logs/Mold").path)
        #else
        return nil
        #endif
    }

    static func bootstrap(_ launch: EngineLaunch?) throws -> EngineLaunch? {
        #if MOLD_EMBEDDED_ENGINE
        guard let launch else { return nil }
        let code = launch.home.withCString { home in
            launch.apiKey.withCString { key in
                launch.logDirectory.withCString { logs in mold_engine_bootstrap(home, key, logs) }
            }
        }
        guard code == 0 else {
            throw EngineLaunchRefusal.home(
                "The engine couldn't prepare itself. Quit and relaunch Mold to try again. "
                    + "Mold's log in ~/Library/Logs/Mold has the detail.")
        }
        #endif
        return launch
    }
}
