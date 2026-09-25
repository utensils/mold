import Foundation

/// Whether this launch starts the embedded engine by itself.
///
/// ON unless somebody turned it off. It used to be off with no way to turn it
/// on: every launch opened onto a machine list with no "This Mac" until
/// Settings ▸ This Mac ▸ Start Engine was found and pressed. The one-shot
/// bootstrap (`MoldEngine.Failure.relaunchNeeded`) is not a reason to wait --
/// a start that fails at launch is the same one shot as one that fails on a
/// click, and Settings ▸ This Mac says why either way.
///
/// Never inside a test run: the unit tests use the app as their host, and an
/// engine there would open and write this Mac's real `MOLD_HOME`.
enum EngineAutostart {
    static let startsAtLaunchKey = "engineStartsAtLaunch"

    /// Pure, so the rule pins without a launched app.
    static func shouldStart(isLinked: Bool, preference: Bool?, isRunningTests: Bool) -> Bool {
        isLinked && !isRunningTests && (preference ?? true)
    }

    static func atLaunch(
        defaults: UserDefaults = AppStorageSuite.defaults,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        shouldStart(
            isLinked: MoldEngine.isLinked,
            preference: defaults.object(forKey: startsAtLaunchKey) as? Bool,
            isRunningTests: environment["XCTestConfigurationFilePath"] != nil)
    }
}
