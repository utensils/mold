import Foundation

/// Whether this build may replace itself.
///
/// A dev build must NEVER offer to. `make build`, `make uat` and the test
/// scheme all produce a Debug bundle sitting in `apps/macos/build/`, usually
/// newer than anything published; an updater there would offer to overwrite
/// the thing being worked on with last night's release, and under
/// `make uat` -- whose whole point is a throwaway first launch -- it would
/// reach the network and write to the real preferences domain that the UAT
/// suite exists to avoid.
///
/// Three independent conditions, because each can be true without the others:
/// a Debug build; a `MOLD_NATIVE_FRESH` UAT run; and a unit-test host, which
/// is a real launched `.app` and would otherwise schedule a check in the
/// middle of a suite. Only a plain Release launch is left, which is the only
/// build a user ever runs.
///
/// A pure function of three booleans rather than a wall of `#if`, so
/// `UpdaterActivationTests` can exercise all eight combinations in whichever
/// configuration it happens to be compiled in -- a test wrapped in `#if DEBUG`
/// runs nothing in the other one.
nonisolated enum UpdaterActivation {
    static func isEnabled(isDebugBuild: Bool, isFreshUAT: Bool, isRunningTests: Bool) -> Bool {
        !isDebugBuild && !isFreshUAT && !isRunningTests
    }

    /// The same question against this process. `NativeUAT.fresh` is the one
    /// reader of that hook and already answers `nil` in Release, so in a
    /// shipped build only `isDebugBuild` and `isRunningTests` can differ --
    /// which is exactly the point of asking all three here rather than
    /// trusting one.
    @MainActor
    static func isEnabled(
        in environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        var debugBuild = false
        #if DEBUG
        debugBuild = true
        #endif
        return isEnabled(
            isDebugBuild: debugBuild,
            isFreshUAT: NativeUAT.fresh.isSet(in: environment),
            // XCTest and swift-testing both run under a host that carries
            // this; it is how a bundle knows it was launched to be tested.
            isRunningTests: environment["XCTestConfigurationFilePath"] != nil
                || environment["XCTestSessionIdentifier"] != nil
        )
    }
}
