import AppKit

/// Starting a fresh copy of Mold and standing this one down.
///
/// The engine bootstraps once per process, so for the failures that consumed
/// that one chance the only honest offer is a relaunch — which the app can do
/// for someone rather than telling them to quit and reopen (review 05-M2).
enum EngineRelaunch {
    static func now() {
        let configuration = NSWorkspace.OpenConfiguration()
        // Without this, Launch Services activates THIS instance instead.
        configuration.createsNewApplicationInstance = true
        NSWorkspace.shared.openApplication(at: Bundle.main.bundleURL, configuration: configuration) {
            _, _ in
            // Through `terminate`, not `exit`: quitting still has to reach
            // the engine's drain and the caches' purge.
            Task { @MainActor in NSApplication.shared.terminate(nil) }
        }
    }
}
