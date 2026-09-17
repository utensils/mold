import Foundation

/// The `MOLD_NATIVE_*` hooks, and the ONE place they are read.
///
/// All eight were compiled into Release with no `#if DEBUG` anywhere, so a
/// shipped app would seed machines, open sheets, preload a picture, swap its
/// whole preferences domain and draw canned fixtures for anyone able to set an
/// environment variable on it -- a launchd plist, a shell wrapper, anything
/// that spawns the bundle. None of that is a feature of a released build.
///
/// `make uat` builds Debug (`Makefile`: `CONFIG ?= Debug`, `uat: build`) and
/// so does the test scheme, so every UAT capture and every test keeps working
/// exactly as before.
///
/// Reading them through one type rather than at each site is what makes that
/// checkable: `NativeUATTests` fails if a `MOLD_NATIVE_` literal appears
/// anywhere else in `Sources/Mold` outside a comment.
enum NativeUAT: String, CaseIterable {
    /// The throwaway preferences domain and secrets directory
    /// (`AppStorageSuite`; `SecretStore.applicationSupport` reads it
    /// package-side under its own `#if DEBUG`, because MoldClient cannot
    /// import the app).
    case fresh = "MOLD_NATIVE_FRESH"
    /// Extra machines as `name=url` pairs (`HostStore.seededHosts`).
    case hosts = "MOLD_NATIVE_HOSTS"
    /// Where the window opens, and which sheet is already up
    /// (`Destination.launch`, `RootView`, `MachinesSettings`).
    case destination = "MOLD_NATIVE_DESTINATION"
    /// Which Settings tab (`SettingsUAT`).
    case settingsTab = "MOLD_NATIVE_SETTINGS_TAB"
    /// A preloaded source picture (`GeneratePane+UAT`).
    case sourceImage = "MOLD_NATIVE_SOURCE_IMAGE"
    /// The Library picker sheet, opened at launch (`GeneratePane+UAT`).
    case libraryPicker = "MOLD_NATIVE_LIBRARY_PICKER"
    /// Canned fixtures for two panes (`QueuePane+UAT`, `PairingSection+UAT`).
    case queueFixture = "MOLD_NATIVE_QUEUE_FIXTURE"
    case pairingFixture = "MOLD_NATIVE_PAIRING_FIXTURE"

    /// This hook's value, or `nil`.
    ///
    /// In a Release build the read is compiled out and this is ALWAYS `nil`,
    /// whatever the process was launched with.
    func value(
        in environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        #if DEBUG
        return environment[rawValue]
        #else
        return nil
        #endif
    }

    /// Whether the hook was given at all -- several of them mean something by
    /// their presence rather than by their value.
    func isSet(in environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        value(in: environment) != nil
    }
}
