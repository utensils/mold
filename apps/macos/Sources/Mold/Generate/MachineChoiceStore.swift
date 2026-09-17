import Foundation
import MoldClient

/// Where the next render goes, remembered across launches.
///
/// `nil` is Auto -- `HostStore.preferredHost`, the default machine or else the
/// first one that's up (M8 decision 2). Persisted in the suite the way
/// `HostStore.defaultMachine` is (`HostStore+Default.swift`), under
/// `generateMachine` -- the desktop's `generateTargetHost`, same meaning.
///
/// A type of its own rather than a computed property reading the suite
/// directly: reading `UserDefaults` in a getter registers no `@Observable`
/// mutation when it is written, so the Machine control refreshed only because
/// `choose()` usually ALSO wrote `modelName`. Picking Auto on a fleet whose
/// `preferredHost` is nil returns before any other write, and the label kept
/// naming the old machine (finding 02#11). `GenerateController.machineChoice`
/// is a stored property now, and this is where it lands.
enum MachineChoiceStore {
    private static let key = "generateMachine"

    static func load(from defaults: UserDefaults = AppStorageSuite.defaults) -> MoldHost.ID? {
        defaults.string(forKey: key).flatMap(UUID.init)
    }

    static func save(_ choice: MoldHost.ID?, to defaults: UserDefaults = AppStorageSuite.defaults) {
        guard let choice else {
            defaults.removeObject(forKey: key)
            return
        }
        defaults.set(choice.uuidString, forKey: key)
    }
}
