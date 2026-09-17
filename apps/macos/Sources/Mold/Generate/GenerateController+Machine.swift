import Foundation
import MoldClient

/// Where the next render goes.
///
/// `nil` is Auto -- `HostStore.preferredHost`, the default machine or else
/// the first one that's up (M8 decision 2). Persisted in the suite the way
/// `HostStore.defaultMachine` is (`HostStore+Default.swift`), under
/// `generateMachine` -- the desktop's `generateTargetHost`, same meaning.
///
/// A computed property reading straight from the suite rather than a stored
/// one: `GenerateController.swift` is already at the line cap, and a stored
/// property cannot live in an extension in another file anyway.
@MainActor
extension GenerateController {
    private static let machineChoiceKey = "generateMachine"

    var machineChoice: MoldHost.ID? {
        get { AppStorageSuite.defaults.string(forKey: Self.machineChoiceKey).flatMap(UUID.init) }
        set {
            guard let newValue else {
                AppStorageSuite.defaults.removeObject(forKey: Self.machineChoiceKey)
                return
            }
            AppStorageSuite.defaults.set(newValue.uuidString, forKey: Self.machineChoiceKey)
        }
    }
}
