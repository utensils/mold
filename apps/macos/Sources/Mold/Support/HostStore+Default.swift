import Foundation
import MoldClient

/// The machine work goes to when nothing else says.
///
/// A different question from `selectedMachine` (`Sidebar.swift`,
/// `ModelsPane.swift`, `MachinesPane.swift`), which is the machine you are
/// LOOKING at: collapsing the two would mean clicking a row in the sidebar
/// silently changed where the next render is submitted.
@MainActor
extension HostStore {
    /// Read and written straight from the suite, the way `HostPersistence`
    /// and `Destination.launch` already do. A stored property cannot live in
    /// an extension, and `HostStore`'s own stored properties (`HostStore.swift`)
    /// are reserved for state this file does not own.
    private static let defaultMachineKey = "defaultMachine"

    var defaultMachine: MoldHost.ID? {
        get { AppStorageSuite.defaults.string(forKey: Self.defaultMachineKey).flatMap(UUID.init) }
        set {
            guard let newValue else {
                AppStorageSuite.defaults.removeObject(forKey: Self.defaultMachineKey)
                return
            }
            AppStorageSuite.defaults.set(newValue.uuidString, forKey: Self.defaultMachineKey)
        }
    }

    func setDefault(_ host: MoldHost) {
        defaultMachine = host.id
    }
}
