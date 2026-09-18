import Foundation
import MoldClient

/// The machine work goes to when nothing else says.
///
/// A different question from `selectedMachine` (`Sidebar.swift`,
/// `ModelsPane.swift`, `MachinesPane.swift`), which is the machine you are
/// LOOKING at: collapsing the two would mean clicking a row in the sidebar
/// silently changed where the next render is submitted.
///
/// `defaultMachine` itself is a stored, observed property on `HostStore` (a
/// stored property cannot live in an extension). It used to be a computed
/// accessor over the suite, and Set as Default then redrew nothing until the
/// pane was left and re-entered (UAT 2026-09-17 #4). This file owns the
/// persistence: the same suite `HostPersistence` and `Destination.launch`
/// use, under the key `PreferencesReset` clears.
@MainActor
extension HostStore {
    private static let defaultMachineKey = "defaultMachine"

    static func storedDefaultMachine() -> MoldHost.ID? {
        AppStorageSuite.defaults.string(forKey: defaultMachineKey).flatMap(UUID.init)
    }

    func persistDefaultMachine() {
        guard let defaultMachine else {
            AppStorageSuite.defaults.removeObject(forKey: Self.defaultMachineKey)
            return
        }
        AppStorageSuite.defaults.set(defaultMachine.uuidString, forKey: Self.defaultMachineKey)
    }

    /// A preferences reset clears the key straight from the suite, the way
    /// every `@AppStorage` view expects; follow it so the store cannot keep a
    /// default the person just reset. The token is never removed: the block
    /// holds `self` weakly, and a store lives as long as the app.
    func followDefaultMachineInSuite(center: NotificationCenter = .default) {
        guard defaultsObserver == nil else { return }
        defaultsObserver = center.addObserver(
            forName: UserDefaults.didChangeNotification, object: AppStorageSuite.defaults, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let stored = Self.storedDefaultMachine()
                if stored != self.defaultMachine { self.defaultMachine = stored }
            }
        }
    }

    func setDefault(_ host: MoldHost) {
        defaultMachine = host.id
    }
}
