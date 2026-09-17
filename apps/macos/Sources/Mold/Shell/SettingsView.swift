import MoldClient
import SwiftUI

/// The Settings scene (⌘,).
struct SettingsView: View {
    var body: some View {
        TabView {
            MachinesSettings()
                .tabItem { Label("Machines", systemImage: "server.rack") }
            AccountsSettings()
                .tabItem { Label("Accounts", systemImage: "key") }
            StorageSettings()
                .tabItem { Label("Storage", systemImage: "internaldrive") }
            LocalEngineSettings()
                .tabItem { Label("This Mac", systemImage: "cpu") }
        }
        // 440, not 400 -- measured against Accounts, the tallest tab now.
        .frame(width: 560, height: 440)
    }
}
