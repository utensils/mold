import MoldClient
import SwiftUI

/// The Settings scene (⌘,).
struct SettingsView: View {
    var body: some View {
        TabView {
            MachinesSettings()
                .tabItem { Label("Machines", systemImage: "server.rack") }
            LocalEngineSettings()
                .tabItem { Label("This Mac", systemImage: "cpu") }
        }
        .frame(width: 560, height: 400)
    }
}
