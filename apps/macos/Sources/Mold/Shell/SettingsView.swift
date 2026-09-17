import MoldClient
import SwiftUI

/// Nine tabs, in the design's own pane-map order: this Mac's own
/// preferences, then what a render is made of, then the fleet, then the raw
/// table. A stable string id per case, so `SettingsUAT` can open one at
/// launch without a menu press and a selection binding can survive a rename
/// of `title`.
enum SettingsTab: String, CaseIterable, Identifiable {
    case general, generation, expansion, library, performance, accounts, machines, thisMac, advanced

    var id: String { rawValue }

    var title: String {
        switch self {
        case .general: "General"
        case .generation: "Generation"
        case .expansion: "Expansion"
        case .library: "Library"
        case .performance: "Performance"
        case .accounts: "Accounts"
        case .machines: "Machines"
        case .thisMac: "This Mac"
        case .advanced: "Advanced"
        }
    }

    var symbol: String {
        switch self {
        case .general: "gearshape"
        case .generation: "photo"
        case .expansion: "wand.and.stars"
        case .library: "photo.stack"
        case .performance: "speedometer"
        case .accounts: "key"
        case .machines: "server.rack"
        case .thisMac: "cpu"
        case .advanced: "gearshape.2"
        }
    }
}

/// The Settings scene (⌘,).
struct SettingsView: View {
    @State private var selection = SettingsUAT.initialTab()

    var body: some View {
        TabView(selection: $selection) {
            ForEach(SettingsTab.allCases) { tab in
                content(for: tab)
                    .tabItem { Label(tab.title, systemImage: tab.symbol) }
                    .tag(tab)
            }
        }
        .frame(width: CGFloat(SettingsLayout.width), height: CGFloat(SettingsLayout.height))
    }

    @ViewBuilder private func content(for tab: SettingsTab) -> some View {
        switch tab {
        case .general: GeneralSettings()
        case .generation: GenerationSettings()
        case .expansion: ExpansionSettings()
        case .library: LibrarySettings()
        case .performance: PerformanceSettings()
        case .accounts: AccountsSettings()
        case .machines: MachinesSettings()
        case .thisMac: LocalEngineSettings()
        case .advanced: AdvancedSettings()
        }
    }
}
