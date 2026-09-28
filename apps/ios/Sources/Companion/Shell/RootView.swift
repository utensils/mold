import SwiftUI

/// The app's one `TabView`: a tab bar on iPhone, the Mac-like sidebar on iPad
/// (`.sidebarAdaptable`). Each destination owns its own `NavigationStack`, so
/// switching tabs keeps every stack where it was (DESIGN.md §4).
struct RootView: View {
    /// Per window, so two iPad windows can each be somewhere different.
    @SceneStorage("selection") private var stored = Destination.generate.rawValue
    @State private var selection = TabSelection.go(.generate)
    @State private var showsSettings = false
    @Environment(\.horizontalSizeClass) private var width

    /// Models joins the list only where there is a sidebar. On iPhone
    /// `.defaultVisibility(.hidden, for: .tabBar)` is ignored, and a sixth tab
    /// pushed Machines and Search into a "More" tab.
    private var destinations: [Destination] {
        Destination.allCases.filter { $0.showsInTabBar || width == .regular }
    }

    var body: some View {
        TabView(selection: $selection) {
            ForEach(destinations) { destination in
                Tab(value: TabSelection.go(destination)) {
                    DestinationHome(destination: destination, selection: $selection,
                                    showsSettings: $showsSettings)
                } label: {
                    Label { Text(destination.title) } icon: { Image(systemName: destination.symbol) }
                }
            }
            Tab(value: TabSelection.search, role: .search) {
                SearchHome()
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        .sheet(isPresented: $showsSettings) { SettingsSheet() }
        .focusedSceneValue(\.tabSelection, $selection)
        .onAppear {
            if let restored = Destination(rawValue: stored) { selection = .go(restored) }
        }
        .onChange(of: width) { _, new in
            // An iPad window narrowed to compact loses the Models row.
            if new != .regular, selection == .go(.models) { selection = .go(.machines) }
        }
        .onChange(of: selection) { _, new in
            // ⌘4 (or a restored iPad selection) on a phone: Models lives
            // under Machines there.
            if new == .go(.models), width != .regular { selection = .go(.machines); return }
            if case .go(let destination) = new { stored = destination.rawValue }
        }
    }
}

extension FocusedValues {
    /// Lets the Go menu's ⌘1–⌘5 reach the focused window's tab selection.
    @Entry var tabSelection: Binding<TabSelection>?
}

/// ⌘1–⌘5 on iPad (and on an iPhone with a keyboard), the Mac's own bindings.
struct GoCommands: Commands {
    @FocusedBinding(\.tabSelection) private var selection

    var body: some Commands {
        CommandMenu("Go") {
            ForEach(Destination.allCases) { destination in
                Button { selection = .go(destination) } label: { Text(destination.title) }
                    .keyboardShortcut(KeyEquivalent(Character(destination.shortcut)), modifiers: .command)
                    .disabled(selection == nil)
            }
        }
    }
}
