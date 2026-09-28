import SwiftUI

/// The app's one `TabView`: a tab bar on iPhone, the Mac-like sidebar on iPad
/// (`.sidebarAdaptable`). Each destination owns its own `NavigationStack`, so
/// switching tabs keeps every stack where it was (DESIGN.md §4).
struct RootView: View {
    /// Per window, so two iPad windows can each be somewhere different.
    @SceneStorage("selection") private var stored = Destination.generate.rawValue
    /// Where THIS window is. Per window, never shared through the stores.
    @State private var router = AppRouter()
    /// This window's share/save/tag doors (`PrintActions`).
    @State private var actions: PrintActions

    init(stores: CompanionStores) {
        _actions = State(initialValue: PrintActions(hosts: stores.hosts))
    }
    @Environment(\.horizontalSizeClass) private var width

    /// Models joins the list only where there is a sidebar. On iPhone
    /// `.defaultVisibility(.hidden, for: .tabBar)` is ignored, and a sixth tab
    /// pushed Machines and Search into a "More" tab.
    private var destinations: [Destination] {
        Destination.allCases.filter { $0.showsInTabBar || width == .regular }
    }

    var body: some View {
        @Bindable var router = router
        TabView(selection: $router.selection) {
            ForEach(destinations) { destination in
                Tab(value: TabSelection.go(destination)) {
                    DestinationHome(destination: destination)
                } label: {
                    Label { Text(destination.title) } icon: { Image(systemName: destination.symbol) }
                }
            }
            Tab(value: TabSelection.search, role: .search) {
                SearchHome()
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        // The iPad sidebar's footer, where DESIGN.md §4 puts Settings.
        .tabViewSidebarBottomBar {
            Button { router.showsSettings = true } label: {
                Label("Settings", systemImage: "gearshape")
            }
        }
        .sheet(isPresented: $router.showsSettings) { SettingsSheet() }
        .printSheets()
        .environment(router)
        .environment(actions)
        .focusedSceneValue(\.tabSelection, $router.selection)
        .focusedSceneValue(\.showsSettings, $router.showsSettings)
        .onAppear {
            if stored == Self.searchKey { router.selection = .search }
            else if let restored = Destination(rawValue: stored) { router.selection = .go(restored) }
        }
        .onChange(of: width) { _, new in
            // An iPad window narrowed to compact loses the Models row.
            if new != .regular, router.selection == .go(.models) { router.selection = .go(.machines) }
        }
        .onChange(of: router.selection) { _, new in
            // ⌘4 (or a restored iPad selection) on a phone: Models lives
            // under Machines there.
            if new == .go(.models), width != .regular { router.selection = .go(.machines); return }
            switch new {
            case .go(let destination): stored = destination.rawValue
            case .search: stored = Self.searchKey
            }
        }
    }
}

extension RootView {
    /// What `@SceneStorage` keeps when the Search tab was last selected.
    static let searchKey = "search"
}

extension FocusedValues {
    /// Lets the Go menu's ⌘1–⌘5 reach the focused window's tab selection.
    @Entry var tabSelection: Binding<TabSelection>?
    /// Lets ⌘, open Settings from any tab, as on the Mac.
    @Entry var showsSettings: Binding<Bool>?
}

/// ⌘1–⌘5 on iPad (and on an iPhone with a keyboard), the Mac's own bindings.
struct GoCommands: Commands {
    @FocusedBinding(\.tabSelection) private var selection
    @FocusedBinding(\.showsSettings) private var showsSettings

    var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { showsSettings = true }
                .keyboardShortcut(",", modifiers: .command)
                .disabled(showsSettings == nil)
        }
        CommandMenu("Go") {
            ForEach(Destination.allCases) { destination in
                Button { selection = .go(destination) } label: { Text(destination.title) }
                    .keyboardShortcut(KeyEquivalent(Character(destination.shortcut)), modifiers: .command)
                    .disabled(selection == nil)
            }
        }
    }
}
