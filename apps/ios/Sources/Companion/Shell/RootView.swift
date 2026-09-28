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
    @Environment(QueueStore.self) private var queue
    @Environment(CompanionStores.self) private var stores
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts

    /// Models joins the list only where there is a sidebar. On iPhone
    /// `.defaultVisibility(.hidden, for: .tabBar)` is ignored, and a sixth tab
    /// pushed Machines and Search into a "More" tab.
    private var destinations: [Destination] {
        Destination.allCases.filter { $0.showsInTabBar || width == .regular }
    }

    var body: some View {
        @Bindable var router = router
        TabView(selection: $router.selection) {
            if width == .regular {
                destinationTab(.generate)
                librarySection(library)
                destinationTab(.queue)
                destinationTab(.models)
                machinesSection(hosts)
            } else {
                ForEach(destinations) { destinationTab($0) }
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
        .sheet(isPresented: $router.showsSettings) {
            // Page-sized on iPad: the default form sheet showed half of it,
            // with the first and last rows under the scroll-edge fades.
            SettingsSheet().presentationSizing(.page)
        }
        .printSheets()
        .modifier(RootLinks(router: router))
        .modifier(UndoBridge())
        .environment(router)
        .environment(actions)
        .focusedSceneValue(\.tabSelection, $router.selection)
        .focusedSceneValue(\.showsSettings, $router.showsSettings)
        .focusedSceneValue(\.refresh, RefreshAction { await stores.becameActive() })
        .onAppear {
            if stored == Self.searchKey { router.selection = .search }
            else if let restored = Destination(rawValue: stored) { router.selection = .go(restored) }
        }
        .onChange(of: width) { _, new in
            // An iPad window narrowed to compact loses the Models row and the
            // sidebar's shelves and machines.
            guard new != .regular else { return }
            switch router.selection {
            case .go(.models), .machine: router.selection = .go(.machines)
            case .shelf: router.selection = .go(.library)
            default: break
            }
        }
        .onChange(of: router.selection) { _, new in
            // ⌘4 (or a restored iPad selection) on a phone: Models lives
            // under Machines there.
            if new == .go(.models), width != .regular { router.selection = .go(.machines); return }
            switch new {
            case .go(let destination): stored = destination.rawValue
            case .search: stored = Self.searchKey
            case .shelf: stored = Destination.library.rawValue
            case .machine: stored = Destination.machines.rawValue
            }
        }
    }
}

extension RootView {
    func destinationTab(_ destination: Destination) -> some TabContent<TabSelection> {
        Tab(value: TabSelection.go(destination)) {
            DestinationHome(destination: destination)
        } label: {
            Label { Text(destination.title) } icon: { Image(systemName: destination.symbol) }
        }
        .badge(destination == .queue ? queue.badge : 0)
    }

    /// What `@SceneStorage` keeps when the Search tab was last selected.
    static let searchKey = "search"
}

extension FocusedValues {
    /// Lets the Go menu's ⌘1–⌘5 reach the focused window's tab selection.
    @Entry var tabSelection: Binding<TabSelection>?
    /// Lets ⌘, open Settings from any tab, as on the Mac.
    @Entry var showsSettings: Binding<Bool>?
    /// ⌘R: ask every machine again.
    @Entry var refresh: RefreshAction?
}

/// ⌘1–⌘5 on iPad (and on an iPhone with a keyboard), the Mac's own bindings.
struct GoCommands: Commands {
    @FocusedBinding(\.tabSelection) private var selection
    @FocusedBinding(\.showsSettings) private var showsSettings
    @FocusedValue(\.refresh) private var refresh

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
            Divider()
            Button("Search") { selection = .search }
                .keyboardShortcut("f", modifiers: .command)
                .disabled(selection == nil)
        }
        CommandGroup(after: .toolbar) {
            Button("Refresh") { if let refresh { Task { await refresh.run() } } }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(refresh == nil)
        }
    }
}

/// A focused-scene action for ⌘R. A struct, not a bare closure, so the
/// focused value is comparable by identity of the window that set it.
struct RefreshAction {
    let run: () async -> Void
}
