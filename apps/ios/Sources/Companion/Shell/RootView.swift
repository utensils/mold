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
                // The sidebar lists all five destinations. Models and the
                // extra sections stay out of the floating bar so it fits
                // at large text sizes.
                destinationTab(.generate)
                destinationTab(.library)
                destinationTab(.queue)
                destinationTab(.models)
                destinationTab(.machines)
                librarySection(library)
                machinesSection(hosts)
                // A sidebar row, not the sidebar's footer: the footer bar
                // held its text at one size (the audit's Dynamic Type check).
                Tab(value: TabSelection.settings) {
                    SettingsSheet(inSidebar: true)
                } label: {
                    Label { Text("Settings") } icon: { Image(systemName: "gearshape") }
                }
                .defaultVisibility(.hidden, for: .tabBar)
            } else {
                ForEach(destinations) { destinationTab($0) }
            }
            Tab(value: TabSelection.search, role: .search) {
                SearchHome()
            }
        }
        .tabViewStyle(.sidebarAdaptable)
        .sheet(isPresented: $router.showsSettings) {
            // Page-sized on iPad: the default form sheet showed half of it,
            // with the first and last rows under the scroll-edge fades.
            SettingsSheet().presentationSizing(.page)
        }
        .sheet(item: Binding(get: { () -> ModelStore.PendingLicense? in
            guard let pending = stores.models.pendingLicense else { return nil }
            if let owner = pending.presentationOwner, owner != router.presentationID { return nil }
            if let job = pending.recoveryJob, let context = router.licenseDetailContext, context.host == pending.host, context.job == job { return nil }
            return pending
        }, set: { value in
            if value == nil { stores.models.cancelLicense() }
        })) { pending in LicenceSheet(pending: pending) }
        .onDisappear {
            actions.cancelExports()
            stores.models.activePresentationOwners.remove(router.presentationID)
            if stores.models.pendingLicense?.presentationOwner == router.presentationID { stores.models.cancelLicense() }
        }
        .printSheets(presentsActions: router.openedPrint == nil)
        .modifier(RootLinks(router: router))
        .modifier(UndoBridge())
        .environment(router)
        .environment(actions)
        .focusedSceneValue(\.tabSelection, $router.selection)
        .focusedSceneValue(\.showsSettings, $router.showsSettings)
        .focusedSceneValue(\.refresh, RefreshAction { await stores.becameActive() })
        .onAppear {
            stores.models.activePresentationOwners.insert(router.presentationID)
            if stored == Self.searchKey { router.selection = .search }
            else if let restored = Destination(rawValue: stored) { router.selection = .go(restored) }
        }
        .onChange(of: width) { _, new in
            // An iPad window narrowed to compact loses the Models row and the
            // sidebar's shelves and machines.
            guard new != .regular else { return }
            switch router.selection {
            case .go(.models), .machine, .settings: router.selection = .go(.machines)
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
            case .machine, .settings: stored = Destination.machines.rawValue
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
        .badge(destination == .queue ? queue.badge : destination == .library ? library.unreadCount : 0)
        .defaultVisibility(destination.showsInTabBar ? .visible : .hidden, for: .tabBar)
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
