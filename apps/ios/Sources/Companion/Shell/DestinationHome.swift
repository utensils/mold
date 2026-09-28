import SwiftUI

/// One destination's root: its own `NavigationStack`, so a tab keeps its place
/// while you visit another. Until a machine is added every destination says
/// what it will hold and how to get there, in DESIGN.md §7's words -- never a
/// blank screen and never a disappearing tab.
struct DestinationHome: View {
    @Environment(AppRouter.self) private var router
    @Environment(HostStore.self) private var hosts
    let destination: Destination

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(destination.title)
                .toolbar {
                    if destination == .machines {
                        ToolbarItem(placement: .topBarLeading) {
                            Button { router.showsSettings = true } label: {
                                Label("Settings", systemImage: "gearshape")
                            }
                        }
                    }
                }
        }
    }

    @ViewBuilder private var content: some View {
        switch destination {
        case .generate:
            GenerateView()
        case .library:
            LibraryView()
        case .queue:
            QueueView()
        case .models:
            ModelsView()
        case .machines:
            MachinesView()
        }
    }
}

/// The Search tab: the Library, searched -- the same grid, tokens and all.
struct SearchHome: View {
    var body: some View {
        NavigationStack { LibraryView(searchFocused: true) }
    }
}
