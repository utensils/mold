import SwiftUI

/// The window's one split view.
///
/// A single `Window` rather than a `WindowGroup`: the Library merges every
/// host's index into one in-memory timeline, and a second window would mean a
/// second copy of it arguing with the first.
struct RootView: View {
    @Environment(HostStore.self) private var hosts
    @State private var destination: Destination = .generate

    var body: some View {
        NavigationSplitView {
            Sidebar(destination: $destination)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            DestinationDetail(destination: destination)
        }
        .navigationTitle("Mold")
        .task { await hosts.refreshAll() }
    }
}

enum Destination: Hashable, CaseIterable, Identifiable {
    case generate, library, queue, models

    var id: Self { self }

    var title: String {
        switch self {
        case .generate: "Generate"
        case .library: "Library"
        case .queue: "Queue"
        // "Models", never "Styles". A person choosing one needs to know what
        // it does, and the manifest already says so in plain language.
        case .models: "Models"
        }
    }

    var symbol: String {
        switch self {
        case .generate: "wand.and.sparkles"
        case .library: "photo.on.rectangle.angled"
        case .queue: "list.bullet.indent"
        case .models: "cube"
        }
    }
}
