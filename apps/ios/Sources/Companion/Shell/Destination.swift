import MoldClient
import SwiftUI

/// Where the app can be: the Mac app's five destinations, in its order, words
/// and symbols (DESIGN.md §3–§4). Search is not a destination -- it is a
/// role the tab bar gives its own glass button -- so it lives on
/// `TabSelection`, not here.
enum Destination: String, CaseIterable, Identifiable, Hashable {
    case generate
    case library
    case queue
    case models
    case machines

    var id: Self { self }

    var title: String {
        switch self {
        case .generate: String(localized: "Generate")
        case .library: String(localized: "Library")
        case .queue: String(localized: "Queue")
        case .models: String(localized: "Models")
        case .machines: String(localized: "Machines")
        }
    }

    var symbol: String {
        switch self {
        case .generate: "wand.and.sparkles"
        case .library: "photo.on.rectangle.angled"
        case .queue: "list.bullet.indent"
        case .models: "cube"
        case .machines: "server.rack"
        }
    }

    /// ⌘1–⌘5, the Mac's own bindings.
    var shortcut: String {
        String((Self.allCases.firstIndex(of: self) ?? 0) + 1)
    }

    /// Models belongs to a machine -- on the Mac, "the machine picked here is
    /// the one the Models pane shows" -- so an iPhone reaches it from
    /// Machines, and only the iPad sidebar lists it on its own.
    var showsInTabBar: Bool { self != .models }
}

/// What the root `TabView` selects: a destination, or the Search tab.
enum TabSelection: Hashable {
    case go(Destination)
    case search
    /// iPad sidebar only: one Library shelf, or one machine.
    case shelf(LibraryScope)
    case machine(UUID)
}
