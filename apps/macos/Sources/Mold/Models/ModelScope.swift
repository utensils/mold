import MoldClient

/// Which half of "Models" the pane shows: the machine's whole install list,
/// or what could be added from its catalog. A toolbar switcher, not a
/// sidebar row -- the sidebar is about PLACES, and Installed/Discover are
/// one pane in two modes (design decision 8, M5).
enum ModelScope: String, CaseIterable, Hashable {
    case installed, discover

    var label: String {
        switch self {
        case .installed: "Installed"
        case .discover: "Discover"
        }
    }

    /// The manifest is available independently of the optional community catalog.
    static func available(capabilities: Capabilities?) -> [ModelScope] { [.installed, .discover] }

    /// Retain the selected scope when moving between machines; every machine
    /// can offer its manifest independently of community browsing.
    static func resolved(stored: ModelScope, available: [ModelScope]) -> ModelScope {
        available.contains(stored) ? stored : .installed
    }
}
