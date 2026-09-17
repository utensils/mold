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

    /// Discover only where THIS machine says it browses a catalog at all.
    /// An older host, or one with catalog access turned off, offers
    /// Installed alone -- and the picker is never drawn for one segment
    /// (design S3).
    static func available(capabilities: Capabilities?) -> [ModelScope] {
        capabilities?.canBrowseCatalog == true ? [.installed, .discover] : [.installed]
    }

    /// A stored `.discover` surviving onto a machine that cannot browse
    /// falls back to `.installed` rather than showing an empty pane.
    static func resolved(stored: ModelScope, available: [ModelScope]) -> ModelScope {
        available.contains(stored) ? stored : .installed
    }
}
