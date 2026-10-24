import Foundation

/// The Library's Filter menu: one pick at a time, as Photos offers it. A
/// pick is one token; choosing a row replaces the pick and keeps every other
/// token (machines, families, collections, tags, text). The Model Family and
/// Machine submenus combine with it.
public enum LibraryFilter: String, CaseIterable, Identifiable, Sendable {
    case all, favorites, photos, videos, meshes
    case inCollection, notInCollection, fromSource, tagged, untagged
    case portrait, landscape, square

    public var id: Self { self }

    /// The menu's groups, in order, each drawn with a divider before it.
    public static let sections: [[LibraryFilter]] = [
        [.all],
        [.favorites, .photos, .videos, .meshes],
        [.inCollection, .notInCollection, .fromSource, .tagged, .untagged],
        [.portrait, .landscape, .square],
    ]

    public var token: LibraryToken? {
        switch self {
        case .all: nil
        case .favorites: .favorite
        case .photos: .kind(.picture)
        case .videos: .kind(.clip)
        case .meshes: .kind(.mesh)
        case .inCollection: .filed(true)
        case .notInCollection: .filed(false)
        case .fromSource: .fromSource
        case .tagged: .tagged(true)
        case .untagged: .tagged(false)
        case .portrait: .orientation(.portrait)
        case .landscape: .orientation(.landscape)
        case .square: .orientation(.square)
        }
    }

    public var title: String {
        switch self {
        case .all: String(localized: "All Items")
        case .favorites: String(localized: "Favourites")
        case .photos: String(localized: "Photos")
        case .videos: String(localized: "Videos")
        case .meshes: String(localized: "3D")
        case .inCollection: String(localized: "In a Collection")
        case .notInCollection: String(localized: "Not in a Collection")
        case .fromSource: String(localized: "Made from a Source")
        case .tagged: String(localized: "Has Tags")
        case .untagged: String(localized: "Untagged")
        case .portrait: String(localized: "Portrait")
        case .landscape: String(localized: "Landscape")
        case .square: String(localized: "Square")
        }
    }

    public var symbol: String {
        switch self {
        case .all: "square.grid.3x3.fill"
        case .favorites: "star"
        case .photos: "photo"
        case .videos: "film"
        case .meshes: "cube"
        case .inCollection: "rectangle.stack"
        case .notInCollection: "rectangle.stack.badge.minus"
        case .fromSource: "photo.badge.arrow.down"
        case .tagged: "tag"
        case .untagged: "tag.slash"
        case .portrait: Self.symbol(for: .portrait)
        case .landscape: Self.symbol(for: .landscape)
        case .square: Self.symbol(for: .square)
        }
    }

    static func symbol(for orientation: PrintOrientation) -> String {
        switch orientation {
        case .portrait: "rectangle.portrait"
        case .landscape: "rectangle"
        case .square: "square"
        }
    }

    /// The checked row, or nil when the query holds several picks at once
    /// (from search syntax): no single row describes that.
    public static func selected(in query: LibraryQuery) -> Self? {
        let picks = query.tokens.filter(isPick)
        if picks.isEmpty { return .all }
        guard picks.count == 1 else { return nil }
        return allCases.first { $0.token == picks[0] }
    }

    public func applying(to query: LibraryQuery) -> LibraryQuery {
        var next = query
        next.tokens.removeAll(where: Self.isPick)
        if let token { next.tokens.append(token) }
        return next
    }

    /// The Filter button's visible title: the pick, then the families, so
    /// the state reads without opening the menu.
    public static func summary(for query: LibraryQuery) -> String {
        let pick = selected(in: query)
        let families = query.tokens.compactMap { if case let .family(_, name) = $0 { name } else { nil } }
        var parts: [String] = []
        if pick != .all { parts.append(pick?.title ?? String(localized: "Multiple Filters")) }
        if families.count == 1 { parts.append(families[0]) }
        if families.count > 1 { parts.append(String(localized: "\(families.count) Families")) }
        return parts.isEmpty ? Self.all.title : parts.joined(separator: " · ")
    }

    /// Whether the menu narrows anything (a pick or a family).
    public static func isActive(_ query: LibraryQuery) -> Bool {
        query.tokens.contains { token in
            if case .family = token { return true }
            return isPick(token)
        }
    }

    /// Clear Filters: drops the pick and the families, keeping machines,
    /// collections, tags and text, which have their own controls.
    public static func clearing(_ query: LibraryQuery) -> LibraryQuery {
        LibraryFamilies.clearing(all.applying(to: query))
    }

    static func isPick(_ token: LibraryToken) -> Bool {
        switch token {
        case .favorite, .kind, .filed, .fromSource, .tagged, .orientation: true
        case .tag, .machine, .collection, .family: false
        }
    }
}
