import Foundation

public enum LibraryMediaFilter: String, CaseIterable, Identifiable, Sendable {
    case all, photos, videos, meshes
    public var id: Self { self }
    public var kind: PrintKind? {
        switch self {
        case .all: nil
        case .photos: .picture
        case .videos: .clip
        case .meshes: .mesh
        }
    }
    public var title: String {
        switch self {
        case .all: String(localized: "All Media")
        case .photos: String(localized: "Photos")
        case .videos: String(localized: "Videos")
        case .meshes: String(localized: "3D")
        }
    }
    /// No checked row for an OR-search across several media types.
    public static func selected(in query: LibraryQuery) -> Self? {
        let kinds = Set(query.tokens.compactMap { token -> PrintKind? in
            if case let .kind(kind) = token { kind } else { nil }
        })
        if kinds.isEmpty { return .all }
        guard kinds.count == 1 else { return nil }
        return allCases.first { $0.kind == kinds.first }
    }

    public func applying(to query: LibraryQuery) -> LibraryQuery {
        var next = query
        next.tokens.removeAll { if case .kind = $0 { true } else { false } }
        if let kind { next.tokens.append(.kind(kind)) }
        return next
    }
}

