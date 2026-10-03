import MoldClient
import SwiftUI

/// Gallery-wide derivations belong to a data/query revision, never a scroll frame.
final class LibraryGridProjectionCache {
    private struct Key: Equatable {
        let revision: Int
        let scope: LibraryScope
        let query: LibraryQuery
    }
    private var key: Key?
    private var projection = LibraryGridProjection(entries: [])
    private(set) var derivations = 0
    var targetLookups: Int { projection.targetLookups }

    func project(entries: [LibraryEntry], revision: Int, scope: LibraryScope,
                 query: LibraryQuery) -> LibraryGridProjection {
        let key = Key(revision: revision, scope: scope, query: query)
        if self.key != key {
            self.key = key
            projection = LibraryGridProjection(entries: entries)
            derivations += 1
        }
        return projection
    }
}

final class LibraryGridProjection {
    let showsHost: Bool
    let favorites: [LibraryEntry]
    private let entries: [LibraryEntry]
    private let positions: [PrintID: Int]
    private(set) var targetLookups = 0

    init(entries: [LibraryEntry]) {
        self.entries = entries
        showsHost = Set(entries.flatMap(\.hostNames)).count > 1
        favorites = entries.filter(\.print.isFavorite)
        positions = Dictionary(uniqueKeysWithValues: entries.enumerated().map { ($0.element.id, $0.offset) })
    }

    func entry(_ id: PrintID) -> LibraryEntry? {
        positions[id].map { entries[$0] }
    }

    /// Bound page construction even when the library contains tens of thousands
    /// of prints. Stable domain tags retain the selected page as the window moves.
    func pages(around id: PrintID) -> ArraySlice<LibraryEntry> {
        guard let index = positions[id] else { return entries[0..<0] }
        return entries[max(0, index - 2)..<min(entries.count, index + 3)]
    }

    /// A refreshed snapshot can remove the old anchor or reorder the selected
    /// print outside its window. Repair before rendering, not one frame later.
    func anchor(for selected: PrintID, preferred: PrintID) -> PrintID {
        guard let index = positions[selected], let center = positions[preferred],
              index >= max(0, center - 2), index <= min(entries.count - 1, center + 2)
        else { return selected }
        return preferred
    }

    func shouldRecenter(selected: PrintID, anchor: PrintID) -> Bool {
        guard selected != anchor, let index = positions[selected], let center = positions[anchor] else { return false }
        return index <= max(0, center - 2) || index >= min(entries.count - 1, center + 2)
    }

    func step(_ by: Int, from id: PrintID) -> PrintID? {
        guard let index = positions[id], entries.indices.contains(index + by) else { return nil }
        return entries[index + by].id
    }

    /// SwiftUI reports an unordered viewport-sized list. Do not scan the gallery.
    func firstVisible(_ ids: [PrintID]) -> PrintID? {
        var first: (id: PrintID, index: Int)?
        for id in ids {
            targetLookups += 1
            guard let index = positions[id] else { continue }
            if first == nil || index < first!.index { first = (id, index) }
        }
        return first?.id
    }
}

/// Geometry updates are observations, not view state invalidations. Freeze the
/// exact viewport while navigation covers it, including a partial tile/header.
final class LibraryViewport {
    private var offset: CGFloat = 0
    private var covered = false
    func report(offset: CGFloat) { if !covered { self.offset = offset } }
    func cover() { covered = true }
    func uncover() -> CGFloat { covered = false; return offset }
}
