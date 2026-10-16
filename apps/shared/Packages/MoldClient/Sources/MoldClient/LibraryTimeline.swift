import Foundation

/// The merged, sorted timeline (`LibraryMerge`) with the indexes that keep a
/// lookup -- and a small delta -- from walking the whole library.
///
/// `rebuild` is the definition: merge every row, then sort. `patch` is the
/// shortcut for the common delta, and it only ever produces what `rebuild`
/// would have: a row whose merge identity (filename, byte size, seed/model
/// identity, ordering timestamp) is unchanged is replaced in place, and a
/// brand-new row that shares no name, identity or sync link with anything is
/// inserted at its ordered position. Anything else -- a copy joining or
/// leaving a tile, a re-dated print -- answers `false`, changes nothing, and
/// the caller rebuilds.
public struct LibraryTimeline: Sendable {
    public enum Order: Sendable {
        /// The live timeline: newest print first.
        case newestFirst
        /// Recently Deleted: most recently trashed first.
        case recentlyTrashedFirst
    }

    public let order: Order
    public private(set) var tiles: [LibraryEntry] = []
    /// Every copy's id -> the tile it is shown under.
    private var positions: [PrintID: Int] = [:]
    private var filenames: [String: Int] = [:]
    private var identities: [String: Int] = [:]
    private var links: [PrintID: PrintID] = [:]
    private var linked: Set<PrintID> = []

    public init(order: Order = .newestFirst) { self.order = order }

    public mutating func rebuild(_ entries: [LibraryEntry], localHost: MoldHost.ID?,
                                 links: [PrintID: PrintID] = [:]) {
        let key = self.key
        tiles = LibraryMerge.merge(entries, localHost: localHost, links: links)
            .sorted { key($0.print) > key($1.print) }
        self.links = links
        linked = Set(links.keys).union(links.values)
        filenames = [:]
        identities = [:]
        for copy in tiles.lazy.flatMap(\.everyCopy) { count(copy.print, by: 1) }
        reindex(from: 0)
    }

    /// The tile a print is shown under, whichever copy `id` names.
    public func tile(containing id: PrintID) -> LibraryEntry? { positions[id].map { tiles[$0] } }

    /// One copy by its own machine's id, lead or not.
    public func copy(_ id: PrintID) -> LibraryEntry? {
        tile(containing: id)?.everyCopy.first { $0.id == id }
    }

    /// Applies changed and removed rows without a merge where that cannot
    /// differ from one. All or nothing: `false` leaves the timeline untouched.
    public mutating func patch(upserted: [LibraryEntry], removed: [PrintID],
                               links: [PrintID: PrintID] = [:]) -> Bool {
        guard links == self.links else { return false }
        let ids = upserted.map(\.id)
        guard Set(ids).count == ids.count, Set(ids).isDisjoint(with: removed) else { return false }
        var updates: [(position: Int, entry: LibraryEntry)] = []
        var inserts: [LibraryEntry] = []
        for entry in upserted {
            if let position = positions[entry.id] {
                guard let old = tiles[position].everyCopy.first(where: { $0.id == entry.id }),
                      Self.sameIdentity(old.print, entry.print), key(old.print) == key(entry.print)
                else { return false }
                updates.append((position, entry))
            } else {
                guard !linked.contains(entry.id), filenames[entry.print.filename, default: 0] == 0,
                      LibraryMerge.identity(of: entry.print).map({ identities[$0, default: 0] == 0 }) ?? true
                else { return false }
                inserts.append(entry)
            }
        }
        // A tie with an existing tile is ordered by machine-list position in a
        // full merge, which an insertion cannot reproduce; two new rows can
        // still match each other.
        guard inserts.allSatisfy({ !hasTile(keyed: key($0.print)) }),
              Set(inserts.map(\.print.filename)).count == inserts.count,
              Set(inserts.compactMap { LibraryMerge.identity(of: $0.print) }).count
                == inserts.compactMap({ LibraryMerge.identity(of: $0.print) }).count
        else { return false }
        var gone: [Int] = []
        for id in removed {
            guard let position = positions[id] else { continue }
            let tile = tiles[position]
            guard tile.copies.isEmpty, filenames[tile.print.filename] == 1,
                  LibraryMerge.identity(of: tile.print).map({ identities[$0] == 1 }) ?? true
            else { return false }
            gone.append(position)
        }

        for (position, entry) in updates { replace(entry, at: position) }
        guard !gone.isEmpty || !inserts.isEmpty else { return true }
        var first = tiles.count
        for position in gone.sorted(by: >) {
            let tile = tiles.remove(at: position)
            count(tile.print, by: -1)
            positions[tile.id] = nil
            first = min(first, position)
        }
        for entry in inserts {
            let at = insertionIndex(for: entry.print)
            tiles.insert(entry, at: at)
            count(entry.print, by: 1)
            first = min(first, at)
        }
        reindex(from: first)
        return true
    }

    private mutating func replace(_ entry: LibraryEntry, at position: Int) {
        var tile = tiles[position]
        if tile.id == entry.id {
            let copies = tile.copies
            tile = entry.replacingPrint(entry.print)
            tile.copies = copies
        } else if let index = tile.copies.firstIndex(where: { $0.id == entry.id }) {
            tile.copies[index] = entry.replacingPrint(entry.print)
        }
        tiles[position] = tile
    }

    private var key: @Sendable (GalleryPrint) -> UInt64 {
        switch order {
        case .newestFirst: { $0.timestamp }
        case .recentlyTrashedFirst: { $0.trashedAt ?? 0 }
        }
    }

    /// After every tile whose key is at least this one's: where a stable sort
    /// of a merge that appended this print last would put it.
    private func insertionIndex(for print: GalleryPrint) -> Int { insertionIndex(for: key(print)) }

    private func insertionIndex(for target: UInt64) -> Int {
        var low = 0
        var high = tiles.count
        while low < high {
            let mid = (low + high) / 2
            if key(tiles[mid].print) >= target { low = mid + 1 } else { high = mid }
        }
        return low
    }

    private func hasTile(keyed target: UInt64) -> Bool {
        let at = insertionIndex(for: target)
        return at > 0 && key(tiles[at - 1].print) == target
    }

    private static func sameIdentity(_ old: GalleryPrint, _ new: GalleryPrint) -> Bool {
        old.filename == new.filename && old.sizeBytes == new.sizeBytes
            && old.timestamp == new.timestamp && old.trashedAt == new.trashedAt
            && LibraryMerge.identity(of: old) == LibraryMerge.identity(of: new)
    }

    private mutating func count(_ print: GalleryPrint, by delta: Int) {
        filenames[print.filename, default: 0] += delta
        if filenames[print.filename] == 0 { filenames[print.filename] = nil }
        if let identity = LibraryMerge.identity(of: print) {
            identities[identity, default: 0] += delta
            if identities[identity] == 0 { identities[identity] = nil }
        }
    }

    private mutating func reindex(from start: Int) {
        if start == 0 { positions.removeAll(keepingCapacity: true) }
        for position in start..<tiles.count {
            for copy in tiles[position].everyCopy { positions[copy.id] = position }
        }
    }
}
