/// Keep viewing the same filtered list when its current media leaves it.
public enum LibraryViewerCursor {
    public static func afterRemoval<ID: Hashable>(
        _ current: ID, previous: [ID], remaining: [ID]
    ) -> ID? {
        let survivors = Set(remaining)
        if survivors.contains(current) { return current }
        if let index = previous.firstIndex(of: current) {
            for id in previous.dropFirst(index + 1) where survivors.contains(id) { return id }
            for id in previous.prefix(index).reversed() where survivors.contains(id) { return id }
        }
        return remaining.first
    }
}

extension LibraryViewerCursor {
    /// A successful removal of a tile's lead must not hide its failed mirror.
    public static func afterRemoval(
        _ current: PrintID, previous: [LibraryEntry], remaining: [LibraryEntry]
    ) -> PrintID? {
        let byCopy = Dictionary(remaining.flatMap { entry in
            entry.everyCopy.map { ($0.id, entry.id) }
        }, uniquingKeysWith: { first, _ in first })
        func survivor(_ entry: LibraryEntry) -> PrintID? {
            entry.everyCopy.lazy.compactMap { byCopy[$0.id] }.first
        }
        if let retained = byCopy[current] { return retained }
        if let index = previous.firstIndex(where: { $0.everyCopy.contains { $0.id == current } }) {
            if let retained = survivor(previous[index]) { return retained }
            for entry in previous.dropFirst(index + 1) {
                if let next = survivor(entry) { return next }
            }
            for entry in previous.prefix(index).reversed() {
                if let next = survivor(entry) { return next }
            }
        }
        return remaining.first?.id
    }
}
