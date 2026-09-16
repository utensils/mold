import MoldClient
import SwiftUI

// The pane's small decisions, kept out of the view so the body stays readable.
extension LibraryPane {

    /// The subtitle, with the trash's retention sentence when there is one.
    /// Its own property because the body grew past what the type checker will
    /// infer in one expression.
    var fullSubtitle: String {
        guard let sentence = retentionSentence else { return subtitle }
        return "\(subtitle) · \(sentence)"
    }

    /// Only worth the ink when the grid can actually be showing two machines.
    var showsHostBadges: Bool {
        guard hosts.hosts.count > 1 else { return false }
        return !navigation.query.tokens.contains { if case .machine = $0 { true } else { false } }
    }

    func entry(_ id: PrintID) -> LibraryEntry? { visible.first { $0.id == id } }

    func host(of entry: LibraryEntry) -> MoldHost? {
        hosts.host(entry.hostID)
    }

    /// Walks the viewer through the list the grid is showing.
    func step(_ delta: Int) {
        guard let viewing, let index = visible.firstIndex(where: { $0.id == viewing })
        else { return }
        let next = min(max(index + delta, 0), visible.count - 1)
        self.viewing = visible[next].id
    }

    // MARK: - What the grid draws

    /// The query the grid is actually drawing: what was typed, plus the
    /// narrowing the chosen shelf adds.
    var resolved: LibraryQuery {
        var query = navigation.query
        query.hiddenCollectionIDs = library.hiddenCollectionIDs
        if let token = navigation.scope.token(in: library.shelves) {
            query.tokens.append(token)
        }
        return query
    }

    var visible: [LibraryEntry] { resolved.apply(to: pool) }

    var sections: [LibrarySection] { LibraryGrouping.byDay(visible) }

    var selected: [LibraryEntry] {
        visible.filter { selection.items.contains($0.id) }
    }
}
