import MoldClient
import SwiftUI

// How the pane plugs the inspector in. Its own file so the trailing column is
// built in one place: the column, the Library menu and File ▸ Export… all read
// `LibraryShowing.inspected(viewing:)`, and a second reading of the selection
// is what made the inspector say "Nothing selected" over an open print.
extension LibraryPane {
    func inspector(_ entries: [LibraryEntry]) -> some View {
        LibraryInspector(entries: entries,
                         host: entries.first.flatMap(host(of:)),
                         scope: navigation.scope, actions: actions,
                         filterByTag: { navigation.query.tokens.append(.tag($0)) })
    }
}
