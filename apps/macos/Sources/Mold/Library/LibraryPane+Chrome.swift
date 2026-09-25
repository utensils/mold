import MoldClient
import SwiftUI

// The pane's own three layers: what it re-reads when the fleet or the shelves
// move under it, the chrome it wears, and what is actually on screen. Split
// from `LibraryPane.swift` past the file-size advisory.
extension LibraryPane {
    /// The pane, plugged in: what it does on appearing, and what it re-does
    /// when the machines or the shelves change under it.
    func watched(_ showing: LibraryShowing) -> some View {
        chrome(showing)
            // The shell loads the library for its sidebar before this pane
            // opens; navigation must not start another full listing.
            // Library can be the launch destination. Read models here too so
            // attachment actions do not depend on Generate having appeared.
            .task(id: attachmentModelKey) { await prepareAttachmentModels() }
            // The clip upscales already running on each machine. One listing
            // per machine, so a job survives a restart and a second Mac.
            .task { await upscales.recover() }
            .onAppear { library.undo.manager = undoManager }
            .onAppear { revealIfNeeded() }
            .onChange(of: undoManager) { _, manager in library.undo.manager = manager }
            .onChange(of: navigation.scope) { _, _ in clearSelection() }
            .onChange(of: navigation.query.tokens) { _, _ in clearSelection() }
            .onChange(of: library.shelves) { _, shelves in navigation.reconcile(with: shelves) }
            .onChange(of: library.rows.value) { _, _ in followMergedTiles() }
            // A click on an already-open Library: `.onAppear` above only
            // fires when the pane is freshly mounted.
            .onChange(of: navigation.reveal) { _, _ in revealIfNeeded() }
    }

    /// The pane, dressed: title, search, toolbar, inspector.
    func chrome(_ showing: LibraryShowing) -> some View {
        @Bindable var navigation = navigation

        return content(showing)
            .failureBanner(hosts)
            .mediaCacheNote(materializer)
            // The OPEN print while the viewer shows one, the grid's selection
            // otherwise (`LibraryInspector+Pane`). Ahead of `.toolbar` so the
            // column's switch is the row's last item -- see `TrailingColumn`.
            .trailingColumn(isShowing: $showsInspector, searchFillsTheColumn: true) {
                inspector(showing.inspected(viewing: viewing))
            }
            .navigationTitle(navigation.scope.title(in: library.shelves))
            .navigationSubtitle(fullSubtitle(showing))
            .searchable(text: $navigation.query.text, tokens: $navigation.query.tokens,
                        suggestedTokens: .constant(suggestedTokens),
                        prompt: "Search, or is:video · tag:name · on:machine") { token in
                Label(token.label, systemImage: token.symbol)
            }
            .onSubmit(of: .search) { commitTypedToken() }
            .searchFocused($isSearchFocused)
            .toolbar { toolbar }
    }

    /// What is actually on screen: a print, the grid, or an explanation.
    @ViewBuilder func content(_ showing: LibraryShowing) -> some View {
        if let viewing, let entry = entry(viewing, in: showing.visible) {
            LibraryViewer(entry: entry, host: host(of: entry), actions: actions,
                          scope: navigation.scope, shelves: library.shelves,
                          enclosingShelf: enclosingShelf, trashCount: library.trashed.count,
                          onClose: { close(viewing) },
                          onStep: { step($0, in: showing.visible) })
        } else if showing.visible.isEmpty {
            empty(showing)
        } else {
            LibraryGrid(
                sections: showing.sections, hosts: hosts.hosts, edge: navigation.edge,
                showsHostBadges: showsHostBadges,
                scope: navigation.scope, actions: actions, entries: showing.visible,
                shelves: library.shelves, enclosingShelf: enclosingShelf,
                trashCount: library.trashed.count,
                selection: $selection, onOpen: { viewing = $0 }
            )
        }
    }
}
