import MoldClient
import SwiftUI

// The pane's small decisions, kept out of the view so the body stays readable.
extension LibraryPane {

    /// The subtitle, with the trash's retention sentence when there is one.
    /// Its own function because the body grew past what the type checker will
    /// infer in one expression.
    func fullSubtitle(_ showing: LibraryShowing) -> String {
        guard let sentence = retentionSentence else { return subtitle(showing) }
        return "\(subtitle(showing)) · \(sentence)"
    }

    /// Only worth the ink when the grid can actually be showing two machines.
    var showsHostBadges: Bool {
        guard hosts.hosts.count > 1 else { return false }
        return !navigation.query.tokens.contains { if case .machine = $0 { true } else { false } }
    }

    func entry(_ id: PrintID, in visible: [LibraryEntry]) -> LibraryEntry? {
        visible.first { $0.id == id }
    }

    /// `navigation.reveal`'s one consumer: opens the named print and clears
    /// the channel right back, so a later visit to the pane does not reopen
    /// it (design M6 S5).
    func revealIfNeeded() {
        guard let reveal = navigation.reveal else { return }
        selection = LibraryCursor.Selection(items: [reveal], anchor: reveal, lead: reveal)
        viewing = reveal
        navigation.reveal = nil
    }

    func host(of entry: LibraryEntry) -> MoldHost? {
        hosts.host(entry.hostID)
    }

    /// Walks the viewer through the list the grid is showing.
    func step(_ delta: Int, in visible: [LibraryEntry]) {
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

    // MARK: - The viewer, the cursor, and what a print seeds

    /// A new shelf is a new list, and a selection made in the old one names
    /// prints that may not be in it.
    func clearSelection() {
        selection = LibraryCursor.Selection.empty
        viewing = nil
    }

    /// Leaving the viewer puts the cursor back on the print you were looking
    /// at, so the arrow keys carry on from there rather than from nothing.
    func close(_ viewed: PrintID) {
        selection = LibraryCursor.Selection(items: [viewed], anchor: viewed, lead: viewed)
        viewing = nil
    }

    /// Seeds the Generate pane from a finished print and goes there.
    ///
    /// The model is adopted from the machine that MADE the print, because a
    /// model installed on one host is not available on another.
    func reuse(_ entry: LibraryEntry) {
        let metadata = entry.print.metadata
        generate.draft = RenderDraft(reusing: metadata)
        let fence = reuseStore.begin()
        if let name = metadata.model {
            if let model = models.model(named: name, on: entry.hostID) {
                generate.adopt(model: model, on: entry.hostID, keepingDraft: true)
            } else {
                // The rest of the recipe still restores -- the numbers, the
                // filing, the sampler -- and the style chip keeps the name
                // the print was made with. Saying so is the whole fix: the
                // controls would otherwise be reconciled against nothing and
                // silently describe a model that is not there.
                reuseStore.notice = "\(name) isn\u{2019}t on \(entry.hostName) any more. "
                    + "Everything else about this print is restored."
            }
        }
        destination = .generate
        // AFTER the adopt, which clamps, parks and echoes the pipeline: this
        // is the draft the pane will show, and the authority is good only
        // while the draft still IS it.
        reuseStore.arm(generate.draft)
        // ALWAYS ask, on every machine that lists this print. The server is
        // the only authority on what it retained -- inline source video,
        // audio and mask bytes leave no marker in the metadata at all -- and
        // mirroring an output does not copy the producing machine's private
        // archive, so one copy's blank says nothing about another's.
        let copies = library.items
            .filter { $0.print.filename == entry.print.filename }
            .map(\.id)
        let ordered = [entry.id] + copies.filter { $0 != entry.id }
        Task {
            await reuseStore.probe(ordered, fence: fence, disclosing: metadata)
            // Then the picture itself, into the well, so the person can see
            // what the render starts from and set its strength.
            let outgoing = hosts.host(entry.hostID).flatMap {
                RetainedSourcePicture.outgoing(generate, on: $0, hosts: hosts)
            }
            if let placed = await reuseStore.placePicture(
                in: generate.draft, outgoing: outgoing, live: { generate.draft }) {
                generate.draft = placed
            }
        }
    }

    /// The three things the Library menu offers about the SHELF it is
    /// showing. The sidebar's right-click menu answers the same three its own
    /// way -- it owns a row, this owns the pane -- but both are offered the
    /// same items, from `LibraryMenuPlan`.
    func performCollection(_ action: LibraryAction) {
        guard let shelf = enclosingShelf else { return }
        switch action {
        case .renameCollection:
            renamingShelf = shelf
        case let .setCollectionHidden(hidden):
            Task { await library.setShelfHidden(shelf, hidden: hidden) }
        case .deleteCollection:
            actions.confirmDestruction?(Destruction(
                title: "Delete “\(shelf.name)”?",
                message: "The prints in it are kept. Only the collection goes.",
                verb: "Delete Collection"
            ) { Task { await library.deleteShelf(shelf) } })
        default:
            break
        }
    }
}
