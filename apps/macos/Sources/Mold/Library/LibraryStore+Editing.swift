import Foundation
import MoldClient

// The one funnel every reversible library edit goes through.
//
// One function, so that undo, the optimistic local apply and the wire request
// can never disagree about what a change was. Registration lives HERE and not
// at the call sites because the store is the one thing that still holds the
// previous value -- a menu item knows what you asked for, not what it altered.
@MainActor
extension LibraryStore {

    /// Applies a planned edit locally, registers its inverse, then queues it
    /// for the machines.
    ///
    /// **Synchronous on purpose, and this is load-bearing.** `UndoManager`
    /// only routes a registration to the REDO stack while it is inside
    /// `undo()`, and that window closes the moment the callback returns. An
    /// `async` funnel registers from a `Task` that runs after it has shut, so
    /// the re-registration lands back on the undo stack: ⌘Z toggles the same
    /// change forever and Redo never arms. So the parts that must happen
    /// inside the window -- the local mutation and the registration -- run
    /// here, and only the network call is deferred.
    ///
    /// Local first for its own reason: a star that waits for a round trip
    /// feels broken on a remote machine. What happens when the machine never
    /// agrees is the outbox's problem, not this function's -- see
    /// `LibraryStore+Outbox`.
    func apply(_ edit: PrintEdit) {
        guard !edit.isEmpty else { return }
        mutate(edit)
        undo.register(edit) { [weak self] inverse in
            self?.apply(inverse)
        }
        send(edit)
    }

    /// The same change, applied to the rows on screen.
    func mutate(_ edit: PrintEdit) {
        for (hostID, filenames) in edit.targets {
            let names = Set(filenames)
            perHost[hostID] = (perHost[hostID] ?? []).map { entry in
                guard names.contains(entry.print.filename) else { return entry }
                var mutable = GalleryPrint.Mutable(entry.print)
                edit.change.applied(to: &mutable, collectionID: collectionID(edit.change, hostID))
                return entry.replacingPrint(mutable.build())
            }
        }
        rebuild()
    }

    /// This machine's own id for the shelf a change is about, if it has one.
    func collectionID(_ change: PrintChange, _ hostID: MoldHost.ID) -> String? {
        guard case let .collection(_, slug, _) = change else { return nil }
        return collectionsPerHost[hostID]?.first { $0.slug == slug }?.id
    }

    func reloadCollections() async {
        etags.removeAll()
        for host in hosts.hosts {
            if let collections = try? await hosts.backend(for: host).collections() {
                collectionsPerHost[host.id] = collections
            }
        }
    }
}

extension PrintChange {
    /// What the machine could not do, in the terms the person used.
    var failureSentence: String {
        switch self {
        case .favorite: "Couldn't update those prints."
        case .tag: "Couldn't change those tags."
        case let .collection(name, _, filing):
            filing ? "Couldn't file those into \(name)." : "Couldn't take those out of \(name)."
        case .title: "Couldn't rename that print."
        }
    }

    /// The change, applied to one print on screen.
    ///
    /// `collectionID` is that machine's id for the shelf. Filing onto a
    /// machine that has never seen it leaves the row's membership alone --
    /// the host mints the id, and guessing one here would put a stranger in
    /// the list until the next refresh corrected it.
    func applied(to print: inout GalleryPrint.Mutable, collectionID: String?) {
        switch self {
        case let .favorite(on):
            print.favorite = on
        case let .tag(name, adding):
            var tags = print.tags ?? []
            tags.removeAll { $0.caseInsensitiveCompare(name) == .orderedSame }
            if adding { tags.append(name) }
            print.tags = tags
        case let .collection(_, _, filing):
            guard let collectionID else { return }
            var members = print.collections ?? []
            members.removeAll { $0 == collectionID }
            if filing { members.append(collectionID) }
            print.collections = members
        case let .title(_, to):
            print.title = to.isEmpty ? nil : to
        }
    }
}
