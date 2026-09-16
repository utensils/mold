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

    /// Applies a planned edit locally, registers its inverse, then tells the
    /// machines.
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
    /// feels broken on a remote machine. A failure puts the whole snapshot
    /// back rather than leaving the screen and the host quietly disagreeing.
    func apply(_ edit: PrintEdit, backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        guard !edit.isEmpty else { return }
        let previous = perHost
        mutate(edit)
        undo.register(edit) { [weak self] inverse in
            self?.apply(inverse, backend: backend)
        }
        Task { await push(edit, previous: previous, backend: backend) }
    }

    /// The half that talks to the machines.
    private func push(_ edit: PrintEdit, previous: [MoldHost.ID: [LibraryEntry]],
                      backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, filenames) in edit.targets {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            do { try await client.mutate(mutation(edit.change, filenames)) } catch {
                perHost = previous
                rebuild()
                failures[hostID] = edit.change.failureSentence
                // The stack now describes changes that never happened.
                undo.forget()
                return
            }
        }
        if case .collection = edit.change { await reloadCollections(backend) }
    }

    /// The wire form of a change, over one machine's filenames.
    private func mutation(_ change: PrintChange, _ filenames: [String]) -> GalleryBulkMutation {
        switch change {
        case let .favorite(on):
            GalleryBulkMutation(filenames: filenames, favorite: on)
        case let .tag(name, adding):
            GalleryBulkMutation(filenames: filenames,
                                addTags: adding ? [name] : [],
                                removeTags: adding ? [] : [name])
        case let .collection(name, slug, filing):
            GalleryBulkMutation(filenames: filenames,
                                addToCollection: filing ? .named(name) : nil,
                                removeFromCollectionSlug: filing ? nil : slug)
        }
    }

    /// The same change, applied to the rows on screen.
    private func mutate(_ edit: PrintEdit) {
        for (hostID, filenames) in edit.targets {
            let names = Set(filenames)
            perHost[hostID] = (perHost[hostID] ?? []).map { entry in
                guard names.contains(entry.print.filename) else { return entry }
                var mutable = GalleryPrint.Mutable(entry.print)
                edit.change.applied(to: &mutable, collectionID: collectionID(edit.change, hostID))
                return LibraryEntry(hostID: entry.hostID, hostName: entry.hostName,
                                    print: mutable.build())
            }
        }
        rebuild()
    }

    /// This machine's own id for the shelf a change is about, if it has one.
    func collectionID(_ change: PrintChange, _ hostID: MoldHost.ID) -> String? {
        guard case let .collection(_, slug, _) = change else { return nil }
        return collectionsPerHost[hostID]?.first { $0.slug == slug }?.id
    }

    func reloadCollections(_ backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        etags.removeAll()
        for hostID in collectionsPerHost.keys {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            if let collections = try? await client.collections() {
                collectionsPerHost[hostID] = collections
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
        }
    }
}
