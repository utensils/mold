import MoldClient

// Naming, tagging and filing. Split from the rest of `LibraryActions` for size.
//
// Each of these is one line because the store is where the thinking is: what a
// change actually alters, what reverses it, and what to do when a machine says
// no. These are the doors, not the rooms.
extension LibraryActions {

    func setTitle(_ title: String, on entry: LibraryEntry) {
        library.setTitle(title, on: entry, backend: backend)
    }

    func file(_ entries: [LibraryEntry], into shelf: CollectionShelf) {
        library.file(entries, into: shelf, backend: backend)
    }

    func unfile(_ entries: [LibraryEntry], from shelf: CollectionShelf) {
        library.unfile(entries, from: shelf, backend: backend)
    }

    func renameTag(_ name: String, to newName: String) {
        library.renameTag(name, to: newName, backend: backend)
    }

    func deleteTag(_ name: String) {
        confirmDestruction?(Destruction(
            title: "Delete the tag \u{201C}\(name)\u{201D}?",
            message: "It comes off every print on every machine. The prints are kept.",
            verb: "Delete Tag",
            perform: { library.deleteTag(name, backend: backend) }
        ))
    }
}
