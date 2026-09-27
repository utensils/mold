import Foundation

// Applying a `PrintChange` to one print's mutable form. Split from
// `PrintEdit.swift` for size -- that file plans a change, this one performs it.
public extension PrintChange {
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
