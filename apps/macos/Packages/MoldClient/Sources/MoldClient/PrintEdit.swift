import Foundation

/// A change to prints that can be put back.
///
/// Deliberately a value, and deliberately here rather than in the app: the
/// question "what would this change, and what reverses it" is arithmetic over
/// the prints, not a UI concern, and it is the one part of undo worth testing
/// exhaustively.
public enum PrintChange: Hashable, Sendable {
    case favorite(Bool)
    case tag(String, adding: Bool)
    /// Filing carries BOTH names because the two directions are addressed
    /// differently: a host is told a collection's `name` and resolves or
    /// creates it, while removal names the `slug` every machine agrees on.
    /// An inverse needs whichever one it is about to use.
    case collection(name: String, slug: String, filing: Bool)
    /// Renaming one print. Carries BOTH ends, because a title's inverse cannot
    /// be worked out from the new value -- only the caller ever knew the old
    /// one, and by the time undo runs the screen no longer does.
    case title(from: String, to: String)

    /// What the Edit menu says after "Undo". Sentence-cased for a menu item,
    /// and never containing the count -- macOS undo names the action, not the
    /// selection.
    public var actionName: String {
        switch self {
        case let .favorite(on): on ? "Favorite" : "Unfavorite"
        case let .tag(_, adding): adding ? "Tag" : "Remove Tag"
        case let .collection(name, _, filing):
            filing ? "Move to \(name)" : "Remove from \(name)"
        case let .title(_, to):
            to.isEmpty ? "Clear Title" : "Rename"
        }
    }

    var reversed: PrintChange {
        switch self {
        case let .favorite(on): .favorite(!on)
        case let .tag(name, adding): .tag(name, adding: !adding)
        case let .collection(name, slug, filing):
            .collection(name: name, slug: slug, filing: !filing)
        case let .title(from, to):
            .title(from: to, to: from)
        }
    }
}

/// A change, and exactly which prints on which machines it alters.
public struct PrintEdit: Hashable, Sendable {
    public let change: PrintChange
    /// Filenames per machine. A machine with nothing to change is ABSENT
    /// rather than present with an empty list -- an empty mutation is a
    /// request that means nothing, and the outbox would retry it forever.
    public let targets: [MoldHost.ID: [String]]

    public init(change: PrintChange, targets: [MoldHost.ID: [String]]) {
        self.change = change
        self.targets = targets.filter { !$0.value.isEmpty }
    }

    public var isEmpty: Bool { targets.isEmpty }
    public var actionName: String { change.actionName }

    /// What puts it back: the opposite change over the SAME prints.
    ///
    /// The same prints and not the whole selection, which is the entire point.
    /// Favouriting five prints of which two were already favourites, then
    /// undoing, must leave those two as it found them.
    public var inverse: PrintEdit {
        PrintEdit(change: change.reversed, targets: targets)
    }

    /// Narrows a change to the prints it would actually alter.
    ///
    /// `collectionIDs` maps each machine to ITS id for the shelf, because a
    /// print's `collections` are host-local ids. Absence is meaningful in both
    /// directions and differently: filing onto a machine that has never seen
    /// the shelf changes every print there (the host creates it), while
    /// unfiling from one changes nothing.
    public static func plan(
        _ change: PrintChange,
        over entries: [LibraryEntry],
        collectionIDs: [MoldHost.ID: String] = [:]
    ) -> PrintEdit {
        var targets: [MoldHost.ID: [String]] = [:]
        for entry in entries {
            guard alters(change, entry, collectionIDs[entry.hostID]) else { continue }
            targets[entry.hostID, default: []].append(entry.print.filename)
        }
        return PrintEdit(change: change, targets: targets)
    }

    private static func alters(_ change: PrintChange, _ entry: LibraryEntry,
                               _ collectionID: String?) -> Bool {
        let print = entry.print
        switch change {
        case let .favorite(on):
            return print.isFavorite != on
        case let .tag(name, adding):
            let carries = print.tagList.contains { $0.caseInsensitiveCompare(name) == .orderedSame }
            return carries != adding
        case let .collection(_, _, filing):
            guard let collectionID else { return filing }
            return (print.collections ?? []).contains(collectionID) != filing
        case let .title(_, to):
            // An untitled print has no title, not an empty one, so clearing
            // what was never set is not a change.
            return (print.title ?? "") != to
        }
    }
}
