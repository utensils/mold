import Foundation
import Testing

@testable import MoldClient

/// A notification opens its print. All Prints leaves out members of hidden
/// collections -- that is what hidden means -- so a print filed into one
/// opened onto a grid without it. The shelf that can show it is chosen
/// instead: its hidden collection.
struct LibraryRevealScopeTests {
    private let workstation = UUID()
    private let hal = UUID()

    private func shelf(_ name: String, hidden: Bool, ids: [UUID: String]) -> CollectionShelf {
        CollectionShelf(slug: name.lowercased(), name: name, count: 1, hidden: hidden, hosts: ids)
    }

    @Test func anOrdinaryPrintOpensInAllPrints() {
        let print = PrintFixtures.entry("a.png", host: workstation, collections: ["c-open"])
        let shelves = [shelf("Open", hidden: false, ids: [workstation: "c-open"])]
        #expect(LibraryScope.revealing(print, shelves: shelves) == .all)
        #expect(LibraryScope.revealing(PrintFixtures.entry("b.png", host: workstation), shelves: shelves) == .all)
    }

    @Test func aMemberOfAHiddenCollectionOpensInThatCollection() {
        let print = PrintFixtures.entry("a.png", host: workstation, collections: ["c-open", "c-nsfw"])
        let shelves = [shelf("Open", hidden: false, ids: [workstation: "c-open"]),
                       shelf("NSFW", hidden: true, ids: [workstation: "c-nsfw"])]
        #expect(LibraryScope.revealing(print, shelves: shelves) == .collection(slug: "nsfw"))
    }

    /// Membership is per machine: another machine's id for the same shelf
    /// never matches, and a merged print is hidden when ANY copy is.
    @Test func membershipIsReadPerCopyOnItsOwnMachine() {
        var print = PrintFixtures.entry("a.png", host: workstation)
        print.copies = [PrintFixtures.entry("a.png", host: hal, collections: ["h-nsfw"])]
        let shelves = [shelf("NSFW", hidden: true, ids: [workstation: "w-nsfw", hal: "h-nsfw"])]
        #expect(LibraryScope.revealing(print, shelves: shelves) == .collection(slug: "nsfw"))
        let elsewhere = PrintFixtures.entry("b.png", host: workstation, collections: ["h-nsfw"])
        #expect(LibraryScope.revealing(elsewhere, shelves: shelves) == .all)
    }

    @Test func severalHiddenCollectionsChooseOneByName() {
        let print = PrintFixtures.entry("a.png", host: workstation, collections: ["z", "a"])
        let shelves = [shelf("Zed", hidden: true, ids: [workstation: "z"]),
                       shelf("Alpha", hidden: true, ids: [workstation: "a"])]
        #expect(LibraryScope.revealing(print, shelves: shelves) == .collection(slug: "alpha"))
    }
}

/// Opening a print from outside the Library may need a different shelf, and
/// changing shelf clears the selection. Only the target shelf's own settled
/// change may select the print; every other pass waits.
struct LibraryRevealSequenceTests {
    private let print = PrintID(host: UUID(), filename: "new.png")
    private let nsfw = LibraryScope.collection(slug: "nsfw")

    @Test func alreadyOnTheRightShelfSelectsAtOnce() {
        var reveal = LibraryRevealSequence(print)
        #expect(reveal.step(current: .all, target: .all, scopeSettled: false) == .select(print))
    }

    /// The case review caught: the Library already open on Favourites. The
    /// reveal's own observer must not select before the hidden collection's
    /// scope change has cleared the selection.
    @Test func aShelfChangeWaitsForThatShelfToSettle() {
        var reveal = LibraryRevealSequence(print)
        #expect(reveal.step(current: .favorites, target: nsfw, scopeSettled: false) == .changeScope(nsfw))
        #expect(reveal.step(current: nsfw, target: nsfw, scopeSettled: false) == .wait,
                "the reveal observer runs again before the scope observer clears")
        #expect(reveal.step(current: .favorites, target: nsfw, scopeSettled: true) == .wait,
                "an earlier shelf settling is not the target")
        #expect(reveal.step(current: nsfw, target: nsfw, scopeSettled: true) == .select(print))
    }

    @Test func anOrdinaryPrintFromAnotherShelfGoesToAllPrintsTheSameWay() {
        var reveal = LibraryRevealSequence(print)
        #expect(reveal.step(current: .favorites, target: .all, scopeSettled: false) == .changeScope(.all))
        #expect(reveal.step(current: .all, target: .all, scopeSettled: true) == .select(print))
    }
}
