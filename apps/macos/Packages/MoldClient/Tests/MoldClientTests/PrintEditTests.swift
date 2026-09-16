import Foundation
import Testing

@testable import MoldClient

/// What an edit would actually change, and what puts it back.
///
/// The rule under test is "only what actually changed": an undo that reverses
/// prints the change never touched is worse than no undo at all, because it
/// silently edits rows the person never acted on.
@Suite struct PrintEditSuite {
    let plato = UUID()
    let hal = UUID()

    // MARK: - Favourite

    @Test func favoritingSkipsThePrintsAlreadyFavorite() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, favorite: true),
            PrintFixtures.entry("b.png", host: plato, favorite: false),
            PrintFixtures.entry("c.png", host: plato),
        ]
        let edit = PrintEdit.plan(.favorite(true), over: entries)
        #expect(edit.targets[plato] == ["b.png", "c.png"])
    }

    @Test func favoritingNothingNewIsNoEditAtAll() {
        let entries = [PrintFixtures.entry("a.png", host: plato, favorite: true)]
        #expect(PrintEdit.plan(.favorite(true), over: entries).isEmpty)
    }

    @Test func theInverseOfFavoriteIsUnfavoriteOverTheSameTargets() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, favorite: true),
            PrintFixtures.entry("b.png", host: plato),
        ]
        let edit = PrintEdit.plan(.favorite(true), over: entries)
        #expect(edit.inverse.change == .favorite(false))
        // "b.png" only -- "a.png" was already a favourite, so putting it back
        // must not un-favourite something the action never touched.
        #expect(edit.inverse.targets[plato] == ["b.png"])
    }

    // MARK: - Tags

    @Test func taggingIsCaseInsensitiveAboutWhatIsAlreadyThere() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, tags: ["Owls"]),
            PrintFixtures.entry("b.png", host: plato, tags: ["barns"]),
        ]
        let edit = PrintEdit.plan(.tag("owls", adding: true), over: entries)
        #expect(edit.targets[plato] == ["b.png"])
        #expect(edit.inverse.change == .tag("owls", adding: false))
    }

    @Test func untaggingOnlyNamesThePrintsThatCarryIt() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, tags: ["owls"]),
            PrintFixtures.entry("b.png", host: plato, tags: []),
        ]
        #expect(PrintEdit.plan(.tag("owls", adding: false), over: entries).targets[plato]
            == ["a.png"])
    }

    // MARK: - Collections

    @Test func filingSkipsThePrintsAlreadyOnTheShelf() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, collections: ["col-7"]),
            PrintFixtures.entry("b.png", host: plato, collections: []),
        ]
        let edit = PrintEdit.plan(.collection(name: "Hangar", slug: "hangar", filing: true),
                                  over: entries, collectionIDs: [plato: "col-7"])
        #expect(edit.targets[plato] == ["b.png"])
        #expect(edit.inverse.change
            == .collection(name: "Hangar", slug: "hangar", filing: false))
    }

    /// A machine that has never seen the shelf will create it, so every print
    /// there is affected -- the absence of an id is not the absence of a change.
    @Test func filingOntoAMachineWithNoSuchShelfAffectsEveryPrint() {
        let entries = [
            PrintFixtures.entry("a.png", host: hal, hostName: "hal9000", collections: ["x"]),
        ]
        let edit = PrintEdit.plan(.collection(name: "Hangar", slug: "hangar", filing: true),
                                  over: entries, collectionIDs: [:])
        #expect(edit.targets[hal] == ["a.png"])
    }

    /// The mirror image: you cannot take a print off a shelf that machine has
    /// never heard of.
    @Test func unfilingFromAMachineWithNoSuchShelfChangesNothing() {
        let entries = [PrintFixtures.entry("a.png", host: hal, collections: ["x"])]
        let edit = PrintEdit.plan(.collection(name: "Hangar", slug: "hangar", filing: false),
                                  over: entries, collectionIDs: [:])
        #expect(edit.isEmpty)
    }

    // MARK: - Titles

    /// A title is the one change whose inverse cannot be derived from the
    /// change alone -- "call it Helmet" reverses to "call it what it was", and
    /// only the caller knows that. So the change carries both ends.
    @Test func aTitleChangeCarriesWhatItWasSoItCanBePutBack() {
        let entries = [PrintFixtures.entry("a.png", host: plato, title: "Old")]
        let edit = PrintEdit.plan(.title(from: "Old", to: "New"), over: entries)
        #expect(edit.targets[plato] == ["a.png"])
        #expect(edit.inverse.change == .title(from: "New", to: "Old"))
    }

    @Test func renamingAPrintToWhatItIsCalledIsNoEdit() {
        let entries = [PrintFixtures.entry("a.png", host: plato, title: "Helmet")]
        #expect(PrintEdit.plan(.title(from: "Helmet", to: "Helmet"), over: entries).isEmpty)
    }

    /// An untitled print has no title, not an empty one -- so clearing a title
    /// that was never set changes nothing.
    @Test func clearingATitleThatWasNeverSetIsNoEdit() {
        let entries = [PrintFixtures.entry("a.png", host: plato)]
        #expect(PrintEdit.plan(.title(from: "", to: ""), over: entries).isEmpty)
    }

    @Test func clearingARealTitleIsAnEdit() {
        let entries = [PrintFixtures.entry("a.png", host: plato, title: "Helmet")]
        #expect(PrintEdit.plan(.title(from: "Helmet", to: ""), over: entries).targets[plato]
            == ["a.png"])
    }

    // MARK: - Fleets

    @Test func targetsAreGroupedByMachine() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato),
            PrintFixtures.entry("b.png", host: hal, hostName: "hal9000"),
        ]
        let edit = PrintEdit.plan(.favorite(true), over: entries)
        #expect(edit.targets[plato] == ["a.png"])
        #expect(edit.targets[hal] == ["b.png"])
    }

    /// A machine with nothing to change must not appear at all, or the store
    /// sends it an empty mutation and the outbox retries an edit that is not one.
    @Test func aMachineWithNothingToChangeIsAbsent() {
        let entries = [
            PrintFixtures.entry("a.png", host: plato, favorite: true),
            PrintFixtures.entry("b.png", host: hal, hostName: "hal9000"),
        ]
        let edit = PrintEdit.plan(.favorite(true), over: entries)
        #expect(edit.targets[plato] == nil)
        #expect(edit.targets.count == 1)
    }

    // MARK: - Display

    @Test func aPrintIsCalledItsTitleIfItHasOneAndItsFilenameOtherwise() {
        #expect(PrintFixtures.print("a.png", title: "Owl at dusk").displayName == "Owl at dusk")
        #expect(PrintFixtures.print("a.png").displayName == "a.png")
        // A cleared field is not a name.
        #expect(PrintFixtures.print("a.png", title: "   ").displayName == "a.png")
    }

    // MARK: - Names

    @Test func everyChangeNamesItselfForTheEditMenu() {
        #expect(PrintChange.favorite(true).actionName == "Favorite")
        #expect(PrintChange.favorite(false).actionName == "Unfavorite")
        #expect(PrintChange.tag("owls", adding: true).actionName == "Tag")
        #expect(PrintChange.tag("owls", adding: false).actionName == "Remove Tag")
        #expect(PrintChange.collection(name: "Hangar", slug: "hangar", filing: true)
            .actionName == "Move to Hangar")
        #expect(PrintChange.collection(name: "Hangar", slug: "hangar", filing: false)
            .actionName == "Remove from Hangar")
        #expect(PrintChange.title(from: "Old", to: "New").actionName == "Rename")
        // Clearing is not renaming, and the Edit menu should say which.
        #expect(PrintChange.title(from: "Old", to: "").actionName == "Clear Title")
    }
}
