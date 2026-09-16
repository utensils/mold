import Foundation
import Testing

@testable import MoldClient

// A collection is host-local -- its id is a UUID in one machine's database --
// but a person filing pictures under "Smurf Village" means one shelf whatever
// machine rendered them. mold's own rule is that collections merge by SLUG,
// and `mold_core` is the authority for the slug. Getting this wrong splits one
// shelf in two and the person cannot tell why.

private let plato = UUID()
private let hal = UUID()

private func collection(_ name: String, slug: String, count: Int,
                        hidden: Bool = false, id: String = UUID().uuidString) -> Collection {
    Collection(id: id, name: name, slug: slug, count: count, hidden: hidden)
}

@Test func oneSlugOnTwoMachinesIsOneShelf() {
    let shelves = CollectionShelf.merge([
        plato: [collection("Smurf Village", slug: "smurf-village", count: 9)],
        hal: [collection("Smurf Village", slug: "smurf-village", count: 4)],
    ])
    #expect(shelves.count == 1)
    #expect(shelves[0].slug == "smurf-village")
    #expect(shelves[0].count == 13)
    #expect(shelves[0].hosts.count == 2)
}

/// Two machines can spell one slug differently. The spelling shown is the one
/// on the machine holding the most of it -- the copy you have actually been
/// using -- and ties break alphabetically so the sidebar never reorders itself
/// between launches.
@Test func theSpellingShownComesFromTheMachineHoldingTheMost() {
    let shelves = CollectionShelf.merge([
        plato: [collection("smurf village", slug: "smurf-village", count: 2)],
        hal: [collection("Smurf Village", slug: "smurf-village", count: 40)],
    ])
    #expect(shelves[0].name == "Smurf Village")

    let tied = CollectionShelf.merge([
        plato: [collection("Zebra", slug: "z", count: 5)],
        hal: [collection("Aardvark", slug: "z", count: 5)],
    ])
    #expect(tied[0].name == "Aardvark")
}

/// Hidden means omitted from the default grid. One machine still showing it
/// means those prints belong in All Prints, so the shelf is only hidden when
/// every machine that has it hides it.
@Test func aShelfIsHiddenOnlyWhenEveryMachineHidesIt() {
    let partly = CollectionShelf.merge([
        plato: [collection("Drafts", slug: "drafts", count: 3, hidden: true)],
        hal: [collection("Drafts", slug: "drafts", count: 1, hidden: false)],
    ])
    #expect(partly[0].hidden == false)

    let fully = CollectionShelf.merge([
        plato: [collection("Drafts", slug: "drafts", count: 3, hidden: true)],
        hal: [collection("Drafts", slug: "drafts", count: 1, hidden: true)],
    ])
    #expect(fully[0].hidden)
}

@Test func aShelfOnOneMachineIsStillAShelf() {
    let shelves = CollectionShelf.merge([
        plato: [collection("Hangar", slug: "hangar", count: 31)],
        hal: [],
    ])
    #expect(shelves.count == 1)
    #expect(shelves[0].count == 31)
    #expect(shelves[0].hosts[hal] == nil)
}

/// Sorted for a person reading a sidebar, not for a byte comparator.
@Test func shelvesAreSortedTheWayAPersonReadsThem() {
    let shelves = CollectionShelf.merge([
        plato: [
            collection("zebra", slug: "zebra", count: 1),
            collection("Éclair", slug: "eclair", count: 1),
            collection("apple", slug: "apple", count: 1),
        ]
    ])
    #expect(shelves.map(\.name) == ["apple", "Éclair", "zebra"])
}

/// Each machine keeps its own id, because removing a print from a shelf and
/// opening one need the id on the machine that holds the print.
@Test func eachMachineKeepsItsOwnIdForTheThingsThatNeedOne() {
    let shelves = CollectionShelf.merge([
        plato: [collection("Hangar", slug: "hangar", count: 3, id: "p-1")],
        hal: [collection("Hangar", slug: "hangar", count: 1, id: "h-9")],
    ])
    #expect(shelves[0].hosts[plato] == "p-1")
    #expect(shelves[0].hosts[hal] == "h-9")
}

/// Filing sends a NAME, never an id: the host resolves it by slug and creates
/// it if it has never seen it. An id would only ever be right on one machine,
/// which is how one shelf becomes two.
@Test func filingSendsANameSoEveryMachineResolvesItsOwn() throws {
    let mutation = GalleryBulkMutation(
        filenames: ["a.png"],
        addToCollection: .named("Smurf Village")
    )
    let json = String(decoding: try MoldJSON.encoder.encode(mutation), as: UTF8.self)
    #expect(json.contains("\"add_to_collection\""))
    #expect(json.contains("\"name\":\"Smurf Village\""))
    #expect(!json.contains("\"id\""))
}

/// Removal is by slug for the same reason, and it is a different field.
@Test func removalNamesTheSlug() throws {
    let mutation = GalleryBulkMutation(filenames: ["a.png"], removeFromCollectionSlug: "hangar")
    let json = String(decoding: try MoldJSON.encoder.encode(mutation), as: UTF8.self)
    #expect(json.contains("\"remove_from_collection_slug\":\"hangar\""))
}

/// An operation id is the replay fence. It must be minted once and reused by
/// a retry, never regenerated -- a fresh id on every attempt is exactly the
/// double-apply the fence exists to prevent.
@Test func anOperationKeepsItsIdAcrossRetries() {
    let mutation = GalleryBulkMutation(filenames: ["a.png"], favorite: true)
    #expect(mutation.operationId == mutation.operationId)
    #expect(GalleryBulkMutation(filenames: ["a.png"], favorite: true).operationId
            != mutation.operationId)
}

/// On a PATCH, an absent field means "leave it alone" and a null would mean
/// "clear it". Renaming a collection must not wipe its cover picture, so the
/// body has to OMIT what the caller did not set.
@Test func changingOneThingAboutACollectionLeavesTheRestAlone() throws {
    let rename = CollectionChange(name: "Hangar Two")
    let json = String(decoding: try MoldJSON.encoder.encode(rename), as: UTF8.self)
    #expect(json.contains("\"name\":\"Hangar Two\""))
    #expect(!json.contains("cover_filename"))
    #expect(!json.contains("hidden"))
    #expect(!json.contains("null"))
}

/// A tag is whatever a person typed: `#blue` is the literal tag `#blue`, and a
/// tag can hold a space or a slash. Putting one in a path without encoding it
/// renames the wrong tag, or a route that does not exist.
@Test(arguments: ["#blue", "half/half", "two words", "café"])
func aTagGoesIntoAPathAsItself(tag: String) {
    let backend = HTTPBackend(host: MoldHost(name: "t", baseURL: URL(string: "http://t:7680")!))
    let escaped = backend.escaped(tag)
    #expect(!escaped.contains("/"))
    #expect(!escaped.contains("#"))
    #expect(!escaped.contains(" "))
    #expect(escaped.removingPercentEncoding == tag)
}
