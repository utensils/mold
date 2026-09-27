import Foundation
import Testing

@testable import MoldClient

// Filtering happens on the client because `GET /api/gallery` has no
// pagination: each host answers with its whole index and the app holds it. So
// this is the only place the question "which prints am I looking at" is
// answered, and it runs on every keystroke over thousands of rows.

private let workstation = UUID()
private let hal = UUID()

private func entry(_ name: String, host: UUID = workstation, hostName: String = "workstation",
                   at seconds: UInt64 = 1_000, tags: [String] = [], favorite: Bool = false,
                   format: String = "png", collections: [String] = [],
                   prompt: String = "a tin robot", bytes: Int = 100) -> LibraryEntry {
    PrintFixtures.entry(name, host: host, hostName: hostName, timestamp: seconds,
                        format: format, tags: tags, favorite: favorite,
                        collections: collections, prompt: prompt, bytes: bytes)
}

private let library = [
    entry("robot.png", at: 300, tags: ["owls", "draft"], prompt: "a tin robot on a wet street"),
    entry("turtle.png", at: 200, tags: ["owls"], favorite: true, prompt: "a turtle"),
    entry("clip.mp4", host: hal, hostName: "hal9000", at: 100, format: "mp4", prompt: "a hangar"),
    entry("chair.glb", host: hal, hostName: "hal9000", at: 50, format: "glb", prompt: "a chair"),
]

@Test func noQueryIsEveryPrintNewestFirst() {
    let shown = LibraryQuery().apply(to: library)
    #expect(shown.map(\.print.filename) == ["robot.png", "turtle.png", "clip.mp4", "chair.glb"])
}

/// Tokens of DIFFERENT kinds narrow: a machine and a kind together mean both.
@Test func differentKindsOfTokenNarrow() {
    var query = LibraryQuery()
    query.tokens = [.machine(id: hal, name: "hal9000"), .kind(.clip)]
    #expect(query.apply(to: library).map(\.print.filename) == ["clip.mp4"])
}

/// Tokens of the SAME kind widen, because a print cannot be on two machines at
/// once — ANDing them would always give nothing, which is never what someone
/// who added a second machine meant.
@Test func tokensOfOneKindWiden() {
    var query = LibraryQuery()
    query.tokens = [.kind(.clip), .kind(.mesh)]
    #expect(query.apply(to: library).map(\.print.filename) == ["clip.mp4", "chair.glb"])

    query.tokens = [.machine(id: workstation, name: "workstation"), .machine(id: hal, name: "hal9000")]
    #expect(query.apply(to: library).count == 4)
}

/// Two tags DO narrow, because one print can carry both.
@Test func twoTagsNarrowBecauseOnePrintCanCarryBoth() {
    var query = LibraryQuery()
    query.tokens = [.tag("owls"), .tag("draft")]
    #expect(query.apply(to: library).map(\.print.filename) == ["robot.png"])
}

@Test func freeTextNarrowsAlongsideTokens() {
    var query = LibraryQuery()
    query.tokens = [.tag("owls")]
    query.text = "turtle"
    #expect(query.apply(to: library).map(\.print.filename) == ["turtle.png"])
}

@Test func favouritesAreATokenLikeAnyOther() {
    var query = LibraryQuery()
    query.tokens = [.favorite]
    #expect(query.apply(to: library).map(\.print.filename) == ["turtle.png"])
}

/// A row's `collections` are the HOST's own ids, so a shelf that spans two
/// machines has a different id on each. Filtering with the wrong machine's id
/// silently shows nothing, which reads as an empty collection.
@Test func aShelfIsFilteredByTheIdOnEachPrintsOwnMachine() {
    let mine = [
        entry("a.png", host: workstation, at: 3, collections: ["p-1"]),
        entry("b.png", host: hal, hostName: "hal9000", at: 2, collections: ["h-9"]),
        entry("c.png", host: hal, hostName: "hal9000", at: 1, collections: ["h-other"]),
    ]
    var query = LibraryQuery()
    query.tokens = [.collection(slug: "hangar", name: "Hangar",
                                ids: [workstation: "p-1", hal: "h-9"])]
    #expect(query.apply(to: mine).map(\.print.filename) == ["a.png", "b.png"])
}

/// Hidden collections are kept out of the default grid — that is what hidden
/// means — but asking for one by name still shows it.
@Test func hiddenShelvesAreOutOfTheWayNotUnreachable() {
    let mine = [
        entry("a.png", at: 3, collections: ["secret"]),
        entry("b.png", at: 2),
    ]
    var query = LibraryQuery()
    query.hiddenCollectionIDs = [workstation: ["secret"]]
    #expect(query.apply(to: mine).map(\.print.filename) == ["b.png"])

    query.tokens = [.collection(slug: "s", name: "Secret", ids: [workstation: "secret"])]
    #expect(query.apply(to: mine).map(\.print.filename) == ["a.png"])
}

@Test(arguments: [
    (LibrarySort.newest, ["robot.png", "turtle.png", "clip.mp4", "chair.glb"]),
    (LibrarySort.oldest, ["chair.glb", "clip.mp4", "turtle.png", "robot.png"]),
])
func sortingRunsBothWays(sort: LibrarySort, expected: [String]) {
    var query = LibraryQuery()
    query.sort = sort
    #expect(query.apply(to: library).map(\.print.filename) == expected)
}

/// Two prints made in the same second are common — a batch writes them
/// together — so the order has to be settled by something else or the grid
/// reshuffles itself between refreshes.
@Test func printsMadeInTheSameSecondStillHaveAStableOrder() {
    let tied = [
        entry("b.png", at: 500), entry("a.png", at: 500), entry("c.png", at: 500),
    ]
    let once = LibraryQuery().apply(to: tied).map(\.print.filename)
    let twice = LibraryQuery().apply(to: tied.reversed()).map(\.print.filename)
    #expect(once == twice)
}

/// Case, accents and width are all ignored: "cafe" finds "Café".
@Test(arguments: ["café", "CAFE", "ｃａｆｅ"])
func searchIsForgivingAboutHowItIsTyped(typed: String) {
    let mine = [entry("x.png", prompt: "a café at night")]
    var query = LibraryQuery()
    query.text = typed
    #expect(query.apply(to: mine).count == 1)
}
