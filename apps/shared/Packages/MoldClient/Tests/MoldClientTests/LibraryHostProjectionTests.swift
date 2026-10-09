import Foundation
import Testing
@testable import MoldClient

@Test func hiddenShelvesMapEveryReplicaAndProtectUnfiledCopies() {
    let local = UUID(), remote = UUID()
    let shelves = CollectionShelf.merge([
        local: [Collection(id: "l", name: "Hidden", slug: "hidden", hidden: true)],
        remote: [Collection(id: "r", name: "Hidden", slug: "hidden", hidden: false)],
    ])
    #expect(CollectionShelf.hiddenIDs(in: shelves) == [local: ["l"], remote: ["r"]])
    var entry = PrintFixtures.entry("x.png", host: local, collections: ["l"])
    entry.copies = [PrintFixtures.entry("x.png", host: remote)]
    var query = LibraryQuery()
    query.hiddenCollectionIDs = CollectionShelf.hiddenIDs(in: shelves)
    query.tokens = [.machine(id: remote, name: "Remote")]
    #expect(query.apply(to: [entry]).isEmpty)
    query.tokens.append(.collection(slug: "hidden", name: "Hidden", ids: shelves[0].hosts))
    #expect(query.apply(to: [entry]).isEmpty) // remote has no membership
}

@Test func hostProjectionCountsOnlyLocalMembershipAndDistinguishesAbsence() {
    let local = UUID(), remote = UUID(), missing = UUID()
    let shelf = CollectionShelf.merge([
        local: [Collection(id: "l", name: "Shelf", slug: "shelf")],
        remote: [Collection(id: "r", name: "Shelf", slug: "shelf")],
    ])[0]
    var entry = PrintFixtures.entry("x.png", host: local, collections: ["l"])
    entry.copies = [PrintFixtures.entry("x.png", host: remote)]
    #expect(shelf.count(in: [entry], on: [remote]) == 0)
    #expect(shelf.count(in: [entry], on: [local]) == 1)
    #expect(shelf.presence(on: [missing], available: [missing]) == .absent)
    #expect(shelf.presence(on: [missing], available: []) == .unavailable)
    #expect(shelf.presence(on: [remote], available: [remote]) == .present)
    let projected = entry.presented(onAnyOf: [remote])!
    #expect(projected.everyCopy.map(\.hostID) == [remote])
    #expect(Set(PrintEdit.plan(.favorite(true), over: projected.everyCopy).targets.keys) == [remote])
}

@Test func mergedFavoritesAndTagsIncludeOtherCopies() {
    var entry = PrintFixtures.entry("x.png", host: UUID())
    entry.copies = [PrintFixtures.entry("x.png", host: UUID(), tags: ["remote"], favorite: true)]
    #expect(entry.isFavorite)
    #expect(entry.tags == ["remote"])
    #expect(entry.presented(onAnyOf: [entry.hostID])?.isFavorite == false)
}

@Test func scopeBaselineCountExcludesOtherHostsAndUnrelatedCollections() {
    let one = UUID(), two = UUID()
    let shelf = CollectionShelf.merge([one: [Collection(id: "c", name: "C", slug: "c")]])
    let entries = [PrintFixtures.entry("filed.png", host: one, collections: ["c"]),
                   PrintFixtures.entry("other.png", host: one),
                   PrintFixtures.entry("remote.png", host: two)]
    #expect(LibraryScope.collection(slug: "c").baselineCount(in: entries, machines: [one], shelves: shelf, hiddenIDs: [:]) == 1)
    #expect(LibraryScope.all.baselineCount(in: entries, machines: [one], shelves: shelf, hiddenIDs: [one: ["c"]]) == 1)
    #expect(LibraryScope.trash.baselineCount(in: entries, machines: [one], shelves: shelf, hiddenIDs: [one: ["c"]]) == 2)
}
