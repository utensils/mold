import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Library following each machine's `GalleryReplica`: unchanged machines
/// leave the timeline alone, and a machine with a change log is caught up by
/// cursor rather than re-listed or rebuilt per frame.
@MainActor
struct LibraryReplicaStoreTests {
    private func bench(feed: Bool) async -> (LibraryStore, FakeBackend, MoldHost, HostStore) {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("a.png"), FakeFixtures.print("b.png")]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        if feed { fake.serveFeed(on: hosts) }
        let library = LibraryStore(hosts: hosts)
        await library.reload()
        return (library, fake, machine, hosts)
    }

    @Test func anUnchangedMachineDoesNotTouchTheTimeline() async {
        for feed in [false, true] {
            let (library, _, _, _) = await bench(feed: feed)
            #expect(library.items.count == 2)
            let revision = library.rows.value
            await library.refresh()
            await library.reload()
            #expect(library.rows.value == revision, "feed: \(feed)")
        }
    }

    @Test func aFeedMachineIsFollowedByCursorNotListed() async {
        let (library, fake, _, _) = await bench(feed: true)
        fake.prints.append(FakeFixtures.print("c.png"))
        await library.refresh()
        #expect(Set(library.items.map(\.print.filename)) == ["a.png", "b.png", "c.png"])
        #expect(fake.callCount("gallery") == 0)
        #expect(fake.feed.requested.first == .some(nil))
        #expect(fake.feed.requested.last != .some(nil))
    }

    @Test func aBurstOfFramesIsOneCatchUpNotOneRebuildEach() async {
        let (library, fake, machine, hosts) = await bench(feed: true)
        let before = fake.callCount("galleryChanges")
        fake.prints.append(FakeFixtures.print("c.png"))
        for _ in 0..<20 {
            hosts.listeners.forEach {
                $0(machine.id, .gallery(.added(filename: "c.png", row: FakeFixtures.print("c.png"))))
            }
        }
        await settle { library.items.contains { $0.print.filename == "c.png" } }
        #expect(library.items.contains { $0.print.filename == "c.png" })
        #expect(fake.callCount("galleryChanges") - before <= 2)
        #expect(fake.callCount("gallery") == 0)
    }

    @Test func aResyncOnAFeedMachineResumesFromItsCursor() async {
        let (library, fake, machine, hosts) = await bench(feed: true)
        fake.prints.removeAll { $0.filename == "b.png" }
        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        await settle { library.items.count == 1 }
        #expect(library.items.map(\.print.filename) == ["a.png"])
        #expect(fake.callCount("gallery") == 0)
        #expect(library.live.relists.reads[machine.id] == nil)
    }

    @Test func aTrashedRowMovesBetweenTimelinesFromOneDelta() async {
        let (library, fake, _, _) = await bench(feed: true)
        let row = fake.prints.removeFirst()
        fake.trashedRows = [FakeFixtures.trashState(row, at: 5)]
        await library.refresh()
        #expect(library.items.map(\.print.filename) == ["b.png"])
        #expect(library.trashed.map(\.print.filename) == ["a.png"])
    }

    @Test func anEditStillOnItsWaySurvivesADelta() async {
        let (library, fake, machine, _) = await bench(feed: true)
        library.mutations.outbox.enqueue(PrintEdit(change: .favorite(true), targets: [machine.id: ["a.png"]]))
        library.mutate(PrintEdit(change: .favorite(true), targets: [machine.id: ["a.png"]]))
        // Another client renames the same print; the feed carries the
        // machine's row, which does not have our star yet.
        fake.prints[0] = FakeFixtures.print("a.png", prompt: "renamed")
        await library.refresh()
        let row = library.perHost[machine.id]?.first { $0.print.filename == "a.png" }
        #expect(row?.print.metadata.prompt == "renamed")
        #expect(row?.print.isFavorite == true)
    }

    @Test func lookupsFindEveryCopyThroughTheIndex() async {
        let (library, _, machine, _) = await bench(feed: false)
        let id = PrintID(host: machine.id, filename: "b.png")
        #expect(library.tile(containing: id)?.print.filename == "b.png")
        #expect(library.entry(id)?.hostID == machine.id)
        #expect(library.entry(PrintID(host: machine.id, filename: "missing.png")) == nil)
    }

    @Test func reloadingCollectionsKeepsTheListingValidators() async {
        let (library, fake, _, _) = await bench(feed: false)
        fake.conditionalGallery = true
        await library.refresh()
        await library.reloadCollections()
        await library.refresh()
        #expect(fake.galleryETags.last == .some("fake-etag"))
    }
}
