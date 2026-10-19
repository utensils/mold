import Foundation
import Testing
@testable import MoldClient

struct LibraryUnreadLedgerTests {
    private let one = UUID(), two = UUID()
    private func entry(_ filename: String, host: UUID, size: Int = 1000) throws -> LibraryEntry {
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"\#(filename)","timestamp":1000,"size_bytes":\#(size),"metadata":{"seed":7,"model":"flux"}}"#.utf8))
        return LibraryEntry(host: MoldHost(id: host, name: "Fixture", baseURL: URL(string: "http://127.0.0.1:7680")!), print: print)
    }
    private func observe(_ rows: [LibraryEntry], ledger: inout LibraryUnreadLedger, visible: [LibraryEntry]? = nil) {
        let merged = LibraryMerge.merge(rows, localHost: nil)
        ledger.observe(entries: merged, visible: visible ?? merged, loadedHosts: [one, two], presentHosts: [one, two])
    }

    @Test func libraryVisitReadsCapturedCopiesOnlyOnDeparture() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let shown = try entry("shown.png", host: one), mirror = try entry("shown.png", host: two)
        let hidden = try entry("hidden.png", host: one), later = try entry("later.png", host: one)
        let merged = LibraryMerge.merge([shown, mirror], localHost: nil)
        observe([shown, mirror, hidden], ledger: &ledger)
        var visit = LibraryVisit()
        visit.capture(merged)
        #expect(ledger.isUnread(shown))
        observe([shown, mirror, hidden, later], ledger: &ledger)
        ledger.markSeen(visit.finish())
        #expect(!ledger.isUnread(shown))
        #expect(!ledger.isUnread(mirror))
        #expect(ledger.isUnread(hidden))
        #expect(ledger.isUnread(later))
        #expect(visit.finish().isEmpty)
    }

    @Test func cachedTrashToAllCapturesDespiteIdenticalResolvedQuery() throws {
        let image = try entry("cached.png", host: one)
        // Scope, not just the resolved query/revision, controls visit capture.
        #expect(LibraryScope.trash.resolve(LibraryQuery(), shelves: [], hiddenCollectionIDs: [:])
            == LibraryScope.all.resolve(LibraryQuery(), shelves: [], hiddenCollectionIDs: [:]))
        var visit = LibraryVisit()
        visit.capture([image], scope: .trash)
        #expect(visit.finish().isEmpty)
        visit.capture([image], scope: .all)
        #expect(visit.finish() == [image.id])
    }

    @Test func firstInventoryBaselineThenViewingAndVisitSeenPersist() throws {
        var ledger = LibraryUnreadLedger()
        let old = try entry("old.png", host: one)
        observe([old], ledger: &ledger)
        #expect(ledger.count == 0)
        let image = try entry("new.png", host: one), video = try entry("new.mp4", host: one)
        observe([old, image, video], ledger: &ledger)
        #expect(ledger.count == 2)
        ledger.view(image)
        #expect(ledger.count == 1)
        var restored = try MoldJSON.decoder.decode(LibraryUnreadLedger.self, from: MoldJSON.encoder.encode(ledger))
        #expect(restored.count == 1)
        restored.markSeen([old, image, video])
        #expect(restored.count == 0)
    }

    @Test func individualViewingPersistsWithoutReadingOtherArrivals() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let image = try entry("image.png", host: one), video = try entry("video.mp4", host: one)
        observe([image, video], ledger: &ledger)
        #expect(ledger.isUnread(image))
        ledger.view(image)
        var restored = try MoldJSON.decoder.decode(LibraryUnreadLedger.self, from: MoldJSON.encoder.encode(ledger))
        observe([image, video], ledger: &restored)
        #expect(!restored.isUnread(image))
        #expect(restored.isUnread(video))
        let later = try entry("later.png", host: one)
        observe([image, video, later], ledger: &restored)
        #expect(restored.count == 2)
        #expect(restored.isUnread(later))
    }

    @Test func duplicateCopiesCountOnceButNameCollisionsRemainSeparate() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let image = try entry("same.png", host: one), copy = try entry("same.png", host: two)
        observe([image, copy], ledger: &ledger)
        #expect(ledger.count == 1)
        let collision = try entry("same.png", host: two, size: 2000)
        observe([image, collision], ledger: &ledger)
        #expect(ledger.count == 2)
        ledger.view(image)
        #expect(ledger.count == 1)
    }

    @Test func viewingKnownPhysicalCopyClearsItsLogicalGroupWithoutAListing() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let original = try entry("source.png", host: one)
        let mirror = try entry("renamed-copy.png", host: two)
        let unrelated = try entry("unrelated.png", host: one, size: 2000)
        observe([original, mirror, unrelated], ledger: &ledger)
        #expect(ledger.count == 2)
        // The viewer can finish a pending load after its in-memory inventory
        // was cleared. Its physical ID still has persisted logical identity.
        ledger.observe(entries: [], visible: [], loadedHosts: [], presentHosts: [one, two])
        ledger.view(mirror.id)
        #expect(ledger.count == 1)
        #expect(!ledger.isUnread(original))
        #expect(!ledger.isUnread(mirror))
        #expect(ledger.isUnread(unrelated))
    }

    @Test func renamedCopyOfReadPrintDoesNotBecomeNew() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let image = try entry("source.png", host: one)
        observe([image], ledger: &ledger)
        ledger.view(image)
        let renamed = try entry("collision-renamed.png", host: two)
        observe([image, renamed], ledger: &ledger)
        #expect(ledger.count == 0)
        observe([renamed], ledger: &ledger)
        #expect(ledger.count == 0, "Read state follows the remaining copy")
    }

    @Test func hiddenTrashUnavailableAndRemovedHosts() throws {
        var ledger = LibraryUnreadLedger()
        observe([], ledger: &ledger)
        let image = try entry("new.png", host: one)
        observe([image], ledger: &ledger)
        #expect(ledger.count == 1)
        observe([image], ledger: &ledger, visible: [])
        #expect(ledger.count == 0)
        observe([image], ledger: &ledger)
        #expect(ledger.count == 1)
        ledger.observe(entries: [], visible: [], loadedHosts: [], presentHosts: [one, two])
        #expect(ledger.count == 1, "Unavailable inventory is not read")
        observe([], ledger: &ledger)
        #expect(ledger.count == 0, "Authoritative removal/trash removes the count")
        observe([image], ledger: &ledger)
        #expect(ledger.count == 1, "Restored unread print remains unread")
        ledger.retainHosts([two])
        #expect(ledger.count == 0)
    }

    @Test func newMachineBaselineDoesNotBadgeItsHistory() throws {
        var ledger = LibraryUnreadLedger()
        let image = try entry("history.png", host: one)
        ledger.observe(entries: [image], visible: [image], loadedHosts: [one], presentHosts: [one])
        #expect(ledger.count == 0)
    }
    @Test func viewingBeforeInventoryPersistsWithoutBootstrappingTheHost() throws {
        var ledger = LibraryUnreadLedger()
        let result = try entry("result.png", host: one)
        ledger.view(result.id)
        ledger = try MoldJSON.decoder.decode(LibraryUnreadLedger.self, from: MoldJSON.encoder.encode(ledger))
        let historical = try entry("history.png", host: one, size: 2000)
        observe([result, historical], ledger: &ledger)
        #expect(!ledger.isUnread(result))
        #expect(!ledger.isUnread(historical), "Pending viewed IDs must not bootstrap their whole host")
        let arriving = try entry("later.png", host: one, size: 3000)
        observe([result, historical, arriving], ledger: &ledger)
        #expect(ledger.isUnread(arriving))
    }
    @Test func largeRepeatedInventoriesKeepUnavailableUnreadCopies() throws {
        let rows = try (0..<20_000).map { try entry("history-\($0).png", host: one, size: $0 + 1) }
        let arriving = try entry("arrival.png", host: one, size: 20_001)
        var ledger = LibraryUnreadLedger()
        ledger.observe(entries: rows, visible: rows, loadedHosts: [one], presentHosts: [one])
        let all = rows + [arriving]
        let start = ContinuousClock.now
        for _ in 0..<3 { ledger.observe(entries: all, visible: all, loadedHosts: [one], presentHosts: [one]) }
        #expect(ledger.count == 1)
        ledger.observe(entries: [], visible: [], loadedHosts: [], presentHosts: [one])
        #expect(ledger.count == 1)
        ledger.observe(entries: all, visible: all, loadedHosts: [one], presentHosts: [one])
        #expect(ledger.count == 1)
        print("20k ledger repeated observation: \(start.duration(to: .now))")
    }
}

extension LibraryUnreadLedgerTests {
    @Test func incrementalArrivalProcessesOnlyChangedIdentityAndMatchesFullObservation() throws {
        let history = try (0..<10_000).map { try entry("history-\($0).png", host: one, size: $0 + 1) }
        var incremental = LibraryUnreadLedger(), full = LibraryUnreadLedger()
        incremental.observe(entries: history, visible: history, loadedHosts: [one], presentHosts: [one])
        full = incremental
        let arriving = try entry("arrival.png", host: one, size: 20_000)
        let work = incremental.observedEntryCount
        incremental.observeChanges(entries: [arriving], visible: [arriving], removed: [], loadedHosts: [one], presentHosts: [one])
        full.observe(entries: history + [arriving], visible: history + [arriving], loadedHosts: [one], presentHosts: [one])
        #expect(incremental == full)
        #expect(incremental.count == 1)
        #expect(incremental.observedEntryCount == work + 1)
        incremental.view(arriving.id)
        #expect(incremental.count == 0)
        incremental.observeChanges(entries: [], visible: [], removed: [arriving.id], loadedHosts: [one], presentHosts: [one])
        full.observe(entries: history, visible: history, loadedHosts: [one], presentHosts: [one])
        full.view(arriving.id)
        #expect(incremental == full)
    }

    @Test func incrementalVisibilityAndNewCopyKeepBaselineAndReadInheritance() throws {
        let original = try entry("original.png", host: one)
        let copy = try entry("renamed.png", host: two)
        var ledger = LibraryUnreadLedger()
        ledger.observe(entries: [original], visible: [original], loadedHosts: [one], presentHosts: [one, two])
        let merged = LibraryMerge.merge([original, copy], localHost: nil)
        ledger.observeChanges(entries: merged, visible: merged, removed: [], loadedHosts: [one, two], presentHosts: [one, two])
        #expect(ledger.count == 0)
        let arriving = try entry("arrival.png", host: one, size: 9_000)
        ledger.observeChanges(entries: [arriving], visible: [arriving], removed: [], loadedHosts: [one, two], presentHosts: [one, two])
        #expect(ledger.count == 1)
        ledger.observeChanges(entries: [arriving], visible: [], removed: [], loadedHosts: [one, two], presentHosts: [one, two])
        #expect(ledger.count == 0)
        ledger.observeChanges(entries: [arriving], visible: [arriving], removed: [], loadedHosts: [one, two], presentHosts: [one, two])
        #expect(ledger.count == 1)
        let data = try MoldJSON.encoder.encode(ledger)
        var restored = try MoldJSON.decoder.decode(LibraryUnreadLedger.self, from: data)
        restored.view(arriving.id)
        #expect(restored.count == 0)
        #expect(!restored.isUnread(original))
    }
}

extension LibraryUnreadLedgerTests {
    @Test func incrementalMetadataDoesNotPersistOrCountTheSameGroupTwice() throws {
        let row = try entry("read.png", host: one)
        var ledger = LibraryUnreadLedger()
        ledger.observe(entries: [row], visible: [row], loadedHosts: [one], presentHosts: [one])
        for _ in 0..<100 {
            let changed = ledger.observeChanges(entries: [row], visible: [row], removed: [], loadedHosts: [one], presentHosts: [one])
            #expect(!changed)
        }
        #expect(ledger.count == 0)
        let encoded = try MoldJSON.encoder.encode(ledger)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(Set(object.keys).isSubset(of: ["known", "unread", "visible_groups", "pending_viewed"]))
        let restored = try MoldJSON.decoder.decode(LibraryUnreadLedger.self, from: encoded)
        #expect(restored == ledger)
    }
}

extension LibraryUnreadLedgerTests {
    @Test func incrementalHostPruningReportsPersistenceChange() throws {
        var ledger = LibraryUnreadLedger()
        ledger.observe(entries: [], visible: [], loadedHosts: [one], presentHosts: [one])
        let arrival = try entry("arrival.png", host: one)
        ledger.observeChanges(entries: [arrival], visible: [arrival], removed: [], loadedHosts: [one], presentHosts: [one])
        #expect(ledger.count == 1)
        let changed = ledger.observeChanges(entries: [], visible: [], removed: [], loadedHosts: [], presentHosts: [])
        #expect(changed)
        #expect(ledger.count == 0)
        let unchanged = ledger.observeChanges(entries: [], visible: [], removed: [], loadedHosts: [], presentHosts: [])
        #expect(!unchanged)
    }
}

extension LibraryUnreadLedgerTests {
    @Test func largeHiddenInventoryPreservesPrivacyWhileObservingArrivals() throws {
        let host = MoldHost(id: one, name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let history = try (0..<20_000).map { index in
            var print = GalleryPrint.Mutable(try entry("history-\(index).png", host: one, size: index + 1).print)
            if index.isMultiple(of: 2) { print.collections = ["private"] }
            return LibraryEntry(host: host, print: print.build())
        }
        var query = LibraryQuery()
        query.hiddenCollectionIDs = [one: ["private"]]
        var ledger = LibraryUnreadLedger()
        ledger.observe(entries: history, visible: query.apply(to: history), loadedHosts: [one], presentHosts: [one])
        let all = history + [try entry("visible-arrival.png", host: one, size: 20_001)]
        let start = ContinuousClock.now
        for _ in 0..<3 {
            ledger.observe(entries: all, visible: query.apply(to: all), loadedHosts: [one], presentHosts: [one])
        }
        #expect(ledger.count == 1)
        #expect(query.apply(to: all).count == 10_001)
        print("20k hidden privacy projection plus ledger, three observations: \(start.duration(to: .now))")
    }
}
