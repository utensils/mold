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
