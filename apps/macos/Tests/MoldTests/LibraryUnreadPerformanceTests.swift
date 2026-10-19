import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor struct LibraryUnreadPerformanceTests {
    @Test func patchedArrivalUpdatesBadgeImmediatelyAndFlushesLatestState() throws {
        let name = "unread-performance-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let hosts = HostStore(hosts: [host]) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts, readDefaults: defaults)
        library.writeLive(host.id, [])
        library.rebuild()
        let entry = LibraryEntry(host: host, print: FakeFixtures.print("arrival.png", prompt: "arrival"))
        library.writeLive(host.id, [entry])
        library.changedLive([entry.id])
        #expect(library.unreadCount == 1)
        #expect(library.unreadSaveTask != nil)
        library.flushUnreadMedia()
        #expect(library.unreadSaveTask == nil)
        #expect(LibraryUnreadLedger.load(from: defaults).count == 1)
        library.markViewed(entry.id)
        #expect(library.unreadCount == 0)
        #expect(LibraryUnreadLedger.load(from: defaults).count == 0)
    }

    @Test func hiddenPrivacyStillProjectsWholeLibraryOnAnIncrementalRefresh() throws {
        let name = "unread-privacy-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let library = LibraryStore(hosts: HostStore(hosts: [host]) { FakeBackend(host: $0) }, readDefaults: defaults)
        library.writeLive(host.id, [])
        library.rebuild()
        var print = GalleryPrint.Mutable(FakeFixtures.print("private.png", prompt: "private"))
        print.collections = ["private"]
        let entry = LibraryEntry(host: host, print: print.build())
        library.writeLive(host.id, [entry])
        library.changedLive([entry.id])
        #expect(library.unreadCount == 1)
        library.collectionsPerHost = [host.id: [Collection(id: "private", name: "Private", slug: "private", hidden: true)]]
        library.refreshUnreadMedia(changed: [], removed: [])
        #expect(library.unreadCount == 0)
        #expect(library.unreadMedia.isUnread(entry))
        library.flushUnreadMedia()
    }
}

extension LibraryUnreadPerformanceTests {
    @Test func syncBurstUsesOnePendingRefreshAndFinalFlushHonorsViewedArrival() throws {
        let name = "unread-sync-burst-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let library = LibraryStore(hosts: HostStore(hosts: [host]) { FakeBackend(host: $0) }, readDefaults: defaults)
        library.writeLive(host.id, [])
        library.rebuild()
        library.localSaveRunning = true
        let arrival = LibraryEntry(host: host, print: FakeFixtures.print("arrival.png", prompt: "arrival"))
        let viewed = LibraryEntry(host: host, print: FakeFixtures.print("viewed.png", prompt: "viewed"))
        library.items = [arrival, viewed]
        for _ in 0..<100 { library.refreshUnreadMedia() }
        #expect(library.unreadRefreshTask != nil)
        #expect(library.unreadCount == 0)
        library.markViewed(viewed.id)
        #expect(library.unreadRefreshTask == nil)
        #expect(library.unreadCount == 1)
        let later = LibraryEntry(host: host, print: FakeFixtures.print("later.png", prompt: "later"))
        library.items.append(later)
        library.refreshUnreadMedia()
        #expect(library.unreadRefreshTask != nil)
        library.localSaveRunning = false
        #expect(library.unreadRefreshTask == nil)
        #expect(library.unreadSaveTask == nil)
        #expect(!library.unreadMedia.isUnread(viewed))
        #expect(library.unreadCount == 2)
        #expect(LibraryUnreadLedger.load(from: defaults).count == 2)
    }
}
