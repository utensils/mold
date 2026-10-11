import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct SourceMediaAccessIndexTests {
    @Test func tenThousandOwnersAreProjectedOnceAcrossRepeatedReads() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let owners = (0..<10_000).map { PrintFixtures.entry("print-\($0).png", host: host.id) }
        let cache = SourceMediaAccessIndex()
        let key = SourceMediaAccessIndex.Key(revision: 0, visibility: UUID(uuidString: "00000000-0000-0000-0000-000000000000")!, hosts: [host], instances: [:])
        var builds = 0
        for _ in 0..<100 {
            let snapshot = cache.snapshot(key: key) {
                builds += 1
                return .init(visible: owners, details: owners, hosts: [host], instances: [:])
            }
            #expect(snapshot.allows(.init(owner: owners[9999], route: host, instance: nil)))
        }
        #expect(builds == 1)
        #expect(cache.derivations == 1)
    }

    @Test func visibilityAndSameCountReplacementInvalidateImmediately() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let first = PrintFixtures.entry("first.png", host: host.id)
        let second = PrintFixtures.entry("second.png", host: host.id)
        let cache = SourceMediaAccessIndex()
        let authority = SourceMediaAuthority(owner: first, route: host, instance: nil)
        let before = cache.snapshot(key: .init(revision: 0, visibility: UUID(uuidString: "00000000-0000-0000-0000-000000000000")!, hosts: [host], instances: [:])) {
            .init(visible: [first], details: [first], hosts: [host], instances: [:])
        }
        #expect(before.allows(authority))
        let hidden = cache.snapshot(key: .init(revision: 0, visibility: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!, hosts: [host], instances: [:])) {
            .init(visible: [], details: [first], hosts: [host], instances: [:])
        }
        #expect(!hidden.allows(authority))
        #expect(hidden.allows(authority, detail: true))
        let replaced = cache.snapshot(key: .init(revision: 1, visibility: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!, hosts: [host], instances: [:])) {
            .init(visible: [second], details: [second], hosts: [host], instances: [:])
        }
        #expect(!replaced.allows(authority, detail: true))
    }

    @Test func removedAndRekeyedHostsCannotAuthorizeOldCopies() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!, apiKey: "before")
        let owner = PrintFixtures.entry("owner.png", host: host.id)
        let authority = SourceMediaAuthority(owner: owner, route: host, instance: "one")
        var changed = host; changed.apiKey = "after"
        let rekeyed = SourceMediaAccessIndex.Snapshot(visible: [owner], details: [owner], hosts: [changed], instances: [host.id: "one"])
        #expect(!rekeyed.allows(authority))
        let restarted = SourceMediaAccessIndex.Snapshot(visible: [owner], details: [owner], hosts: [host], instances: [host.id: "two"])
        #expect(!restarted.allows(authority))
        let removed = SourceMediaAccessIndex.Snapshot(visible: [owner], details: [owner], hosts: [], instances: [:])
        #expect(!removed.allows(authority, detail: true))
        #expect(removed.owners.isEmpty)
    }
}
