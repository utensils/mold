import Foundation
import Observation
import Synchronization
import MoldClient
import Testing
@testable import Mold

@MainActor struct LibraryVisibilityTests {
    @Test func mixedHiddenFlagsProtectAndRepairEveryReplica() async {
        let firstHost = MoldHost(name: "This Mac", baseURL: URL(string: "http://local")!)
        let secondHost = MoldHost(name: "Workstation", baseURL: URL(string: "http://remote")!)
        let first = FakeBackend(host: firstHost), second = FakeBackend(host: secondHost)
        let slug = "visibility-\(UUID().uuidString)"
        first.collectionRows = [Collection(id: "local", name: "Hidden", slug: slug, hidden: true)]
        second.collectionRows = [Collection(id: "remote", name: "Hidden", slug: slug, hidden: false)]
        let hosts = HostStore(hosts: [firstHost, secondHost]) { $0.id == firstHost.id ? first : second }
        let library = LibraryStore(hosts: hosts)
        await library.refreshOrganization()
        #expect(library.hiddenCollectionIDs[firstHost.id] == ["local"])
        #expect(library.hiddenCollectionIDs[secondHost.id] == ["remote"])
        #expect(second.collectionRows.first?.hidden == true)
        #expect(second.callCount("updateCollection") == 1)
        await library.refreshOrganization()
        #expect(library.collectionVisibility.intents[slug] == nil)
    }
}

extension LibraryVisibilityTests {
    @Test func sourceDetailsRetainTrashAccessWithoutAddingTrashToBrowser() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let hosts = HostStore(hosts: [host]) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)
        let entry = LibraryEntry(host: host, print: FakeFixtures.print("trashed.png"))
        library.trashed = [entry]
        let authority = SourceMediaAuthority(owner: entry, route: host, instance: nil)
        #expect(!library.allowsSource(authority))
        #expect(library.allowsSource(authority, detail: true))
        library.trashed = []
        #expect(!library.allowsSource(authority, detail: true))
    }
}

extension LibraryVisibilityTests {
    @Test func explicitlyOpenedHiddenPrintDoesNotEnterSourceBrowser() async throws {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let fake = FakeBackend(host: host)
        fake.collectionRows = [Collection(id: "private", name: "Private", slug: "source-private-\(UUID())", hidden: true)]
        let hosts = HostStore(hosts: [host]) { _ in fake }, library = LibraryStore(hosts: HostStore(hosts: []))
        let scoped = LibraryStore(hosts: hosts)
        await scoped.refreshOrganization()
        let row = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"hidden.png","timestamp":1000,"metadata":{},"collections":["private"]}"#.utf8))
        let entry = LibraryEntry(host: host, print: row)
        scoped.items = [entry]
        let authority = SourceMediaAuthority(owner: entry, route: host, instance: nil)
        #expect(!scoped.allowsSource(authority))
        #expect(scoped.allowsSource(authority, detail: true))
        #expect(!library.allowsSource(authority, detail: true))
    }
}

extension LibraryVisibilityTests {
    @Test func sourceAccessReusesIndexAndInvalidatesSameCountChanges() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let hosts = HostStore(hosts: [host]) { FakeBackend(host: $0) }
        let library = LibraryStore(hosts: hosts)
        let first = LibraryEntry(host: host, print: FakeFixtures.print("first.png"))
        let second = LibraryEntry(host: host, print: FakeFixtures.print("second.png"))
        library.items = [first]
        let authority = SourceMediaAuthority(owner: first, route: host, instance: nil)
        for _ in 0..<100 { #expect(library.allowsSource(authority)) }
        #expect(library.sourceAccessIndex.derivations == 1)
        library.items = [second]
        #expect(!library.allowsSource(authority))
        #expect(library.sourceAccessIndex.derivations == 2)
        library.items = [first]
        library.collectionsPerHost = [host.id: [Collection(id: "private", name: "Private", slug: "private", hidden: true)]]
        _ = library.sourceAccess
        #expect(library.sourceAccessIndex.derivations == 3)
    }
}


extension LibraryVisibilityTests {
    @Test func receiptPrivacyProtectsPrintsWhileSourceMediaIncludesAllCollections() async throws {
        let source = MoldHost(name: "Hidden source", baseURL: URL(string: "http://hidden-source")!)
        let destination = MoldHost(name: "Existing backup", baseURL: URL(string: "http://existing-backup")!)
        let hosts = HostStore(hosts: [source, destination]) { FakeBackend(host: $0) }
        hosts.reachability[source.id] = .up(FakeFixtures.serverStatus())
        hosts.reachability[destination.id] = .up(FakeFixtures.serverStatus())
        let library = LibraryStore(hosts: hosts)
        var original = GalleryPrint.Mutable(FakeFixtures.print("hidden-source.png", prompt: "private"))
        original.collections = ["private"]
        let copied = LibraryEntry(host: destination, print: FakeFixtures.print("collision-copy.png", prompt: "private"))
        library.collectionsPerHost = [source.id: [Collection(id: "private", name: "Private", slug: "private", hidden: true)]]
        library.items = [LibraryEntry(host: source, print: original.build()), copied]
        let authority = SourceMediaAuthority(owner: copied, route: destination, instance: hosts.instanceID(of: destination.id))
        #expect(library.allowsSource(authority))
        let derivations = library.sourceAccessIndex.derivations
        let changed = Mutex(false)
        withObservationTracking { _ = library.sourcePrivacyRevision } onChange: { changed.withLock { $0 = true } }
        var receipt = try #require(LocalSyncRecord(receipt: original.build(), from: source.id,
            as: copied.print.filename, destinationInstance: hosts.instanceID(of: destination.id)))
        receipt.destinationHost = destination.id; receipt.sourceRoute = library.originContext(source.id)
        receipt.destinationRoute = destination.baseURL.absoluteString
        try await library.syncRecords.checkpoint(receipt,
            key: LocalSyncRecord.key(hostID: source.id, filename: original.build().filename), instance: library.recordScope(destination))
        #expect(changed.withLock { $0 })
        #expect(library.allowsSource(authority), "The dedicated Source Media section includes hidden collections")
        #expect(library.sourceAccessIndex.derivations == derivations)
        #expect(library.syncLinks().isEmpty)
        #expect(LibraryScope.all.resolve(LibraryQuery(), shelves: library.shelves,
            hiddenCollectionIDs: library.hiddenCollectionIDs, privacyEvidence: library.trashed,
            privacyLinks: library.syncPrivacyLinks()).apply(to: library.items).isEmpty)
        let privacyRevision = library.sourcePrivacyRevision
        receipt.organization = LibraryOrganizationState(observed: .init())
        try await library.syncRecords.checkpoint(receipt,
            key: LocalSyncRecord.key(hostID: source.id, filename: original.build().filename), instance: library.recordScope(destination))
        #expect(library.sourcePrivacyRevision == privacyRevision)
        #expect(library.allowsSource(authority))
        #expect(library.sourceAccessIndex.derivations == derivations,
            "organization-only journal writes must not re-project all source owners")
        let protectedRevision = library.sourcePrivacyRevision
        library.collectionsPerHost = [:]
        #expect(library.sourcePrivacyRevision != protectedRevision,
            "removing the final hidden shelf must not alias a receipt privacy revision")
    }
}
