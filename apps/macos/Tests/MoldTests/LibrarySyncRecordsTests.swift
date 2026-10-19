import Foundation
import MoldClient
import Testing

@testable import Mold

/// Library Sync's records live in their own file, written off the main actor,
/// with a one-time migration out of the defaults suite.
@MainActor
struct LibrarySyncRecordsTests {
    private func scratch() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "mold-sync-records-\(UUID().uuidString)")
    }

    private func print(_ name: String) -> GalleryPrint {
        let json = try! JSONSerialization.data(withJSONObject: [
            "filename": name, "timestamp": 1000, "size_bytes": 3,
            "media_version": "1000:3", "metadata": ["prompt": "fox"],
        ])
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: json)
    }

    private func record(_ name: String, instance: String?) -> LocalSyncRecord {
        LocalSyncRecord(source: print(name), from: UUID(), local: print(name), on: UUID(),
                        destinationInstance: instance)!
    }

    @Test func legacyDefaultsMigrateOnceIntoTheFileAndAreRemoved() async throws {
        let suite = "io.utensils.mold.native.tests.sync-records.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let directory = scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let host = UUID()
        let stored = record("copied.png", instance: "local-1")
        defaults.set(try MoldJSON.encoder.encode(["\(host.uuidString):copied.png": stored]), forKey: "librarySyncCopiesV1")
        struct Completed: Codable { let key: String; let record: LocalSyncRecord }
        let receipt = record("receipt.png", instance: "local-1")
        defaults.set(try MoldJSON.encoder.encode(Completed(key: "\(host.uuidString):receipt.png", record: receipt)),
                     forKey: "librarySyncCompletedCopyV1.fixture")
        defaults.set(try MoldJSON.encoder.encode(["\(host.uuidString):pending.png": "pending.png"]),
                     forKey: "librarySyncPendingOrganizationV1")

        let migrated = LibrarySyncRecords(directory: directory, migrating: defaults)
        let records = migrated.records("local-1")
        #expect(records["\(host.uuidString):copied.png"] == stored)
        #expect(records["\(host.uuidString):receipt.png"] == receipt)
        // Pending organization never named its instance; the first known one adopts it.
        #expect(migrated.pending("\(host.uuidString):pending.png", instance: "local-1") == "pending.png")
        #expect(defaults.object(forKey: "librarySyncCopiesV1") == nil)
        #expect(defaults.object(forKey: "librarySyncPendingOrganizationV1") == nil)
        #expect(defaults.object(forKey: "librarySyncCompletedCopyV1.fixture") == nil)
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "records.json").path))

        await migrated.flush()
        let relaunched = LibrarySyncRecords(directory: directory, migrating: defaults)
        #expect(relaunched.records("local-1").count == 2)
        #expect(relaunched.pending("\(host.uuidString):pending.png", instance: "local-1") == "pending.png")
    }

    @Test func aReceiptIsDurableBeforeAnyCompaction() async throws {
        let directory = scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = LibrarySyncRecords(directory: directory, migrating: nil)
        let receipt = LocalSyncRecord(receipt: print("landed.png"), from: UUID(), as: "landed.png", destinationInstance: "i")!
        try await store.checkpoint(receipt, key: "k", instance: "i")
        try await store.markPending("k", filename: "landed.png", instance: "i")
        // A crash now: a new process reads the journal.
        let relaunched = LibrarySyncRecords(directory: directory, migrating: nil)
        #expect(relaunched.records("i")["k"] == receipt)
        #expect(relaunched.records("i")["k"]?.provisional == true)
        #expect(relaunched.pending("k", instance: "i") == "landed.png")
    }

    @Test func aFinishedRunAppliesOnceAndCompactsTheJournal() async throws {
        let directory = scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = LibrarySyncRecords(directory: directory, migrating: nil)
        try await store.checkpoint(record("a.png", instance: "i"), key: "a", instance: "i")
        try await store.checkpoint(record("b.png", instance: "i"), key: "b", instance: "i")
        let links = store.linksRevision
        var outcome = LibrarySyncRecords.RunOutcome()
        outcome.removed = ["b"]
        outcome.retried = ["a"]
        await store.finishRun(outcome, instance: "i")
        #expect(store.linksRevision == links + 1)
        #expect(store.records("i").keys.sorted() == ["a"])
        #expect(store.bucket("i").retry == ["a"])
        let journal = try Data(contentsOf: directory.appending(path: "journal.jsonl"))
        #expect(journal.isEmpty)
        let relaunched = LibrarySyncRecords(directory: directory, migrating: nil)
        #expect(relaunched.records("i").keys.sorted() == ["a"])
        #expect(relaunched.bucket("i").retry == ["a"])
    }

    @Test(arguments: ["{not json", #"{"version":2,"seq":0,"buckets":{}}"#])
    func anUnreadableDocumentIsSetAsideAndNeverOverwritten(_ contents: String) async throws {
        let directory = scratch()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(contents.utf8).write(to: directory.appending(path: "records.json"))
        let store = LibrarySyncRecords(directory: directory, migrating: nil)
        #expect(store.blockedReason != nil)
        #expect(store.records("i").isEmpty)
        let receipt = LocalSyncRecord(receipt: print("x.png"), from: UUID(), as: "x.png", destinationInstance: "i")!
        await #expect(throws: (any Error).self) { try await store.checkpoint(receipt, key: "k", instance: "i") }
        await store.finishRun(.init(), instance: "i")
        await store.flush()
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(!names.contains("records.json"), "nothing replaced the unreadable document")
        let aside = try #require(names.first { $0.hasPrefix("records.json.unreadable-") })
        #expect(try Data(contentsOf: directory.appending(path: aside)) == Data(contents.utf8))
        // A relaunch is still blocked until the set-aside file is dealt with.
        #expect(LibrarySyncRecords(directory: directory, migrating: nil).blockedReason != nil)
        try FileManager.default.removeItem(at: directory.appending(path: aside))
        #expect(LibrarySyncRecords(directory: directory, migrating: nil).blockedReason == nil)
    }

    @Test func receiptsDoNotInvalidateLinksUntilTheRunFinishes() async throws {
        let store = LibrarySyncRecords(directory: nil, migrating: nil)
        let local = UUID()
        let before = store.links(localHost: local, instance: "i")
        try await store.checkpoint(record("a.png", instance: "i"), key: "\(UUID().uuidString):a.png", instance: "i")
        #expect(store.links(localHost: local, instance: "i") == before)
        await store.finishRun(.init(), instance: "i")
        #expect(store.links(localHost: local, instance: "i").count == 2)
    }

    @Test func recipeDigestsAreCachedPerMediaVersion() {
        let host = UUID()
        let wire = print("cached.png").carryingRawMetadata(Data(#"{"prompt":"fox"}"#.utf8))
        #expect(!LocalSyncRecord.recipeCached(wire, host: host))
        let first = LocalSyncRecord.recipe(wire, host: host)
        #expect(LocalSyncRecord.recipeCached(wire, host: host))
        #expect(LocalSyncRecord.recipe(wire, host: host) == first)
        // Another machine's same-named print is never answered from this one,
        // an edited recipe at the same media version is another key, and a
        // row without raw recipe bytes is never cached at all.
        #expect(!LocalSyncRecord.recipeCached(wire, host: UUID()))
        let edited = print("cached.png").carryingRawMetadata(Data(#"{"prompt":"owl"}"#.utf8))
        #expect(!LocalSyncRecord.recipeCached(edited, host: host))
        #expect(LocalSyncRecord.recipe(edited, host: host) != first)
        _ = LocalSyncRecord.recipe(print("typed.png"), host: host)
        #expect(!LocalSyncRecord.recipeCached(print("typed.png"), host: host))
    }
    @Test func remoteDestinationCannotAdoptUnboundHistoricalLocalRecords() async throws {
        let store = LibrarySyncRecords()
        let key = LocalSyncRecord.key(hostID: UUID(), filename: "legacy.png")
        try await store.checkpoint(record("legacy.png", instance: nil), key: key, instance: nil)
        let remoteScope = "\(UUID().uuidString)\nhttps://destination.example\nremote-instance"
        #expect(store.records(remoteScope).isEmpty)
        #expect(store.records("historical-local")[key] != nil)
    }

    @Test func verifiedDestinationComponentsPreserveFanoutAcrossBuckets() async throws {
        let store = LibrarySyncRecords()
        let sourceHost = UUID(), firstHost = UUID(), secondHost = UUID()
        let source = PrintID(host: sourceHost, filename: "source.png")
        let key = LocalSyncRecord.key(hostID: sourceHost, filename: source.filename)
        var first = record("renamed-first.png", instance: "first")
        first.destinationHost = firstHost
        var second = record("renamed-second.png", instance: "second")
        second.destinationHost = secondHost
        try await store.checkpoint(first, key: key, instance: "first")
        await store.finishRun(.init(), instance: "first")
        try await store.checkpoint(second, key: key, instance: "second")
        await store.finishRun(.init(), instance: "second")
        #expect(store.verifiedCopy(of: source, on: firstHost)?.filename == first.destinationFilename)
        #expect(store.verifiedCopy(of: source, on: secondHost)?.filename == second.destinationFilename)
        #expect(store.verifiedCopies(of: source).count == 3)
    }

    @Test func conflictingIndependentLineageNeverChoosesAnOriginByDictionaryOrder() async throws {
        let store = LibrarySyncRecords()
        let copyHost = UUID(), firstSource = UUID(), secondSource = UUID()
        var copy = record("shared.png", instance: "backup")
        copy.destinationHost = copyHost
        try await store.checkpoint(copy, key: LocalSyncRecord.key(hostID: firstSource, filename: "first.png"), instance: "backup")
        try await store.checkpoint(copy, key: LocalSyncRecord.key(hostID: secondSource, filename: "second.png"), instance: "backup")
        await store.finishRun(.init(), instance: "backup")
        let id = PrintID(host: copyHost, filename: "shared.png")
        #expect(store.hasVerifiedIncoming(id))
        #expect(store.verifiedOrigin(of: id) == nil)
        #expect(store.organizationAtOrigin(of: id) == nil)
    }

    @Test func replacedDestinationInstanceCannotInheritVerifiedLineage() async throws {
        let store = LibrarySyncRecords()
        let source = UUID(), destination = UUID()
        let key = LocalSyncRecord.key(hostID: source, filename: "source.png")
        var copy = record("copy.png", instance: "old-instance")
        copy.destinationHost = destination
        copy.sourceRoute = "\(source.uuidString)\nhttps://source.example\nsource-instance"
        copy.destinationRoute = "https://destination.example"
        try await store.checkpoint(copy, key: key, instance: "old-instance")
        await store.finishRun(.init(), instance: "old-instance")
        let id = PrintID(host: destination, filename: "copy.png")
        var authority = [source: copy.sourceRoute!, destination: "\(destination.uuidString)\nhttps://destination.example\nold-instance"]
        #expect(store.verifiedOrigin(of: id, authority: authority) == key)
        authority[destination] = "\(destination.uuidString)\nhttps://destination.example\nreplacement-instance"
        #expect(!store.hasVerifiedIncoming(id, authority: authority))
        #expect(store.verifiedOrigin(of: id, authority: authority) == nil)
        #expect(store.verifiedCopy(of: PrintID(host: source, filename: "source.png"), on: destination, authority: authority) == nil)
    }

    @Test func convergingLineageKeepsEveryEdgeRegardlessOfRecordTraversal() async throws {
        // Two independently recorded paths converge on both B and D. A
        // dictionary keyed by just the destination loses a verified edge.
        for _ in 0..<24 {
            let store = LibrarySyncRecords()
            let ids = (0..<5).map { _ in UUID() }
            let a = PrintID(host: ids[0], filename: "a.png"), b = PrintID(host: ids[1], filename: "b.png")
            let c = PrintID(host: ids[2], filename: "c.png"), d = PrintID(host: ids[3], filename: "d.png")
            let e = PrintID(host: ids[4], filename: "e.png")
            for (source, copy) in [(a,b),(c,b),(c,d),(e,d)].shuffled() {
                var record = LocalSyncRecord(source: print(source.filename), from: source.host,
                    local: print(copy.filename), on: copy.host, destinationInstance: "i")!
                record.destinationHost = copy.host
                try await store.checkpoint(record, key: LocalSyncRecord.key(hostID: source.host, filename: source.filename), instance: copy.host.uuidString)
            }
            let graph = store.links(localHost: ids[0], instance: nil)
            #expect(Set([a,b,c,d,e].compactMap { graph[$0] }).count == 1)
            #expect(graph.count == 5)
        }
    }

    /// Reconcile stamps the local host on records written without one, once
    /// per print. The copy is unchanged, so the identity clock -- which every
    /// Library count and privacy projection keys on -- must not move.
    @Test func reconcileBookkeepingNeverMovesThePrivacyClock() async throws {
        let source = MoldHost(name: "Origin", baseURL: URL(string: "http://origin")!)
        let original = print("kept.png")
        var record = LocalSyncRecord(receipt: original, from: source.id, as: "kept.png", destinationInstance: "local")!
        record.provisional = nil
        let key = LocalSyncRecord.key(hostID: source.id, filename: original.filename)
        let records = LibrarySyncRecords()
        try await records.checkpoint(record, key: key, instance: "local")
        let revision = records.privacyRevision
        let links = records.privacyLinks(localHost: MoldEngine.localHostID, instance: "local")
        record.destinationHost = MoldEngine.localHostID
        record.organization = LibraryOrganizationState(observed: .init())
        record.initialOrganizationPending = true
        try await records.checkpoint(record, key: key, instance: "local")
        record.initialOrganizationPending = nil
        try await records.checkpoint(record, key: key, instance: "local")
        #expect(records.privacyRevision == revision)
        #expect(records.privacyLinks(localHost: MoldEngine.localHostID, instance: "local") == links)
        record.destinationRoute = "http://elsewhere"
        try await records.checkpoint(record, key: key, instance: "local")
        #expect(records.privacyRevision != revision, "a re-routed copy is a different identity")
    }

    /// A receipt verified mid-run leaves the provisional set before `finishRun`
    /// publishes ordinary links; it must not drop out of privacy evidence between.
    @Test func aCopyVerifiedMidRunStaysPrivacyEvidence() async throws {
        let source = MoldHost(name: "Origin", baseURL: URL(string: "http://origin")!)
        let original = print("secret.png")
        var receipt = LocalSyncRecord(receipt: original, from: source.id, as: "copy.png", destinationInstance: "local")!
        let key = LocalSyncRecord.key(hostID: source.id, filename: original.filename)
        let records = LibrarySyncRecords()
        try await records.checkpoint(receipt, key: key, instance: "local")
        receipt.provisional = nil
        try await records.checkpoint(receipt, key: key, instance: "local")
        #expect(records.links(localHost: MoldEngine.localHostID, instance: "local").isEmpty)
        let privacy = records.privacyLinks(localHost: MoldEngine.localHostID, instance: "local")
        #expect(privacy[PrintID(host: MoldEngine.localHostID, filename: "copy.png")]
            == privacy[PrintID(host: source.id, filename: original.filename)])
        #expect(privacy[PrintID(host: source.id, filename: original.filename)] != nil)
    }

    @Test func provisionalCollisionReceiptHidesOnlyAndNeverAuthorizesSkipOrOwnership() async throws {
        let source = MoldHost(name: "Hidden origin", baseURL: URL(string: "http://hidden-origin")!)
        let destination = MoldHost(name: "Unfiled backup", baseURL: URL(string: "http://unfiled-backup")!)
        var hidden = GalleryPrint.Mutable(print("hidden.png")); hidden.collections = ["secret"]
        let original = hidden.build(), copied = print("collision-renamed.png")
        var receipt = LocalSyncRecord(receipt: original, from: source.id, as: copied.filename, destinationInstance: "destination")!
        receipt.destinationHost = destination.id
        receipt.sourceRoute = [source.id.uuidString, source.baseURL.absoluteString, "origin"].joined(separator: "\n")
        receipt.destinationRoute = destination.baseURL.absoluteString
        let authority = [source.id: receipt.sourceRoute!, destination.id: [destination.id.uuidString, destination.baseURL.absoluteString, "destination"].joined(separator: "\n")]
        let records = LibrarySyncRecords()
        try await records.checkpoint(receipt, key: LocalSyncRecord.key(hostID: source.id, filename: original.filename), instance: "destination")
        var query = LibraryQuery(); query.hiddenCollectionIDs = [source.id: ["secret"]]
        query.privacyLinks = records.privacyLinks(localHost: destination.id, instance: "destination", authority: authority)
        #expect(query.apply(to: [LibraryEntry(host: source, print: original), LibraryEntry(host: destination, print: copied)]).isEmpty)
        #expect(records.links(localHost: destination.id, instance: "destination", authority: authority).isEmpty)
        #expect(records.verifiedOrigin(of: PrintID(host: destination.id, filename: copied.filename), authority: authority) == nil)
        #expect(!receipt.sealed)
        #expect(!receipt.matches(source: original, from: source.id, local: copied, on: destination.id))
        var replaced = authority
        replaced[destination.id] = "replacement route/instance"
        #expect(records.privacyLinks(localHost: destination.id, instance: "destination", authority: replaced).isEmpty)
    }

}
