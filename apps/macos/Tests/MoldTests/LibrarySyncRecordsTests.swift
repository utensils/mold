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
}
