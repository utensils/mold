import Foundation
import MoldClient
import Testing

@testable import Mold

/// Library Sync costs O(changes): each run plans only what changed since each
/// source's mirror cursor, verifies only what it touched, and a run in which
/// nothing changed asks the machines nothing per print.
@MainActor
struct LibrarySyncDeltaTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func versionedPrint(_ name: String, favorite: Bool = false) -> GalleryPrint {
        let json = try! JSONSerialization.data(withJSONObject: [
            "filename": name, "timestamp": 1000, "size_bytes": 3, "favorite": favorite,
            "media_version": "1000:3", "metadata": ["prompt": "fox"],
        ])
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: json)
    }

    private struct Bench {
        let local: MoldHost
        let remote: MoldHost
        let source: FakeBackend
        let target: FakeBackend
        let hosts: HostStore
        let records: LibrarySyncRecords
        func library(_ session: LibrarySyncSession? = nil) -> LibraryStore {
            LibraryStore(hosts: hosts, syncSession: session ?? LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!),
                         syncRecords: records)
        }
    }

    /// A source whose prints a real destination lists once imported, with
    /// checkpoint revisions on both sides so copies seal.
    private func bench(_ names: [String], label: String) -> Bench {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("\(label)-\(UUID().uuidString.prefix(8))")
        let source = FakeBackend(host: remote, noRetainedMedia: true)
        let target = FakeBackend(host: local, noRetainedMedia: true)
        let prints = names.map { versionedPrint($0) }
        source.prints = prints
        source.mediaAnswer = Data([1, 2, 3])
        source.syncCheckpoint = Dictionary(uniqueKeysWithValues: names.map { ($0, "source-\($0)") })
        target.syncCheckpoint = Dictionary(uniqueKeysWithValues: names.map { ($0, "local-\($0)") })
        target.listsImports = { name in prints.first { $0.filename == name } }
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        return Bench(local: local, remote: remote, source: source, target: target, hosts: hosts,
                     records: LibrarySyncRecords(directory: nil, migrating: nil))
    }

    /// Per-print routes: anything a run asks about one print at a time.
    private func perPrintCalls(_ backend: FakeBackend) -> Int {
        ["mediaFile", "importPrint", "galleryPrint", "retainedMediaTransferOffer", "retainedMediaInventory",
         "gallerySyncCheckpoint", "boundedSyncCheckpoint", "mutate"].reduce(0) { $0 + backend.callCount($1) }
    }

    @Test func aRunWithNothingChangedAsksNothingPerPrint() async {
        let bench = bench(["a.png", "b.png", "c.png"], label: "unchanged")
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.target.importedNames.sorted() == ["a.png", "b.png", "c.png"])
        #expect(library.localSaveFailures.isEmpty)
        let records = bench.records.records(bench.hosts.instanceID(of: bench.local.id))
        #expect(records.count == 3)
        #expect(records.values.allSatisfy { $0.sealed })

        let source = perPrintCalls(bench.source), target = perPrintCalls(bench.target)
        await library.syncAllLocally()
        #expect(perPrintCalls(bench.source) == source)
        #expect(perPrintCalls(bench.target) == target)
        #expect(library.localSaveReport.contains("3 were already here"))
        #expect(library.localSaveFailures.isEmpty)
        // No receipt was ever read back one print at a time.
        #expect(bench.target.callCount("galleryPrint") == 0)
    }

    @Test func aCursorScopedPlanTouchesOnlyTheChangedName() async {
        let bench = bench(["a.png", "b.png", "c.png"], label: "scoped")
        let library = bench.library()
        await library.syncAllLocally()
        let downloads = bench.source.callCount("mediaFile")
        let probes = bench.source.boundedCheckpointRequests.count
        bench.source.prints = [versionedPrint("a.png"), versionedPrint("b.png", favorite: true), versionedPrint("c.png")]
        await library.syncAllLocally()
        #expect(bench.source.boundedCheckpointRequests.count == probes + 1)
        #expect(bench.source.boundedCheckpointRequests.last == ["b.png"])
        #expect(bench.source.callCount("mediaFile") == downloads, "a sealed, unchanged output is not downloaded again")
        #expect(bench.target.importedNames.count == 3)
        #expect(library.localSaveReport.contains("3 were already here"))
    }

    @Test func discoveryReadsEverySourceAtOnce() async {
        let bench = bench(["a.png"], label: "slow")
        let other = host("fast-\(UUID().uuidString.prefix(8))")
        let fast = FakeBackend(host: other, noRetainedMedia: true)
        let hosts = HostStore(hosts: [bench.local, bench.remote, other]) { machine in
            machine.id == bench.local.id ? bench.target : machine.id == other.id ? fast : bench.source
        }
        hosts.reachability[bench.local.id] = .up(FakeFixtures.serverStatus())
        bench.source.delays["gallery"] = .seconds(1)
        let library = LibraryStore(hosts: hosts, syncRecords: bench.records)
        let started = ContinuousClock.now
        let run = Task { await library.syncAllLocally() }
        await settle { fast.callCount("gallery") > 0 }
        #expect(fast.callCount("gallery") > 0)
        #expect(ContinuousClock.now - started < .milliseconds(900), "the second source waited for the first")
        await run.value
        #expect(bench.target.importedNames == ["a.png"])
    }

    @Test func verificationAsksTheBoundedCheckpointForTouchedNamesAndFallsBackOnAnOlderServer() async {
        let bench = bench(["a.png", "b.png"], label: "bounded")
        bench.target.boundedCheckpointUnsupported = true
        let library = bench.library()
        await library.syncAllLocally()
        // Both final reads named exactly the touched copies...
        #expect(bench.target.boundedCheckpointRequests.suffix(2).allSatisfy { Set($0) == ["a.png", "b.png"] })
        // ...and an older server answered them through the whole GET.
        #expect(bench.target.callCount("gallerySyncCheckpoint") >= 3)
        let records = bench.records.records(bench.hosts.instanceID(of: bench.local.id))
        #expect(records.values.allSatisfy { $0.sealed })
        let target = perPrintCalls(bench.target)
        await library.syncAllLocally()
        #expect(perPrintCalls(bench.target) == target)
    }

    @Test func aRecordIsPrunedOnlyWhenBothSidesAreGone() async {
        let bench = bench(["kept.png", "gone.png"], label: "tombstone")
        let library = bench.library()
        await library.syncAllLocally()
        let instance = bench.hosts.instanceID(of: bench.local.id)
        #expect(bench.records.records(instance).count == 2)
        // Removed here only: the record is what keeps it removed.
        bench.target.prints.removeAll { $0.filename == "kept.png" }
        // Removed on both sides, Trash included: nothing can bring it back.
        bench.source.prints.removeAll { $0.filename == "gone.png" }
        bench.target.prints.removeAll { $0.filename == "gone.png" }
        await library.syncAllLocally()
        let records = bench.records.records(instance)
        #expect(records.keys.contains("\(bench.remote.id.uuidString):kept.png"))
        #expect(!records.keys.contains("\(bench.remote.id.uuidString):gone.png"))
        #expect(library.localSaveReport.contains("kept removed"))
        #expect(bench.target.importedNames.count == 2)
    }

    @Test func quittingMidRunStopsTransfersAndTheNextLaunchFinishesWithoutDuplicates() async {
        let names = (1...6).map { "clip-\($0).png" }
        let bench = bench(names, label: "quit")
        bench.target.delays["importPrint"] = .milliseconds(150)
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let library = bench.library(session)
        session.syncNow(in: library)
        await settle { bench.target.callCount("importPrint") > 0 }
        session.cancelForQuit(in: library)
        await settle { library.localSaveTask == nil && !library.localSaveRunning }
        #expect(library.localSaveTask == nil)
        #expect(bench.target.importedNames.count < names.count)
        // Nothing starts once quitting.
        session.syncNow(in: library)
        #expect(library.localSaveTask == nil)

        bench.target.delays["importPrint"] = nil
        let relaunched = bench.library()
        await relaunched.syncAllLocally()
        #expect(Set(bench.target.importedNames) == Set(names))
        #expect(bench.target.importedNames.count == Set(bench.target.importedNames).count)
        #expect(relaunched.localSaveFailures.isEmpty)
    }
}
