import Foundation
import MoldClient
import Testing

@testable import Mold

/// Library Sync costs O(changes): Keep in Sync's automatic runs plan only what
/// changed since each source's mirror cursor, a press plans everything against
/// checkpoint evidence, every run verifies only what it touched, and a run in
/// which nothing changed asks the machines nothing per print.
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
        var instance: String? { hosts.instanceID(of: local.id) }
        func key(_ name: String) -> String { LocalSyncRecord.key(hostID: remote.id, filename: name) }
    }

    /// A source whose prints a real destination lists once imported, with
    /// checkpoint revisions on both sides so copies seal. Each bench owns
    /// its records.
    private func bench(_ names: [String], label: String, records: LibrarySyncRecords = LibrarySyncRecords()) -> Bench {
        let local = MoldEngine.localHost(port: 7680, apiKey: "test")!
        let remote = host("\(label)-\(UUID().uuidString.prefix(8))")
        let source = FakeBackend(host: remote, noRetainedMedia: true)
        let target = FakeBackend(host: local, noRetainedMedia: true)
        let prints = names.map { versionedPrint($0) }
        source.prints = prints
        source.mediaAnswer = Data([1, 2, 3])
        source.syncCheckpoint = Dictionary(uniqueKeysWithValues: names.map { ($0, "source-\($0)") })
        target.syncCheckpoint = Dictionary(uniqueKeysWithValues: names.map { ($0, "local-\($0)") })
        target.mediaAnswer = Data([1, 2, 3])
        target.listsImports = { name in prints.first { $0.filename == name } }
        let hosts = HostStore(hosts: [local, remote]) { $0.id == local.id ? target : source }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        return Bench(local: local, remote: remote, source: source, target: target, hosts: hosts, records: records)
    }

    /// Per-print routes: anything a run asks about one print at a time.
    private func perPrintCalls(_ backend: FakeBackend) -> Int {
        ["mediaFile", "importPrint", "galleryPrint", "retainedMediaTransferOffer", "retainedMediaInventory",
         "boundedSyncCheckpoint", "mutate"].reduce(0) { $0 + backend.callCount($1) }
    }

    @Test func aRunWithNothingChangedAsksNothingPerPrint() async {
        let bench = bench(["a.png", "b.png", "c.png"], label: "unchanged")
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.target.importedNames.sorted() == ["a.png", "b.png", "c.png"])
        #expect(library.localSaveFailures.isEmpty)
        let records = bench.records.records(bench.instance)
        #expect(records.count == 3)
        #expect(records.values.allSatisfy { $0.sealed })

        // Keep in Sync: nothing changed, nothing asked.
        var source = perPrintCalls(bench.source), target = perPrintCalls(bench.target)
        let wholeSource = bench.source.callCount("gallerySyncCheckpoint")
        let wholeTarget = bench.target.callCount("gallerySyncCheckpoint")
        await library.syncAllLocally(fullPlan: false)
        #expect(perPrintCalls(bench.source) == source)
        #expect(perPrintCalls(bench.target) == target)
        #expect(bench.source.callCount("gallerySyncCheckpoint") == wholeSource)
        #expect(library.localSaveReport.contains("3 were already here"))

        // A press plans every print, but skips on one checkpoint read per
        // machine and cached recipe digests: still nothing per print.
        source = perPrintCalls(bench.source)
        target = perPrintCalls(bench.target)
        await library.syncAllLocally()
        #expect(perPrintCalls(bench.source) == source)
        #expect(perPrintCalls(bench.target) == target)
        #expect(bench.source.callCount("gallerySyncCheckpoint") == wholeSource + 1)
        #expect(bench.target.callCount("gallerySyncCheckpoint") == wholeTarget + 1)
        #expect(library.localSaveReport.contains("3 were already here"))
        #expect(library.localSaveFailures.isEmpty)
        #expect(bench.target.callCount("galleryPrint") == 0)
    }

    @Test func aCursorScopedPlanTouchesOnlyTheChangedName() async {
        let bench = bench(["a.png", "b.png", "c.png"], label: "scoped")
        let library = bench.library()
        await library.syncAllLocally()
        let downloads = bench.source.callCount("mediaFile")
        let probes = bench.source.boundedCheckpointRequests.count
        bench.source.prints = [versionedPrint("a.png"), versionedPrint("b.png", favorite: true), versionedPrint("c.png")]
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.source.boundedCheckpointRequests.count == probes + 1)
        #expect(bench.source.boundedCheckpointRequests.last == ["b.png"])
        #expect(bench.source.callCount("mediaFile") == downloads, "a sealed, unchanged output is not downloaded again")
        #expect(bench.target.importedNames.count == 3)
        #expect(library.localSaveReport.contains("3 were already here"))
    }

    @Test func aDayOldFullPlanMakesTheNextAutomaticRunPlanEverything() async {
        let bench = bench(["a.png"], label: "stale")
        let library = bench.library()
        await library.syncAllLocally()
        let whole = bench.source.callCount("gallerySyncCheckpoint")
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.source.callCount("gallerySyncCheckpoint") == whole)
        var aged = LibrarySyncRecords.RunOutcome()
        aged.fullPlan = Date(timeIntervalSinceNow: -LibraryStore.fullPlanInterval - 60)
        await bench.records.finishRun(aged, instance: bench.instance)
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.source.callCount("gallerySyncCheckpoint") == whole + 1)
        #expect(bench.records.bucket(bench.instance).lastFullPlan.map { Date().timeIntervalSince($0) < 60 } == true)
    }

    @Test func aFailedItemIsRetriedByTheNextCursorScopedRun() async {
        let bench = bench(["a.png", "b.png"], label: "retry")
        bench.target.importFailures = ["b.png"]
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.target.importedNames == ["a.png"])
        #expect(bench.records.bucket(bench.instance).retry.contains(bench.key("b.png")))
        bench.target.importFailures = []
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.target.importedNames.sorted() == ["a.png", "b.png"])
        #expect(!bench.records.bucket(bench.instance).retry.contains(bench.key("b.png")))
        #expect(library.localSaveFailures.isEmpty)
    }

    @Test func aStoppedRunKeepsItsCursorsAndTheNextRunPlansAgain() async {
        let bench = bench(["a.png", "b.png"], label: "stopped")
        let library = bench.library()
        await library.syncAllLocally()
        let cursors = bench.records.bucket(bench.instance).cursors
        let destination = bench.records.bucket(bench.instance).destinationCursor
        bench.source.prints = [versionedPrint("a.png"), versionedPrint("b.png", favorite: true)]
        bench.source.delays["boundedSyncCheckpoint"] = .milliseconds(150)
        let run = Task { await library.syncAllLocally(fullPlan: false) }
        await settle { bench.source.callCount("boundedSyncCheckpoint") > 0 && library.localSaveRunning }
        library.localSaveStopRequested = true
        await run.value
        #expect(bench.records.bucket(bench.instance).cursors == cursors)
        #expect(bench.records.bucket(bench.instance).destinationCursor == destination)
        bench.source.delays["boundedSyncCheckpoint"] = nil
        let probes = bench.source.boundedCheckpointRequests.count
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.source.boundedCheckpointRequests.count > probes)
        #expect(bench.source.boundedCheckpointRequests.last == ["b.png"])
    }

    @Test func copiesRemovedOrTrashedHereStayRemovedUnderCursorPlans() async {
        let bench = bench(["gone.png", "trashed.png", "kept.png"], label: "removed")
        let library = bench.library()
        await library.syncAllLocally()
        bench.target.prints.removeAll { $0.filename == "gone.png" }
        let trashed = bench.target.prints.filter { $0.filename == "trashed.png" }
        bench.target.prints.removeAll { $0.filename == "trashed.png" }
        bench.target.trashedRows = trashed.map { FakeFixtures.trashState($0, at: 1) }
        for _ in 0..<2 {
            await library.syncAllLocally(fullPlan: false)
            #expect(bench.target.importedNames.count == 3, "a removed copy is never recreated")
            #expect(library.localSaveReport.contains("2 copies were kept removed"))
            #expect(library.localSaveReport.contains("1 was already here"))
        }
    }

    @Test func aForeignChangeAcrossTheDestinationCursorIsOwedALook() async {
        let bench = bench(["a.png", "b.png"], label: "foreign")
        let library = bench.library()
        await library.syncAllLocally()
        // Mid-run, while This Mac imports a new print, another client edits
        // an existing copy here -- after the run read its cursor.
        let newcomer = versionedPrint("c.png")
        bench.source.prints.append(newcomer)
        let target = bench.target
        bench.target.listsImports = { name in
            if let index = target.prints.firstIndex(where: { $0.filename == "a.png" }) {
                var edited = GalleryPrint.Mutable(target.prints[index])
                edited.title = "edited here"
                target.prints[index] = edited.build()
            }
            return name == "c.png" ? newcomer : nil
        }
        bench.target.syncCheckpoint?["c.png"] = "local-c.png"
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.target.importedNames.contains("c.png"))
        #expect(bench.records.bucket(bench.instance).retry.contains(bench.key("a.png")))
        let asked = bench.target.boundedCheckpointRequests.count
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.target.boundedCheckpointRequests.dropFirst(asked).contains { $0.contains("a.png") })
    }

    @Test func aMachineThatCouldNotBeReadStillGetsThisMacsChangesToItsCopies() async {
        let bench = bench(["mine.png"], label: "offline")
        let library = bench.library()
        await library.syncAllLocally()
        // That machine is unreachable while its copy is removed here...
        bench.source.plantedErrors["gallery"] = MoldClientError.unreachable("offline")
        bench.target.prints.removeAll { $0.filename == "mine.png" }
        await library.syncAllLocally(fullPlan: false)
        #expect(bench.records.bucket(bench.instance).retry.contains(bench.key("mine.png")))
        // ...and back: the removal is noticed, never recreated.
        bench.source.plantedErrors["gallery"] = nil
        await library.syncAllLocally(fullPlan: false)
        #expect(library.localSaveReport.contains("1 copies were kept removed"))
        #expect(bench.target.importedNames == ["mine.png"])
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
        let records = bench.records.records(bench.instance)
        #expect(records.values.allSatisfy { $0.sealed })
        let target = perPrintCalls(bench.target)
        await library.syncAllLocally(fullPlan: false)
        #expect(perPrintCalls(bench.target) == target)
    }

    @Test func aReceiptFromAnEarlierRunLinksOnlyThroughByteComparison() async throws {
        let bench = bench(["landed.png"], label: "receipt")
        // A crash right after an import answered: the copy is listed here,
        // with only a receipt naming it.
        let print = versionedPrint("landed.png")
        bench.target.prints = [print]
        bench.target.mediaAnswers["landed.png"] = Data([1, 2, 3])
        let receipt = try #require(LocalSyncRecord(receipt: print, from: bench.remote.id, as: "landed.png",
                                                   destinationInstance: bench.instance))
        try await bench.records.checkpoint(receipt, key: bench.key("landed.png"), instance: bench.instance)
        let library = bench.library()
        await library.syncAllLocally()
        // The copy's bytes were compared before it was linked; nothing was
        // imported again.
        #expect(bench.target.callCount("mediaFile") == 1)
        #expect(bench.source.callCount("mediaFile") == 1)
        #expect(bench.target.importedNames.isEmpty)
        #expect(library.localSaveReport.contains("1 was already here"))
        #expect(bench.records.records(bench.instance)[bench.key("landed.png")]?.sealed == true)
    }

    @Test func aReceiptThatCannotBeSavedFailsTheItemBeforeAnythingMoves() async throws {
        // A directory path that is a file: nothing can be journaled there.
        let blocked = FileManager.default.temporaryDirectory.appending(path: "mold-sync-unwritable-\(UUID().uuidString)")
        try Data().write(to: blocked)
        defer { try? FileManager.default.removeItem(at: blocked) }
        let bench = bench(["a.png"], label: "unwritable", records: LibrarySyncRecords(directory: blocked))
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.target.importedNames.isEmpty)
        #expect(bench.source.callCount("retainedMediaTransferOffer") <= 1, "no retained input moved after the failure")
        #expect(library.localSaveFailures.contains { $0.contains("a.png") })
    }

    @Test(arguments: ["{not json", #"{"version":99,"seq":0,"buckets":{}}"#])
    func unreadableRecordsPauseSyncAllInsteadOfRecreatingRemovedCopies(_ contents: String) async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "mold-sync-unreadable-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(contents.utf8).write(to: directory.appending(path: "records.json"))
        let bench = bench(["removed.png"], label: "unreadable", records: LibrarySyncRecords(directory: directory))
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.target.importedNames.isEmpty)
        #expect(bench.source.callCount("mediaFile") == 0)
        #expect(library.localSaveFailures.contains { $0.contains("could not be read") })
        // Set aside, never overwritten -- and still paused after a relaunch.
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        #expect(!names.contains("records.json"))
        #expect(names.contains { $0.hasPrefix("records.json.unreadable-") })
        let relaunched = LibraryStore(hosts: bench.hosts, syncRecords: LibrarySyncRecords(directory: directory))
        await relaunched.syncAllLocally()
        #expect(bench.target.importedNames.isEmpty)
        #expect(!(try FileManager.default.contentsOfDirectory(atPath: directory.path)).contains("records.json"))
    }

    @Test func aRecordIsPrunedOnlyWhenBothSidesAreGone() async {
        let bench = bench(["kept.png", "gone.png"], label: "tombstone")
        let library = bench.library()
        await library.syncAllLocally()
        #expect(bench.records.records(bench.instance).count == 2)
        // Removed here only: the record is what keeps it removed.
        bench.target.prints.removeAll { $0.filename == "kept.png" }
        // Removed on both sides, Trash included: nothing can bring it back.
        bench.source.prints.removeAll { $0.filename == "gone.png" }
        bench.target.prints.removeAll { $0.filename == "gone.png" }
        await library.syncAllLocally(fullPlan: false)
        let records = bench.records.records(bench.instance)
        #expect(records.keys.contains(bench.key("kept.png")))
        #expect(!records.keys.contains(bench.key("gone.png")))
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

    @Test func aCancelledVisibilityWaiterLeavesTheQueue() async {
        let lock = VisibilityWriterLock()
        #expect(lock.tryAcquire())
        let waiter = Task { try await lock.acquire() }
        await settle { lock.waiting == 1 }
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(lock.waiting == 0)
        lock.release()
        #expect(!lock.isHeld)
    }
}
