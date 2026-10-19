import Foundation
import ImageIO
import MoldClient
import MoldClientTesting
import Testing
import UIKit
import UniformTypeIdentifiers

@testable import MoldCompanion

/// The Library's speed and its offline copy: thumbnails only in the sizes
/// machines serve, a saved listing that shows before (and without) any
/// machine, pictures read from disk with no connection, and a pinch that
/// walks the tile sizes.
@MainActor
struct OfflineLibraryTests {
    private actor CancelledThumbnail {
        var cancellations = 0
        func load() async throws -> Data {
            do { try await Task.sleep(for: .seconds(30)); return Data() }
            catch { cancellations += 1; throw error }
        }
    }

    @Test func cancellingEveryCoalescedConsumerCancelsTheDownload() async throws {
        let fake = FakeBackend()
        let cancellation = CancelledThumbnail()
        fake.stub("thumbnail(_:size:trashed:)") { _ in try await cancellation.load() }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("cancel-waiters"), limit: .mb250)
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("shared.png", version: "v1"))
        let first = Task { await loader.image(for: entry, pixels: 256) }
        for _ in 0..<100 where fake.count("thumbnail(_:size:trashed:)") == 0 { try await Task.sleep(for: .milliseconds(5)) }
        var secondStarted = false
        let second = Task { secondStarted = true; return await loader.image(for: entry, pixels: 256) }
        while !secondStarted { await Task.yield() }
        first.cancel(); second.cancel()
        for _ in 0..<100 where await cancellation.cancellations == 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await cancellation.cancellations == 1)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 1)
        await loader.emptyCaches()
        #expect(await first.value == nil)
        #expect(await second.value == nil)
    }

    @Test func cancellingOneConsumerKeepsTheOtherDownloadAlive() async throws {
        let fake = FakeBackend(), gate = ThumbnailGate(), bytes = png()
        fake.stub("thumbnail(_:size:trashed:)") { _ in await gate.pause(); try Task.checkCancellation(); return bytes }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("surviving-waiter"), limit: .mb250)
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("shared.png", version: "v1"))
        let first = Task { await loader.image(for: entry, pixels: 256) }
        while await gate.count() == 0 { await Task.yield() }
        var secondStarted = false
        let second = Task { secondStarted = true; return await loader.image(for: entry, pixels: 256) }
        while !secondStarted { await Task.yield() }
        first.cancel()
        await gate.open()
        #expect(await first.value == nil)
        #expect(await second.value != nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 1)
        await loader.emptyCaches()
    }

    private actor ThumbnailGate {
        private var arrivals = 0
        private var waiting: [CheckedContinuation<Void, Never>] = []

        func pause() async {
            arrivals += 1
            await withCheckedContinuation { waiting.append($0) }
        }

        func count() -> Int { arrivals }

        func open() {
            for continuation in waiting { continuation.resume() }
            waiting.removeAll()
        }
    }

    private func temp(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID())")
    }

    private func print(_ file: String, version: String? = nil, sized: Bool = true) throws -> GalleryPrint {
        let versionField = version.map { #","media_version":"\#($0)""# } ?? ""
        let sizeField = sized ? #","size_bytes":1234"# : ""
        let json = #"{"filename":"\#(file)","metadata":{"prompt":"an owl","seed":7,"model":"flux-dev:q4","width":64,"height":64},"#
            + #""timestamp":1790000000\#(sizeField)\#(versionField),"favorite":true,"tags":["owls"]}"#
        return try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
    }

    private func png() -> Data {
        let context = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(UIColor.red.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: 8, height: 8))
        let data = NSMutableData()
        let out = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(out, context.makeImage()!, nil)
        CGImageDestinationFinalize(out)
        return data as Data
    }

    private func hosts(_ fake: FakeBackend, up: Bool) async throws -> HostStore {
        if up {
            fake.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self, from: Data(
                #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#.utf8)))
            fake.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8)))
            fake.stub("models()", returning: [Model]())
        } else {
            fake.stub("status()", throwing: MoldClientError.unreachable("The request timed out."))
        }
        let file = HostListFile(url: temp("hosts").appending(path: "h.json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        try hosts.add(name: "hal9000", address: "10.0.0.9", apiKey: nil, makeDefault: true)
        await hosts.refreshAll()
        return hosts
    }

    // MARK: - Thumbnail sizes

    /// Machines serve 256 and 512 only; asking for 1024 got a 422 and an empty
    /// tile -- which is what a Medium tile on a 3× iPhone asked for.
    @Test func everyRequestIsASizeMachinesServe() {
        #expect(ThumbnailLoader.bucket(120) == 256)
        #expect(ThumbnailLoader.bucket(256) == 256)
        #expect(ThumbnailLoader.bucket(420) == 512)
        #expect(ThumbnailLoader.bucket(537) == 512)
        #expect(ThumbnailLoader.bucket(3_000) == 512)
    }

    /// A machine with no `media_version` still gets a stable, change-aware
    /// disk key, so its library is kept offline too.
    @Test func aPrintWithoutAMediaVersionStillHasADiskKey() throws {
        #expect(ThumbnailLoader.diskVersion(try print("a.png", version: "v7")) == "v7")
        #expect(ThumbnailLoader.diskVersion(try print("a.png")) == "t1790000000-1234")
    }

    /// With neither a version nor a size nothing tells a re-render from the
    /// old print: memory only, never disk.
    @Test func aPrintWithNothingToVersionItIsNeverStored() throws {
        let host = MoldHost(name: "old", baseURL: URL(string: "http://old:7680")!)
        let entry = LibraryEntry(host: host, print: try print("a.png", sized: false))
        #expect(ThumbnailLoader.diskKey(entry, size: 256) == nil)
    }

    @Test func unlimitedStoragePersistsWithoutOverflow() throws {
        let defaults = try #require(UserDefaults(suiteName: "offline-limit-test-\(UUID())"))
        defaults.set(OfflineLimit.unlimited.rawValue, forKey: Preference.offlineLimit)
        #expect(OfflineLimit.current(defaults) == .unlimited)
        #expect(OfflineLimit.unlimited.thumbnailBytes == Int.max)
        #expect(OfflineLimit.unlimited.originalBytes == Int.max)
    }

    // MARK: - Pinch

    @Test func aPinchWalksTheSizesLiveAndStopsAtTheEnds() {
        #expect(TileSize.pinched(from: .medium, magnification: 1.1) == .medium)
        #expect(TileSize.pinched(from: .medium, magnification: 1.4) == .large)
        #expect(TileSize.pinched(from: .medium, magnification: 2.0) == .huge)
        #expect(TileSize.pinched(from: .medium, magnification: 0.7) == .small)
        #expect(TileSize.pinched(from: .medium, magnification: 0.2) == .tiny)
        #expect(TileSize.pinched(from: .huge, magnification: 5) == .huge)
    }

    /// Fingers resting on a boundary do not flip between two sizes.
    @Test func aPinchRestingOnABoundaryDoesNotFlicker() {
        // One step is at 1.35; just past it the size changed to Large...
        #expect(TileSize.pinched(from: .medium, magnification: 1.36, current: .medium) == .medium)
        #expect(TileSize.pinched(from: .medium, magnification: 1.4, current: .medium) == .large)
        // ...and drifting just back under it keeps Large.
        #expect(TileSize.pinched(from: .medium, magnification: 1.33, current: .large) == .large)
        #expect(TileSize.pinched(from: .medium, magnification: 0.97, current: .large) == .medium)
    }

    // MARK: - Saved listing

    /// What builds before replicas saved: the whole listing, beside the
    /// collections. Written as such a build wrote it.
    private func saveLegacy(_ snapshots: LibrarySnapshots, _ prints: [GalleryPrint], etag: String? = "e1",
                            collections: [Collection]? = nil, for id: UUID) throws {
        let legacy = LibrarySnapshot.Legacy(prints: prints, trashed: nil, collections: collections,
                                            etag: etag, trashEtag: nil)
        try FileManager.default.createDirectory(at: snapshots.directory, withIntermediateDirectories: true)
        try MoldJSON.localEncoder.encode(legacy).write(to: snapshots.directory.appending(path: "\(id.uuidString).json"))
    }

    @Test func savedCollectionsReadBackExactly() throws {
        let store = LibrarySnapshots(directory: temp("snapshots"))
        let id = UUID()
        let snapshot = LibrarySnapshot(collections: [Collection(id: "c1", name: "Owls", slug: "owls")])
        store.save(snapshot, for: id)
        #expect(store.load(id) == snapshot)
        store.remove(id)
        #expect(store.load(id) == nil)
    }

    /// An update keeps the offline Library: the old listing becomes the
    /// replica's saved state once, and the file keeps only its collections.
    @Test func anOlderBuildsSavedListingMovesIntoTheReplicaOnce() async throws {
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let id = UUID()
        let shelf = Collection(id: "c1", name: "Owls", slug: "owls")
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], collections: [shelf], for: id)
        snapshots.migrate(id, from: snapshots.generation)
        #expect(snapshots.load(id) == LibrarySnapshot(collections: [shelf]))
        let restored = try #require(await GalleryReplica(hostID: id, store: snapshots.replicas).restoreSaved())
        #expect(restored.live.map(\.filename) == ["owl.png"])
        // Already migrated: a second pass changes nothing.
        snapshots.migrate(id, from: snapshots.generation)
        #expect(snapshots.load(id) == LibrarySnapshot(collections: [shelf]))
    }

    /// A seed that could not be written leaves the old listing where it is:
    /// the only copy of those rows is never overwritten.
    @Test func aFailedSeedKeepsTheOldListing() throws {
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let id = UUID()
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], for: id)
        // A file where the replicas directory should be: the seed cannot write.
        try Data().write(to: snapshots.replicas.directory)
        snapshots.migrate(id, from: snapshots.generation)
        let raw = try Data(contentsOf: snapshots.directory.appending(path: "\(id.uuidString).json"))
        #expect(try MoldJSON.localDecoder.decode(LibrarySnapshot.Legacy.self, from: raw).prints.map(\.filename)
                == ["owl.png"])
    }

    /// A migration queued before Clear Cache purges writes nothing after it.
    @Test func aMigrationFromBeforeAPurgeWritesNothing() throws {
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let id = UUID()
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], for: id)
        let before = snapshots.generation
        snapshots.purge()
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], for: id)
        snapshots.migrate(id, from: before)
        #expect(!snapshots.replicas.hasSaved(id))
    }

    @Test func clearingOfflineLibraryRemovesSavedListingsAndOfflineRows() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let id = hosts.hosts[0].id
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], for: id)
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        await library.restoreSaved()
        #expect(library.pool.count == 1)
        #expect(snapshots.diskBytes > 0)
        await library.clearOfflineCache()
        #expect(snapshots.load(id) == nil)
        #expect(snapshots.diskBytes == 0)
        #expect(library.pool.isEmpty)
        #expect(library.knownTags.isEmpty)
    }

    @Test func aRestoreRacingClearCannotBringOfflineRowsBack() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        try saveLegacy(snapshots, [try print("stale.png", version: "v1")], etag: "old", for: hosts.hosts[0].id)
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        async let restoring: Void = library.restoreSaved()
        async let clearing: Void = library.clearOfflineCache()
        _ = await (restoring, clearing)
        #expect(library.pool.isEmpty)
        #expect(library.knownTags.isEmpty)
        await library.restoreSaved()
        #expect(library.pool.isEmpty, "nothing saved survived the clear")
    }

    /// With the machine down, the Library still shows its saved prints and
    /// says it is doing so.
    @Test func aMachineThatIsDownStillShowsItsSavedPrints() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], for: hosts.hosts[0].id)
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        await library.restoreSaved()
        await library.reload()
        #expect(library.pool.map(\.print.filename) == ["owl.png"])
        #expect(library.knownTags == ["owls"])
        #expect(library.offlineHosts.map(\.name) == ["hal9000"])
    }

    /// Before a machine has been asked (launch, return to the app) it is not
    /// "down": no note claiming it is.
    @Test func aMachineNotYetAskedIsNotCalledOffline() async throws {
        let fake = FakeBackend()
        // Still being asked when the saved library appears.
        fake.stub("status()") { _ in
            try await Task.sleep(for: .seconds(30))
            throw CancellationError()
        }
        let file = HostListFile(url: temp("hosts").appending(path: "h.json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        try hosts.add(name: "hal9000", address: "10.0.0.9", apiKey: nil, makeDefault: true)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        try saveLegacy(snapshots, [try print("owl.png", version: "v1")], etag: nil, for: hosts.hosts[0].id)
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        await library.restoreSaved()
        #expect(library.pool.count == 1)
        #expect(library.offlineHosts.isEmpty)
    }

    /// A fresh listing is saved for next time.
    @Test func aFreshListingIsSavedForNextTime() async throws {
        let fake = FakeBackend()
        fake.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try print("new.png", version: "v2")], etag: "e2"))
        let hosts = try await hosts(fake, up: true)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        await library.reload()
        #expect(library.pool.map(\.print.filename) == ["new.png"])
        #expect(fake.count("gallery(etag:)") == 1)
        await library.waitForSnapshotWrites()
        let id = hosts.hosts[0].id
        let saved = try #require(await GalleryReplica(hostID: id, store: snapshots.replicas).restoreSaved())
        #expect(saved.live.map(\.filename) == ["new.png"])
        #expect(library.offlineHosts.isEmpty)
    }

    // MARK: - Pictures offline

    /// Saved thumbnails and opened prints come from disk with no machine at all.
    @Test func savedPicturesShowWithNoConnection() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("owl.png", version: "v1"))
        await loader.disk.store(png(), for: try #require(ThumbnailLoader.diskKey(entry, size: 256)))
        let originalBytes = png()
        fake.stub("mediaFile(_:trashed:)") { _ in
            let file = FileManager.default.temporaryDirectory.appending(path: "test-original-\(UUID())")
            try originalBytes.write(to: file)
            return file
        }
        let opened = try await loader.originalFile(for: entry)
        try FileManager.default.removeItem(at: opened)
        fake.stub("mediaFile(_:trashed:)") { _ in throw CancellationError() }
        #expect(await loader.image(for: entry, pixels: 200) != nil)
        #expect(await loader.original(for: entry) != nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 0)
        #expect(fake.count("media(_:trashed:)") == 0)
        let directory = await loader.disk.directory.deletingLastPathComponent()
        let relaunched = ThumbnailLoader(hosts: hosts, directory: directory, limit: .unlimited)
        #expect(await relaunched.original(for: entry) != nil)
        let actions = PrintActions(hosts: hosts, thumbnails: relaunched)
        let exports = try #require(await actions.files(for: [entry]))
        defer { PrintActions.removeFiles(exports) }
        #expect(try Data(contentsOf: exports[0]) == originalBytes)
        #expect(fake.count("mediaFile(_:trashed:)") == 1)
    }

    @Test func viewedVideoSharesItsOriginalFromAFreshLoaderOffline() async throws {
        let fake = FakeBackend(), bytes = Data([0, 1, 2, 3])
        fake.stub("mediaFile(_:trashed:)") { _ in
            let file = FileManager.default.temporaryDirectory.appending(path: "test-video-\(UUID())")
            try bytes.write(to: file)
            return file
        }
        let hosts = try await hosts(fake, up: false), directory = temp("offline-video")
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("clip.mp4", version: "v1"))
        let loader = ThumbnailLoader(hosts: hosts, directory: directory, limit: .unlimited)
        let opened = try await loader.originalFile(for: entry)
        try FileManager.default.removeItem(at: opened)
        fake.stub("mediaFile(_:trashed:)") { _ in throw CancellationError() }
        let relaunched = ThumbnailLoader(hosts: hosts, directory: directory, limit: .unlimited)
        let localPlayback = try await relaunched.originalFile(for: entry)
        defer { try? FileManager.default.removeItem(at: localPlayback) }
        let actions = PrintActions(hosts: hosts, thumbnails: relaunched)
        let exports = try #require(await actions.files(for: [entry]))
        defer { PrintActions.removeFiles(exports) }
        #expect(exports[0].pathExtension == "mp4")
        #expect(try Data(contentsOf: exports[0]) == bytes)
        #expect(fake.count("mediaFile(_:trashed:)") == 1)
    }

    /// A 512 saved for offline draws a 256 tile: the smallest sizes are not
    /// blank offline just because the save kept the larger picture.
    @Test func eitherSavedSizeDrawsATileOffline() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("owl.png", version: "v1"))
        await loader.disk.store(png(), for: try #require(ThumbnailLoader.diskKey(entry, size: 512)))
        #expect(await loader.image(for: entry, pixels: 150) != nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 0)
    }

    /// A machine that is down is not asked at all.
    @Test func aMachineThatIsDownIsNotAsked() async throws {
        let fake = FakeBackend()
        fake.stub("thumbnail(_:size:trashed:)") { _ in Data() }
        let hosts = try await hosts(fake, up: false)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let entry = LibraryEntry(host: hosts.hosts[0], print: try print("new.png", version: "v1"))
        #expect(await loader.image(for: entry, pixels: 150) == nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 0)
    }

    /// Saving for offline fetches only what is not already on disk.
    @Test func savingSkipsWhatIsAlreadySaved() async throws {
        let fake = FakeBackend()
        let image = png()
        fake.stub("thumbnail(_:size:trashed:)") { _ in image }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let saved = LibraryEntry(host: hosts.hosts[0], print: try print("a.png", version: "v1"))
        let fresh = LibraryEntry(host: hosts.hosts[0], print: try print("b.png", version: "v1"))
        await loader.disk.store(image, for: try #require(ThumbnailLoader.diskKey(saved, size: 512)))
        loader.save([saved, fresh])
        // The UI says Checking: the denominator includes disk-cache hits,
        // while the network count below covers only actual downloads.
        #expect(loader.saving?.done == 0)
        #expect(loader.saving?.total == 2)
        for _ in 0..<100 where loader.saving != nil { try await Task.sleep(for: .milliseconds(20)) }
        #expect(loader.saving == nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 1)
        #expect(await loader.disk.contains(try #require(ThumbnailLoader.diskKey(fresh, size: 512))))
    }

    /// Stop, then Save All: the new run's progress survives the old run's end.
    @Test func aReplacedSaveKeepsTheNewRunsProgress() async throws {
        let fake = FakeBackend()
        let image = png()
        fake.stub("thumbnail(_:size:trashed:)") { _ in
            try await Task.sleep(for: .milliseconds(30))
            return image
        }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let entries = try (0..<12).map { LibraryEntry(host: hosts.hosts[0], print: try print("p\($0).png", version: "v1")) }
        loader.save(Array(entries.prefix(6)))
        loader.cancelSaving()
        loader.save(entries)
        try await Task.sleep(for: .milliseconds(60))
        #expect(loader.saving?.total == 12, "the old run's end did not clear the new run's progress")
        for _ in 0..<200 where loader.saving != nil { try await Task.sleep(for: .milliseconds(20)) }
        #expect(loader.saving == nil)
    }

    @Test func clearingWhileSavingLeavesNoThumbnailBehind() async throws {
        let fake = FakeBackend()
        let image = png()
        fake.stub("thumbnail(_:size:trashed:)") { _ in
            try await Task.sleep(for: .milliseconds(50))
            return image
        }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let entries = try (0..<8).map { LibraryEntry(host: hosts.hosts[0], print: try print("p\($0).png", version: "v1")) }
        loader.save(entries)
        await loader.emptyCaches()
        #expect(await loader.diskBytes() == 0)
        #expect(loader.saving == nil)
    }

    @Test func clearingWaitsForAReplacedSaveRun() async throws {
        let fake = FakeBackend()
        let gate = ThumbnailGate()
        let image = png()
        fake.stub("thumbnail(_:size:trashed:)") { _ in
            await gate.pause()
            return image
        }
        let hosts = try await hosts(fake, up: true)
        let loader = ThumbnailLoader(hosts: hosts, directory: temp("offline"), limit: .mb250)
        let first = LibraryEntry(host: hosts.hosts[0], print: try print("first.png", version: "v1"))
        let next = LibraryEntry(host: hosts.hosts[0], print: try print("next.png", version: "v1"))
        loader.save([first])
        for _ in 0..<100 where await gate.count() == 0 { try await Task.sleep(for: .milliseconds(10)) }
        #expect(await gate.count() > 0)
        loader.save([next])
        var cleared = false
        let clear = Task { @MainActor in
            await loader.emptyCaches()
            cleared = true
        }
        try await Task.sleep(for: .milliseconds(50))
        #expect(!cleared, "Clear must wait even for a cancelled predecessor in flight")
        await gate.open()
        await clear.value
        #expect(cleared)
        #expect(await loader.diskBytes() == 0)
    }

    /// Lowering the limit applies at once.
    @Test func theStorageLimitSplitsBetweenThumbnailsAndOpenedPrints() {
        #expect(OfflineLimit.gb1.thumbnailBytes == 300_000_000)
        #expect(OfflineLimit.gb1.originalBytes == 700_000_000)
        #expect(OfflineLimit.current(UserDefaults(suiteName: "offline-\(UUID())")!) == .standard)
    }
}
