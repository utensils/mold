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

    @Test func aSavedListingReadsBackExactly() throws {
        let store = LibrarySnapshots(directory: temp("snapshots"))
        let id = UUID()
        let snapshot = LibrarySnapshot(prints: [try print("a.png", version: "v1"), try print("b.png")],
                                       trashed: [], collections: nil, etag: "e1", trashEtag: nil)
        store.save(snapshot, for: id)
        #expect(store.load(id) == snapshot)
        store.remove(id)
        #expect(store.load(id) == nil)
    }

    @Test func clearingOfflineLibraryRemovesSavedListingsAndOfflineRows() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let id = hosts.hosts[0].id
        snapshots.save(LibrarySnapshot(prints: [try print("owl.png", version: "v1")], trashed: nil,
                                       collections: nil, etag: "e1", trashEtag: nil), for: id)
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

    @Test func aRestoreStartedBeforeClearCannotBringOfflineRowsBack() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        let saved = LibrarySnapshot(prints: [try print("stale.png", version: "v1")], trashed: nil,
                                    collections: nil, etag: "old", trashEtag: nil)
        await library.clearOfflineCache()
        library.acceptRestored([(hosts.hosts[0].id, saved)], from: 0)
        #expect(library.pool.isEmpty)
        #expect(library.knownTags.isEmpty)
    }

    /// With the machine down, the Library still shows its saved prints and
    /// says it is doing so.
    @Test func aMachineThatIsDownStillShowsItsSavedPrints() async throws {
        let fake = FakeBackend()
        let hosts = try await hosts(fake, up: false)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        snapshots.save(LibrarySnapshot(prints: [try print("owl.png", version: "v1")], trashed: nil,
                                       collections: nil, etag: "e1", trashEtag: nil), for: hosts.hosts[0].id)
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
        snapshots.save(LibrarySnapshot(prints: [try print("owl.png", version: "v1")], trashed: nil,
                                       collections: nil, etag: nil, trashEtag: nil), for: hosts.hosts[0].id)
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
        #expect(snapshots.load(id)?.prints.map(\.filename) == ["new.png"])
        #expect(snapshots.load(id)?.etag == "e2")
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
        await loader.originals.store(png(), for: try #require(ThumbnailLoader.diskKey(entry, size: 0)))
        #expect(await loader.image(for: entry, pixels: 200) != nil)
        #expect(await loader.original(for: entry) != nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 0)
        #expect(fake.count("media(_:trashed:)") == 0)
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
