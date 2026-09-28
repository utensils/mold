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
    private func temp(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appending(path: "\(name)-\(UUID())")
    }

    private func print(_ file: String, version: String? = nil) throws -> GalleryPrint {
        let versionField = version.map { #","media_version":"\#($0)""# } ?? ""
        let json = #"{"filename":"\#(file)","metadata":{"prompt":"an owl","seed":7,"model":"flux-dev:q4","width":64,"height":64},"#
            + #""timestamp":1790000000,"size_bytes":1234\#(versionField),"favorite":true,"tags":["owls"]}"#
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
        #expect(ThumbnailLoader.version(try print("a.png", version: "v7")) == "v7")
        #expect(ThumbnailLoader.version(try print("a.png")) == "t1790000000-1234")
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
        #expect(library.offlineHosts.map(\.name) == ["hal9000"])
    }

    /// A fresh listing is saved for next time.
    @Test func aFreshListingIsSavedForNextTime() async throws {
        let fake = FakeBackend()
        fake.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try print("new.png", version: "v2")], etag: "e2"))
        let hosts = try await hosts(fake, up: true)
        let snapshots = LibrarySnapshots(directory: temp("snapshots"))
        let library = LibraryStore(hosts: hosts, snapshots: snapshots)
        await library.reload()
        let id = hosts.hosts[0].id
        for _ in 0..<50 where snapshots.load(id) == nil { try await Task.sleep(for: .milliseconds(20)) }
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
        await loader.disk.store(png(), for: ThumbnailLoader.diskKey(entry, size: 256))
        await loader.originals.store(png(), for: ThumbnailLoader.diskKey(entry, size: 0))
        #expect(await loader.image(for: entry, pixels: 200) != nil)
        #expect(await loader.original(for: entry) != nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 0)
        #expect(fake.count("media(_:trashed:)") == 0)
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
        await loader.disk.store(image, for: ThumbnailLoader.diskKey(saved, size: 512))
        loader.save([saved, fresh])
        for _ in 0..<100 where loader.saving != nil { try await Task.sleep(for: .milliseconds(20)) }
        #expect(loader.saving == nil)
        #expect(fake.count("thumbnail(_:size:trashed:)") == 1)
        #expect(await loader.disk.contains(ThumbnailLoader.diskKey(fresh, size: 512)))
    }

    /// Lowering the limit applies at once.
    @Test func theStorageLimitSplitsBetweenThumbnailsAndOpenedPrints() {
        #expect(OfflineLimit.gb1.thumbnailBytes == 300_000_000)
        #expect(OfflineLimit.gb1.originalBytes == 700_000_000)
        #expect(OfflineLimit.current(UserDefaults(suiteName: "offline-\(UUID())")!) == .standard)
    }
}
