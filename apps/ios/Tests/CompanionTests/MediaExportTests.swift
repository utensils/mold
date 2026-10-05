import CryptoKit
import Foundation
import MoldClient
import MoldClientTesting
import Photos
import Testing
@testable import MoldCompanion

@MainActor struct MediaExportTests {
    @Test func cancellationDuringPhotosAuthorizationCannotPublishLateDenial() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let filename = "cancel-auth-\(UUID()).mp4"
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            "{\"filename\":\"\(filename)\",\"metadata\":{},\"timestamp\":1790000000}")
        let bytes = Data(base64Encoded: "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")!
        fake.stub("exportVideo(_:request:)", returning: bytes)
        let actions = PrintActions(hosts: hosts)
        var pending: CheckedContinuation<PHAuthorizationStatus, Never>?
        defer { pending?.resume(returning: .denied) }
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions,
            requestPhotosAccess: { await withCheckedContinuation { pending = $0 } })
        session.options = try QueueStoreTests.decode(ExportOptions.self, #"{"formats":["gif"]}"#)
        session.loading = false; session.destination = .photos
        actions.sheet = .export(session)
        session.submit()
        for _ in 0..<100 where pending == nil { try await Task.sleep(for: .milliseconds(10)) }
        let authorization = try #require(pending)
        let output = VideoExportRequest.filename(filename, format: "gif")
        let directories = try FileManager.default.contentsOfDirectory(at: FileManager.default.temporaryDirectory,
            includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("mold-print-export-") }
        let staged = try #require(directories.map { $0.appending(path: output) }
            .first { FileManager.default.fileExists(atPath: $0.path) })
        actions.cancelExports()
        pending = nil; authorization.resume(returning: .denied)
        for _ in 0..<100 where FileManager.default.fileExists(atPath: staged.path) {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(actions.permissionRecovery == nil)
        #expect(session.error == nil && actions.status == nil)
        #expect(actions.pendingDelivery == nil && actions.activeExportID == nil && !actions.busy)
        #expect(!FileManager.default.fileExists(atPath: staged.deletingLastPathComponent().path))
    }

    @Test func deepLinkPresentsTheRequestedCopyForExport() throws {
        let print = try QueueStoreTests.decode(GalleryPrint.self, #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        let first = MoldHost(name: "First", baseURL: URL(string: "http://first.test")!)
        let second = MoldHost(name: "Second", baseURL: URL(string: "http://second.test")!)
        var merged = LibraryEntry(host: first, print: print)
        let copy = LibraryEntry(host: second, print: print)
        merged.copies = [copy]
        #expect(LinkedPrint.presentedEntry(in: [merged], id: copy.id)?.hostID == second.id)
        #expect(LinkedPrint.presentedEntry(in: [merged], id: copy.id)?.everyCopy.count == 2)
        #expect(ResultPager.presentedEntry(in: [merged], host: second.id, filename: print.filename)?.hostID == second.id)
    }

    @Test func closingOptionsPreservesPendingDeliveryUntilItsOwnDismissal() async throws {
        let (_, hosts, _) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000,"format":"mp4"}"#)
        let actions = PrintActions(hosts: hosts)
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions)
        let directory = FileManager.default.temporaryDirectory.appending(path: "mold-print-export-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appending(path: "loop.gif")
        try Data("staged".utf8).write(to: url)
        actions.presentedSheet = .export(session)
        actions.pendingDelivery = .share([url])
        actions.presentationDismissed()
        #expect(FileManager.default.fileExists(atPath: url.path))
        guard case .share = actions.sheet else { Issue.record("Pending delivery missing"); return }
        actions.presentedSheet = actions.sheet
        actions.sheet = nil
        actions.presentationDismissed()
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test func persistentFolderCopiesNeverOverwriteAndSurviveCleanup() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "export-test-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appending(path: "loop.gif")
        try Data("fixture".utf8).write(to: source)
        let first = try ExportFiles.saveToFolder(source, root: root)
        let second = try ExportFiles.saveToFolder(source, root: root)
        #expect(first.lastPathComponent == "loop.gif")
        #expect(second.lastPathComponent == "loop (2).gif")
        PrintActions.removeFiles([first, second])
        #expect(FileManager.default.fileExists(atPath: first.path))
        #expect(FileManager.default.fileExists(atPath: second.path))
    }

    @Test func cancelledOptionsCannotPublishLateCapabilities() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000,"format":"mp4"}"#)
        let options = try QueueStoreTests.decode(ExportOptions.self, #"{"formats":["gif"]}"#)
        fake.stub("exportOptions()") { _ in try await Task.sleep(for: .milliseconds(100)); return options }
        let actions = PrintActions(hosts: hosts)
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions)
        session.load(); session.cancel()
        try await Task.sleep(for: .milliseconds(150))
        #expect(session.options == nil)
        #expect(actions.sheet == nil)
    }
    @Test func unknownAdvertisedGifChoicesCannotBeSubmitted() async throws {
        let (_, hosts, _) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        let actions = PrintActions(hosts: hosts)
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions)
        session.options = try QueueStoreTests.decode(ExportOptions.self,
            #"{"formats":["gif","apng"],"gif_playback":["future"],"gif_repeat":[]}"#)
        session.loading = false
        #expect(!session.valid)
        session.format = "apng"
        #expect(session.valid)
    }

    @Test func assetBytesMustMatchTheirAdvertisedDigest() throws {
        let asset = try QueueStoreTests.decode(GenerationAsset.self,
            #"{"asset_id":"base-color","role":"future_map","display_name":"base-color.png","media_type":"image/png","size_bytes":3,"sha256":"wrong"}"#)
        #expect(throws: MoldClientError.self) { try ExportFiles.stage(Data([1, 2, 3]), filename: asset.displayName, asset: asset) }
        #expect(throws: MoldClientError.self) { try ExportFiles.stage(Data([1]), filename: "../escape.png") }
        #expect(throws: MoldClientError.self) { try ExportFiles.stage(Data("<html>error</html>".utf8), filename: "loop.gif") }
    }

    @Test func sceneCancellationCannotPresentALateAssetDownload() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"object.glb","metadata":{},"timestamp":1790000000}"#)
        let bytes = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let asset = try QueueStoreTests.decode(GenerationAsset.self,
            "{\"asset_id\":\"base-color\",\"role\":\"base_color\",\"display_name\":\"base-color.png\",\"media_type\":\"image/png\",\"size_bytes\":\(bytes.count),\"sha256\":\"\(digest)\"}")
        fake.stub("generationAsset(_:assetID:)") { _ in try? await Task.sleep(for: .milliseconds(100)); return bytes }
        let actions = PrintActions(hosts: hosts)
        actions.deliverAsset(asset, entry: LibraryEntry(host: hosts.hosts[0], print: print), destination: .share)
        actions.cancelExports()
        try await Task.sleep(for: .milliseconds(200))
        #expect(actions.sheet == nil)
        #expect(!actions.busy)
        if case let .share(urls) = actions.sheet { PrintActions.removeFiles(urls) }
    }

    @Test func cancellingConversionSuppressesLateDelivery() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self, #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        let bytes = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        fake.stub("exportVideo(_:request:)") { _ in try? await Task.sleep(for: .milliseconds(100)); return bytes }
        let actions = PrintActions(hosts: hosts)
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions)
        session.options = try QueueStoreTests.decode(ExportOptions.self, #"{"formats":["apng"]}"#)
        session.format = "apng"; session.loading = false
        session.submit()
        try await Task.sleep(for: .milliseconds(20))
        session.cancel()
        try await Task.sleep(for: .milliseconds(150))
        #expect(actions.pendingDelivery == nil)
        #expect(actions.status == nil)
        #expect(!actions.busy && !session.converting)
    }

    @Test func disconnectedHostRetainsChoicesForRetry() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self, #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        fake.stub("exportVideo(_:request:)") { _ in throw URLError(.notConnectedToInternet) }
        let actions = PrintActions(hosts: hosts)
        let session = MediaExportSession(entry: LibraryEntry(host: hosts.hosts[0], print: print), actions: actions)
        session.options = try QueueStoreTests.decode(ExportOptions.self, #"{"formats":["gif"],"gif_pause":{"min":0,"max":5000,"step":10,"default":0}}"#)
        session.loading = false; session.playback = .bounce; session.pauseText = "250"
        session.submit()
        try await Task.sleep(for: .milliseconds(100))
        #expect(session.error != nil)
        #expect(session.pauseText == "250" && session.playback == .bounce)
        #expect(session.valid && !actions.busy)
        #expect(actions.pendingDelivery == nil)
    }

}
