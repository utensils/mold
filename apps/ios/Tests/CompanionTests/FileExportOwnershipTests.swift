import CryptoKit
import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor struct FileExportOwnershipTests {
    @MainActor private final class DeferredResponse<Value: Sendable> {
        var continuation: CheckedContinuation<Value, any Error>?
        func request() async throws -> Value {
            try await withCheckedThrowingContinuation { continuation = $0 }
        }
    }

    @Test func aCancelledAssetCannotReleaseANewerOriginalExport() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"object.glb","metadata":{},"timestamp":1790000000}"#)
        let bytes = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let asset = try QueueStoreTests.decode(GenerationAsset.self,
            "{\"asset_id\":\"base-color\",\"role\":\"base_color\",\"display_name\":\"base-color.png\",\"media_type\":\"image/png\",\"size_bytes\":\(bytes.count),\"sha256\":\"\(digest)\"}")
        let oldResponse = DeferredResponse<Data>(), newResponse = DeferredResponse<URL>()
        let actions = PrintActions(hosts: hosts)
        defer {
            actions.cancelExports()
            oldResponse.continuation?.resume(throwing: CancellationError())
            newResponse.continuation?.resume(throwing: CancellationError())
        }
        fake.stub("generationAsset(_:assetID:)") { _ in try await oldResponse.request() }
        fake.stub("mediaFile(_:trashed:)") { _ in try await newResponse.request() }
        let entry = LibraryEntry(host: hosts.hosts[0], print: print)
        actions.deliverAsset(asset, entry: entry, destination: .share)
        let oldTask = try #require(actions.fileExportTask)
        for _ in 0..<100 where oldResponse.continuation == nil { try await Task.sleep(for: .milliseconds(10)) }
        let oldDownload = try #require(oldResponse.continuation)
        actions.cancelExports()
        #expect(!actions.busy)
        actions.deliverOriginal(entry, destination: .share)
        let newTask = try #require(actions.fileExportTask)
        let newIdentity = try #require(actions.activeExportID)
        for _ in 0..<100 where newResponse.continuation == nil { try await Task.sleep(for: .milliseconds(10)) }
        let newDownload = try #require(newResponse.continuation)
        oldResponse.continuation = nil; oldDownload.resume(returning: bytes)
        await oldTask.value
        #expect(actions.busy && actions.activeExportID == newIdentity && actions.fileExportTask != nil)
        #expect(actions.sheet == nil && actions.pendingDelivery == nil)
        let source = FileManager.default.temporaryDirectory.appending(path: "original-\(UUID())")
        try Data("fixture original".utf8).write(to: source)
        defer { try? FileManager.default.removeItem(at: source) }
        actions.cancelExports()
        newResponse.continuation = nil; newDownload.resume(returning: source)
        await newTask.value
        #expect(!actions.busy && actions.activeExportID == nil && actions.fileExportTask == nil)
        #expect(actions.sheet == nil && actions.pendingDelivery == nil && actions.status == nil)
        #expect(hosts.failures.isEmpty && !FileManager.default.fileExists(atPath: source.path))
    }
}
