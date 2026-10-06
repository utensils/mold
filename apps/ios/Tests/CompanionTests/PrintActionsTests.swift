import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

@MainActor
struct PrintActionsTests {
    @MainActor private final class DeferredOriginalDownload {
        var continuation: CheckedContinuation<URL, any Error>?
        func request() async throws -> URL {
            try await withCheckedThrowingContinuation { continuation = $0 }
        }
    }

    @Test func immediateRepeatedOriginalDeliveryKeepsTheActiveDownloadCancellable() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        let actions = PrintActions(hosts: hosts)
        let pending = DeferredOriginalDownload()
        defer { pending.continuation?.resume(throwing: CancellationError()) }
        fake.stub("mediaFile(_:trashed:)") { _ in try await pending.request() }
        let entry = LibraryEntry(host: hosts.hosts[0], print: print)
        actions.deliverOriginal(entry, destination: .share)
        let first = try #require(actions.fileExportTask)
        #expect(actions.busy)
        actions.deliverOriginal(entry, destination: .share)
        let tracked = try #require(actions.fileExportTask)
        for _ in 0..<100 where pending.continuation == nil { try await Task.sleep(for: .milliseconds(10)) }
        let download = try #require(pending.continuation)
        let source = FileManager.default.temporaryDirectory.appending(path: "original-\(UUID())")
        try Data("fixture original".utf8).write(to: source)
        defer {
            try? FileManager.default.removeItem(at: source)
            if case let .share(urls) = actions.sheet { PrintActions.removeFiles(urls) }
        }
        actions.cancelExports()
        pending.continuation = nil; download.resume(returning: source)
        await first.value; await tracked.value
        #expect(fake.count("mediaFile(_:trashed:)") == 1)
        #expect(actions.sheet == nil && actions.pendingDelivery == nil && actions.status == nil)
        #expect(hosts.failures.isEmpty && !actions.busy)
        #expect(!FileManager.default.fileExists(atPath: source.path))
    }

    @Test func cancelledOriginalDeliveryDoesNotPublishAMachineFailure() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"loop.mp4","metadata":{},"timestamp":1790000000}"#)
        let actions = PrintActions(hosts: hosts)
        let pending = DeferredOriginalDownload()
        defer { pending.continuation?.resume(throwing: CancellationError()) }
        fake.stub("mediaFile(_:trashed:)") { _ in try await pending.request() }
        actions.deliverOriginal(LibraryEntry(host: hosts.hosts[0], print: print), destination: .share)
        let operation = try #require(actions.fileExportTask)
        for _ in 0..<100 where pending.continuation == nil { try await Task.sleep(for: .milliseconds(10)) }
        let download = try #require(pending.continuation)
        actions.cancelExports()
        pending.continuation = nil; download.resume(throwing: URLError(.cancelled))
        await operation.value
        #expect(hosts.failures.isEmpty)
        #expect(actions.sheet == nil && actions.pendingDelivery == nil && actions.status == nil)
        #expect(!actions.busy && actions.fileExportTask == nil)
    }

    @Test(arguments: [("dawn.mp4", Optional<String>.none, true), ("dawn.mp4", "mp4", true),
                      ("camera.mov", nil, true), ("camera.m4v", nil, true),
                      ("loop.gif", nil, false), ("loop.gif", "gif", false),
                      ("loop.webp", "webp", false), ("loop.apng", "apng", false),
                      ("still.png", nil, false)])
    func photosExportDistinguishesVideoContainersFromAnimatedPhotos(filename: String, format: String?, video: Bool) async throws {
        let (_, hosts, _) = try await QueueStoreTests.setUp()
        var json: [String: Any] = ["filename": filename, "metadata": ["frames": 10], "timestamp": 1790000000]
        if let format { json["format"] = format }
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: JSONSerialization.data(withJSONObject: json))
        let url = URL(fileURLWithPath: "/tmp/" + filename)
        let resources = PrintActions.photoResources(urls: [url], entries: [LibraryEntry(host: hosts.hosts[0], print: print)])
        #expect(resources.count == 1)
        #expect(resources[0].video == video)
        #expect(resources[0].url == url)
    }

    @Test func aRemovedMachineDoesNotSilentlyDropAnExportEntry() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let host = hosts.hosts[0]
        let print = try QueueStoreTests.decode(GalleryPrint.self,
            #"{"filename":"dawn.mp4","metadata":{},"timestamp":1790000000,"format":"mp4"}"#)
        let entry = LibraryEntry(host: host, print: print)
        hosts.setHosts([])
        let actions = PrintActions(hosts: hosts)
        #expect(await actions.files(for: [entry]) == nil)
        #expect(fake.count("mediaFile(_:trashed:)") == 0)
        #expect(!hosts.failures.isEmpty)
    }

    @Test func sharingPreservesTheMediaFilenameAndExtension() async throws {
        let (_, hosts, fake) = try await QueueStoreTests.setUp()
        let host = hosts.hosts[0]
        let downloaded = FileManager.default.temporaryDirectory.appending(path: "download-\(UUID())")
        try Data("fixture media".utf8).write(to: downloaded)
        defer { try? FileManager.default.removeItem(at: downloaded) }
        fake.stub("mediaFile(_:trashed:)", returning: downloaded)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"dawn.mp4","metadata":{"prompt":"dawn"},"timestamp":1790000000,"format":"mp4"}"#.utf8))
        let entry = LibraryEntry(host: host, print: print)
        let actions = PrintActions(hosts: hosts)
        actions.share([entry])
        for _ in 0..<100 where actions.sheet == nil { try await Task.sleep(for: .milliseconds(10)) }
        guard case let .share(urls) = try #require(actions.sheet) else { Issue.record("Share sheet missing"); return }
        #expect(urls.first?.lastPathComponent == "dawn.mp4")
        let shared = try #require(urls.first)
        #expect(try Data(contentsOf: shared) == Data("fixture media".utf8))
        #expect(!FileManager.default.fileExists(atPath: downloaded.path))
        actions.shareFinished()
        #expect(!FileManager.default.fileExists(atPath: shared.deletingLastPathComponent().path))
    }
}
