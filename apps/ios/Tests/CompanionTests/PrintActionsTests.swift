import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

@MainActor
struct PrintActionsTests {
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
