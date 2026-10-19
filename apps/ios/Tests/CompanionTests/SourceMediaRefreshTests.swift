import Foundation
import MoldClient
import MoldClientTesting
import Testing
import UIKit
@testable import MoldCompanion

@MainActor struct SourceMediaRefreshTests {
    private actor ThumbnailReads {
        var count = 0
        func read(_ image: Data) async throws -> Data {
            count += 1
            if count == 1 { try await Task.sleep(for: .seconds(30)) }
            return image
        }
    }

    @Test func invalidatingAnInFlightPreviewRecoversAtThePublishedEpochWithoutScrolling() async throws {
        let backend = FakeBackend(), reads = ThumbnailReads()
        let image = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        backend.stub("retainedSourceMediaThumbnail(for:member:)") { _ in try await reads.read(image) }
        backend.stub("status()", throwing: MoldClientError.unreachable("fixture offline"))
        let directory = FileManager.default.temporaryDirectory.appending(path: "source-preview-race-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        let hosts = HostStore(list: HostListFile(url: directory.appending(path: "hosts.json")),
                              credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in backend })
        try hosts.add(name: "Fixture", address: "127.0.0.1:34219", apiKey: nil, makeDefault: true)
        let host = try #require(hosts.hosts.first)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"owner.png","timestamp":1000,"media_version":"v1","metadata":{}}"#.utf8))
        let owner = LibraryEntry(host: host, print: print)
        let member = RetainedSourceMedia.Member(memberId: "original", role: "source_image", displayName: "image.png", sizeBytes: image.count)
        let key = SourcePreviewKey(authority: .init(owner: owner, route: host, instance: nil), member: member)
        let library = LibraryStore(hosts: hosts)
        let oldLoad = Task { await SourceMediaPreview.image(key: key, backend: backend) }
        for _ in 0..<200 where await reads.count == 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await reads.count == 1)
        let before = library.sourceMediaCatalog.contentRevision(for: owner.id)
        library.sourceMediaChanged(.resyncRequired, host: host.id)
        for _ in 0..<200 where library.sourceMediaCatalog.contentRevision(for: owner.id) == before {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(library.sourceMediaCatalog.contentRevision(for: owner.id) != before)
        // This is exactly when the tile's .task(id:) begins after revalidation.
        // No scroll/reparent/second invalidation is allowed to rescue the read.
        let recovered = await SourceMediaPreview.image(key: key, backend: backend)
        #expect(recovered != nil)
        #expect(await oldLoad.value == nil)
        #expect(await reads.count == 2)
        await SourceMediaPreview.invalidate(host: host.id)
    }
}
