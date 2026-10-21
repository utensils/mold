import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct VideoSettingsArchiveTests {
    @Test func ownerChangeDuringInventoryCannotPublishOrFetchPrivateMedia() async throws {
        let backend = FakeBackend(), gate = AsyncStream<Void>.makeStream()
        let owner = VideoJobOwner(host: backend.host, instanceId: "installation")
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"video","prompt":"original"}"#.utf8))
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"original.mp4","format":"mp4","metadata":{"model":"video","prompt":"original"},"timestamp":1}"#.utf8))
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([print], etag: nil))
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8)))
        backend.stub("retainedSourceMedia(for:)") { _ in
            for await _ in gate.stream { break }
            return RetainedSourceMedia.Inventory(availability: .unavailableLegacy)
        }
        var current = true
        let task = Task {
            await #expect(throws: (any Error).self) {
                try await VideoSettingsArchive.inspect(filename: "original.mp4", metadata: metadata,
                    owner: owner, backend: backend, current: { current })
            }
        }
        for _ in 0..<100 where backend.count("retainedSourceMedia(for:)") == 0 { await Task.yield() }
        current = false; gate.continuation.finish(); await task.value
        #expect(backend.count("retainedSourceMediaBytes(for:member:)") == 0)
        #expect(backend.count("retainedMediaTransferOffer(for:)") == 0)
    }
}
