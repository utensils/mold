import Foundation
import Testing
@testable import MoldClient

private final class ArtifactWire: StubTransport {
    nonisolated(unsafe) static var sent: [URLRequest] = []
    nonisolated(unsafe) static var oversized = false
    override func startLoading() { Self.sent.append(request); super.startLoading() }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path.hasPrefix("/api/gallery/import/") { return (200,Data(#"{"filename":"copy.mp4"}"#.utf8)) }
        if !path.contains("/transfer") { return (200,Data(repeating:1,count:oversized ? 4 * 1024 * 1024 + 1 : 3)) }
        return (200,Data("""
        {"handle":"\(String(repeating:"f",count:64))","archive_identity_sha256":"\(String(repeating:"b",count:64))","offsets":{"provenance":"3"},"complete":true}
        """.utf8))
    }
}

@Suite(.serialized) struct VideoArtifactWireTests {
    let instance = "66687089-6429-4a48-8a5d-f302c8889092"
    let archive = String(repeating:"b",count:64)
    @Test func sourceAndDestinationRequestsCarryExactAuthorityAndLosslessOffsets() async throws {
        ArtifactWire.sent = []; ArtifactWire.oversized = false
        let backend = ArtifactWire.backend()
        _ = try await backend.videoArtifactBytes(filename:"movie +.mp4",role:.pcm,offset:UInt64.max,sourceInstance:instance,archive:archive)
        let source = try #require(ArtifactWire.sent.last)
        #expect(source.value(forHTTPHeaderField:"x-mold-source-instance") == instance)
        #expect(source.value(forHTTPHeaderField:"x-mold-gallery-archive-identity") == archive)
        #expect(source.url?.query == "offset_bytes=18446744073709551615")
        _ = try await backend.appendVideoArtifact(filename:"copy.mp4",handle:String(repeating:"f",count:64),role:.provenance,
            offset:UInt64.max,bytes:Data([1,2,3]),destinationInstance:instance,archive:archive)
        let target = try #require(ArtifactWire.sent.last)
        #expect(target.httpMethod == "PUT")
        #expect(target.value(forHTTPHeaderField:"x-mold-destination-instance") == instance)
        #expect(target.value(forHTTPHeaderField:"x-mold-gallery-archive-identity") == archive)
        #expect(target.url?.query == "offset_bytes=18446744073709551615")
    }
    @Test func sourceStreamCannotAllocateMoreThanOneBoundedChunk() async throws {
        ArtifactWire.sent = []; ArtifactWire.oversized = true
        defer { ArtifactWire.oversized = false }
        await #expect(throws:(any Error).self) {
            _ = try await ArtifactWire.backend().videoArtifactBytes(filename:"movie.mp4",role:.provenance,
                offset:0,sourceInstance:instance,archive:archive)
        }
    }
    @Test func artifactOutputImportRequiresCapturedDestinationInsteadOfFreshStatus() async throws {
        let offer = try VideoArtifactRelayTests().offer("a",count:3)
        let row = try MoldJSON.decoder.decode(GalleryPrint.self,from:Data("""
        {"filename":"copy.mp4","timestamp":1,"metadata":\(String(data:MoldJSON.encoder.encode(offer.metadata),encoding:.utf8)!)}
        """.utf8))
        var item = GalleryImport(mirroring:row,file:Data([1,2,3]))
        ArtifactWire.sent = []
        await #expect(throws:(any Error).self) { _ = try await ArtifactWire.backend().importPrint(item,as:"copy.mp4") }
        #expect(ArtifactWire.sent.isEmpty)
        item.destinationInstance = instance
        _ = try await ArtifactWire.backend().importPrint(item,as:"copy.mp4")
        #expect(ArtifactWire.sent.last?.value(forHTTPHeaderField:"x-mold-destination-instance") == instance)
        #expect(!ArtifactWire.sent.contains { $0.url?.path == "/api/status" })
    }
}
