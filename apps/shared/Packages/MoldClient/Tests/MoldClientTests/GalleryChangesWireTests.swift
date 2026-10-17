import Foundation
import Testing

@testable import MoldClient

private final class ChangesTransport: StubTransport {
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        guard path == "/api/gallery/changes" else { return nil }
        return (200, Data("""
        {"protocol_version":1,"instance_id":"i-1","cursor":"c-2","more":false,"reset":false,
         "changes":[
          {"filename":"a.png","removed":false,"image":{"filename":"a.png","metadata":{"prompt":"cat","future":true},
           "timestamp":5,"format":"png","size_bytes":3,"media_version":"v1","trashed_at":9}},
          {"filename":"b.png","removed":true},
          {"filename":"../evil.png","removed":false,"image":{"filename":"../evil.png","metadata":{},"timestamp":1}}
         ]}
        """.utf8))
    }
}

private final class MissingFeedTransport: StubTransport {
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        (404, Data(#"{"error":"not found","code":"NOT_FOUND"}"#.utf8))
    }
}

struct GalleryChangesWireTests {
    @Test func aPageDecodesRowsRemovalsAndRefusals() async throws {
        let page = try await ChangesTransport.backend().galleryChanges(since: "c-1", limit: 5_000)
        #expect(page.instanceId == "i-1")
        #expect(page.cursor == "c-2")
        #expect(page.changes.map(\.filename) == ["a.png", "b.png", "../evil.png"])
        let row = try #require(page.changes[0].image)
        #expect(row.trashedAt == 9)
        #expect(row.metadata.prompt == "cat")
        // The original recipe survives for a byte-preserving import.
        let raw = try #require(row.rawMetadataJSON)
        let object = try #require(try JSONSerialization.jsonObject(with: raw) as? [String: Any])
        #expect(object["future"] as? Bool == true)
        #expect(page.changes[1].removed)
        // A name this client refuses to hold is a removal of it, not a failure.
        #expect(page.changes[2].removed)
        #expect(page.changes[2].image == nil)
    }

    @Test func anOlderMachineAnswersWithTheStatusTheReplicaFallsBackOn() async {
        do {
            _ = try await MissingFeedTransport.backend().galleryChanges(since: nil, limit: 10)
            Issue.record("expected a refusal")
        } catch {
            #expect(GalleryReplica.fallsBackToListing(error))
        }
    }

    @Test func theCapabilityIsReadOnlyForProtocolOne() throws {
        func capabilities(_ json: String) throws -> Capabilities {
            try MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
        }
        #expect(try capabilities(#"{"gallery":{"changes_feed":{"protocol_version":1}}}"#).hasGalleryChangesFeed)
        #expect(try !capabilities(#"{"gallery":{"changes_feed":{"protocol_version":2}}}"#).hasGalleryChangesFeed)
        #expect(try !capabilities(#"{"gallery":{}}"#).hasGalleryChangesFeed)
        #expect(try !capabilities("{}").hasGalleryChangesFeed)
    }
}
