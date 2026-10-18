import Foundation
import Testing
@testable import MoldClient

struct AuthoringReferenceTests {
    @Test func freshVideoHasOriginalBytesWithoutInventedCanonicalFacts() throws {
        let source = FreshVideoReference(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "picked.mov")
        let input = AuthoringReference.freshVideo(source)
        #expect(input.kind == "video")
        #expect(input.canonicalReference == nil)
        #expect(source.sha256 == TransportDigest.sha256(Data([1, 2, 3])))
        #expect(input.durationMs == nil)
        #expect(input.frameCount == nil)
    }

    @Test func pendingReferencesCannotEscapeThroughOrdinaryGenerationOrPlacementEncoding() throws {
        let source = FreshVideoReference(data: Data([1, 2, 3]), mimeType: "video/webm", name: "picked.webm")
        var draft = RenderDraft()
        draft.media.generationReferences = [.freshVideo(source)]
        let request = RenderRequest.one(draft, model: "reference-model")
        #expect(request.references == nil)
        #expect(request.pendingReferenceInputs?.count == 1)
        #expect(throws: (any Error).self) { try MoldJSON.encoder.encode(request) }
        #expect(throws: (any Error).self) { try MoldJSON.encoder.encode(request.redactedForPlacement()) }
    }

    @Test func mixedReferenceOrderSurvivesMovesAndParkingWithoutChangingOriginalBytes() throws {
        let source = FreshVideoReference(data: Data([1, 2, 3]), mimeType: "video/webm", name: "picked.webm")
        let image = GenerationReference(kind: "image", media: .init(authority: "inline", data: "AQ=="), mimeType: "image/png", width: 1, height: 1)
        var media = DraftMedia()
        media.generationReferences = [.canonical(image), .freshVideo(source)]
        media.moveGenerationReference(from: 1, to: 0)
        #expect(media.generationReferences[0] == .freshVideo(source))
        #expect(media.generationReferences[1].canonicalReference == image)
    }
    @Test func privateSnapshotPreservesFreshOriginalAndCannotTurnItIntoRetainedAuthority() throws {
        let source = FreshVideoReference(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "picked.mov")
        var media = DraftMedia()
        media.generationReferences = [.freshVideo(source)]
        let encoded = try MoldJSON.localEncoder.encode(DraftInputSnapshot(media))
        let snapshot = try MoldJSON.localDecoder.decode(DraftInputSnapshot.self, from: encoded)
        var restored = DraftMedia()
        try snapshot.apply(to: &restored)
        #expect(restored.generationReferences == media.generationReferences)
        #expect(!RetainedReferenceGuard.canHydrate(references: restored.generationReferences,
            original: media.generationReferences, members: []))
        #expect(restored.generationReferences.canonicalReferences == nil)
    }

}
