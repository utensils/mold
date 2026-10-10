import Foundation
import Testing
@testable import MoldClient

struct SourceMediaPresentationTests {
    @Test func legacyRolesAreClassifiedWithoutTrustingFilenames() {
        let image = RetainedSourceMedia.Member(memberId: "one", role: "keyframes", displayName: "item.json", sizeBytes: 20)
        let unknown = RetainedSourceMedia.Member(memberId: "two", role: "future", displayName: "photo.png", sizeBytes: 20)
        #expect(image.contentKind == "image")
        #expect(unknown.contentKind == "unknown")
    }
    @Test func additiveMetadataPreservesExactFrameAndOrder() throws {
        let member = try MoldJSON.decoder.decode(RetainedSourceMedia.Member.self, from: Data(#"{"member_id":"one","role":"keyframes","display_name":"frame.json","size_bytes":20,"media_kind":"image","label":"End frame","position":1,"frame_index":80}"#.utf8))
        #expect(member.sourceLabel == "End frame")
        #expect(member.frameIndex == 80)
        #expect(member.position == 1)
    }
    @Test func unknownKindDoesNotBecomeAnImageFromItsRole() {
        let member = RetainedSourceMedia.Member(memberId: "one", role: "source_image", displayName: "image.png", sizeBytes: 20, mediaKind: "future")
        #expect(member.contentKind == "unknown")
    }
}
