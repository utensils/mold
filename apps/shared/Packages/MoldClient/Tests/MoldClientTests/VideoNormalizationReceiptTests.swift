import Foundation
import Testing
@testable import MoldClient

struct VideoNormalizationReceiptTests {
    @Test func onlyFreshVideoMayChangeBytesAndCanonicalMetadataBindsReceipt() throws {
        let caps = try VideoNormalizationPolicyTests.decode(VideoNormalizationPolicyTests.advertisement)
        let (_, originalCaps) = try VideoNormalizationCaptureTests().fixture()
        var original = GenerateRequest(prompt: "scene", model: "minimax-h3-ref2va:official-bf16", width: 1280, height: 720, steps: 30, guidance: 1)
        original.pendingReferenceInputs = [.freshVideo(.init(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "a.mov"))]
        let capture = try VideoNormalizationCapture.capture(original, capabilities: originalCaps, apiKey: "key")
        let entry = try #require(capture.entries.first)
        let digest = String(repeating: "a", count: 64)
        let payload: [String: Any] = ["version": 1, "kind": "video", "original_sha256": TransportDigest.sha256(Data([1, 2, 3])),
            "original_byte_length": 3, "original_mime_type": "video/quicktime", "canonical_sha256": digest,
            "canonical_byte_length": 9, "canonical_mime_type": "video/mp4", "converted": true]
        let receipt = try MoldJSON.decoder.decode(VideoNormalizationReceipt.self, from: JSONSerialization.data(withJSONObject: payload))
        let metadata: [String: Any] = ["index": 1, "kind": "video", "mime_type": "video/mp4", "sha256": digest, "name": "a.mov",
            "width": 32, "height": 32, "duration_ms": 2000, "frame_count": 8, "fps": 4, "has_audio": false]
        let facts = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: JSONSerialization.data(withJSONObject: metadata))
        let canonical = try receipt.canonical(metadata: facts, entry: entry, capabilities: caps.videoNormalization!)
        #expect(canonical.provenance?.sha256 == digest)
        #expect(canonical.frameCount == 8)
        var invalid = payload; invalid["converted"] = false
        let unchanged = try MoldJSON.decoder.decode(VideoNormalizationReceipt.self, from: JSONSerialization.data(withJSONObject: invalid))
        #expect(throws: (any Error).self) { try unchanged.canonical(metadata: facts, entry: entry, capabilities: caps.videoNormalization!) }
        invalid = payload; invalid["original_byte_length"] = 4
        let wrongOriginal = try MoldJSON.decoder.decode(VideoNormalizationReceipt.self, from: JSONSerialization.data(withJSONObject: invalid))
        #expect(throws: (any Error).self) { try wrongOriginal.canonical(metadata: facts, entry: entry, capabilities: caps.videoNormalization!) }
    }
}
