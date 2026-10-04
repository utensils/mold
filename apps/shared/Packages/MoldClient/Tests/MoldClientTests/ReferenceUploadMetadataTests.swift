import Foundation
import Testing
@testable import MoldClient

@Test func uploadMetadataCanonicalizesWaveAliasesAndExactSampleFacts() throws {
    let digest = String(repeating: "a", count: 64)
    let metadata = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: Data("""
        {"index":1,"kind":"audio","mime_type":"audio/wav","sha256":"\(digest)","duration_ms":2000,"sample_rate":48000,"sample_count":96000,"channels":2}
        """.utf8))
    let original = GenerationReference(kind: "audio", media: .init(authority: "inline", data: "AQID"),
                                       mimeType: "audio/x-wav", provenance: .init(sha256: digest))
    let canonical = try metadata.canonical(original: original, index: 1)
    #expect(canonical.mimeType == "audio/wav")
    #expect(canonical.sampleCount == 96000)
    #expect(canonical.media.authority == "descriptor")
    #expect(canonical.media.data == nil)
}

@Test func videoMetadataRequiresSoundtrackFactsAndRejectsSpuriousAudio() throws {
    let digest = String(repeating: "a", count: 64)
    let original = GenerationReference(kind: "video", media: .init(authority: "inline", data: "AQID"),
                                       mimeType: "video/mp4", provenance: .init(sha256: digest))
    let base: [String: Any] = ["index": 1, "kind": "video", "mime_type": "video/mp4", "sha256": digest,
                               "width": 640, "height": 480, "frame_count": 61, "duration_ms": 2000,
                               "fps": 30.0, "has_audio": false]
    let silent = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: JSONSerialization.data(withJSONObject: base))
    #expect(try silent.canonical(original: original, index: 1).frameCount == 61)
    for (key, value) in [("has_audio", true as Any), ("audio_duration_ms", 2000 as Any), ("fps", 0 as Any), ("frame_count", 0 as Any)] {
        var broken = base; broken[key] = value
        let metadata = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: JSONSerialization.data(withJSONObject: broken))
        #expect(throws: MoldClientError.self) { try metadata.canonical(original: original, index: 1) }
    }
    var sounding = base
    sounding["has_audio"] = true; sounding["audio_duration_ms"] = 2000
    sounding["audio_sample_count"] = 96000; sounding["audio_sample_rate"] = 48000; sounding["audio_channels"] = 1
    let full = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: JSONSerialization.data(withJSONObject: sounding))
    #expect(try full.canonical(original: original, index: 1).audioSampleCount == 96000)
}

@Test func uploadSessionRejectsDuplicateSlotsExpiredAuthorityAndInvalidScope() throws {
    let now = HTTPBackend.referenceUploadNow
    let base: [String: Any] = ["instance_id": "instance", "expires_at_ms": now + 60000,
        "request_scope_sha256": String(repeating: "a", count: 64), "session_handle": "session",
        "uploads": [["reference": 1, "handle": "slot1"], ["reference": 2, "handle": "slot2"]]]
    for (key, value) in [("expires_at_ms", now as Any), ("request_scope_sha256", "bad" as Any),
                         ("session_handle", "newline\ncredential" as Any),
                         ("uploads", [["reference": 1, "handle": "duplicate"], ["reference": 2, "handle": "duplicate"]] as Any),
                         ("uploads", [["reference": 1, "handle": "slot1"]] as Any)] {
        var broken = base; broken[key] = value
        let session = try MoldJSON.decoder.decode(ReferenceUploadSessionResponse.self, from: JSONSerialization.data(withJSONObject: broken))
        #expect(throws: MoldClientError.self) { try session.validate(instance: "instance", indices: [1, 2], now: now) }
    }
}

@Test func silentVideoMetadataUsesServerOmissionOfFalseHasAudio() throws {
    let digest = String(repeating: "a", count: 64)
    let original = GenerationReference(kind: "video", media: .init(authority: "inline", data: "AQID"),
        mimeType: "video/mp4", provenance: .init(sha256: digest))
    let metadata = try MoldJSON.decoder.decode(ReferenceUploadMetadata.self, from: Data("""
    {"index":1,"kind":"video","mime_type":"video/mp4","sha256":"\(digest)",
     "width":128,"height":128,"frame_count":72,"duration_ms":3000,"fps":24}
    """.utf8))
    #expect(try metadata.canonical(original: original, index: 1).hasAudio == false)
}
