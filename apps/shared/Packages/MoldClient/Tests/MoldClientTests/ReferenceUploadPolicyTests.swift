import Foundation
import Testing
@testable import MoldClient

@Test func uploadCapabilitiesRejectUnsafeRoutesAndCredentialHeaders() throws {
    let base: [String: Any] = ["available": true, "protocol_version": 2, "requires_api_key": true,
        "session_path": "/api/reference-sessions", "upload_path": "/api/reference-upload",
        "session_handle_header": "x-mold-reference-session", "upload_handle_header": "x-mold-reference-upload",
        "max_file_bytes": 1024, "max_session_bytes": 2048, "max_active_sessions": 4, "session_ttl_ms": 60000]
    let caps = try MoldJSON.decoder.decode(ReferenceUploadCapabilities.self, from: JSONSerialization.data(withJSONObject: base))
    #expect(throws: Never.self) { try ReferenceUploadPolicy.validate(caps) }
    for (key, value) in [("session_path", "https://foreign.example/api/upload"), ("upload_path", "/api/../private"),
                         ("session_handle_header", "Authorization"), ("upload_handle_header", "Cookie"),
                         ("upload_handle_header", "x-mold-reference-session")] {
        var changed = base; changed[key] = value
        let unsafe = try MoldJSON.decoder.decode(ReferenceUploadCapabilities.self, from: JSONSerialization.data(withJSONObject: changed))
        #expect(throws: MoldClientError.self) { try ReferenceUploadPolicy.validate(unsafe) }
    }
}

@Test func inlineVideoSoundtracksRequireServerCanonicalization() throws {
    var video = GenerationReference(kind: "video", media: .init(authority: "inline", data: "AA=="), mimeType: "video/mp4")
    video.hasAudio = true
    #expect(throws: MoldClientError.self) { try ReferenceUploadPolicy.validateInlineReferences([video]) }
    video.hasAudio = nil
    #expect(throws: Never.self) { try ReferenceUploadPolicy.validateInlineReferences([video]) }
    video.hasAudio = true; video.media = .init(authority: "upload", handle: "opaque")
    #expect(throws: Never.self) { try ReferenceUploadPolicy.validateInlineReferences([video]) }
}
