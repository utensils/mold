import Foundation
import Testing
@testable import MoldClient

struct VideoNormalizationPolicyTests {
    static var advertisement: [String: Any] {
        ["available": true, "protocol_version": 2, "requires_api_key": true,
         "session_path": "/api/reference-sessions", "upload_path": "/api/reference-upload",
         "session_handle_header": "x-mold-reference-session", "upload_handle_header": "x-mold-reference-upload",
         "max_file_bytes": 4096, "max_session_bytes": 8192, "max_active_sessions": 4, "session_ttl_ms": 60000,
         "video_normalization": ["available": true, "protocol_version": 1, "input_roles": ["reference_video"],
            "candidate_session_path": "/api/reference-candidates", "finalize_path": "/api/reference-finalize",
            "candidate_digest_header": "x-mold-candidate-sha256", "max_original_file_bytes": 4096,
            "max_canonical_file_bytes": 4096, "max_session_bytes": 8192,
            "accepted_mime_types": ["video/mp4", "video/quicktime", "video/webm"],
            "accepted_extensions": ["mp4", "mov", "webm"], "accepted_video_codecs": ["h264", "hevc", "vp9"],
            "canonical_mime_type": "video/mp4"]]
    }
    static func decode(_ value: [String: Any]) throws -> ReferenceUploadCapabilities {
        try MoldJSON.decoder.decode(ReferenceUploadCapabilities.self, from: JSONSerialization.data(withJSONObject: value))
    }
    @Test func freshVideoNegotiationRequiresAuthenticatedSafeAdditiveProtocol() throws {
        let caps = try Self.decode(Self.advertisement)
        #expect(VideoNormalizationPolicy.capability(caps, apiKey: "key")?.acceptedExtensions == ["mp4", "mov", "webm"])
        #expect(VideoNormalizationPolicy.capability(caps, apiKey: " ") == nil)
        for (key, value): (String, Any) in [
            ("protocol_version", 2), ("input_roles", ["source_video"]),
            ("candidate_session_path", "https://elsewhere/api/session"),
            ("finalize_path", "/api/reference-candidates"), ("candidate_digest_header", "Authorization"),
            ("candidate_digest_header", "x-mold-reference-upload"), ("max_original_file_bytes", 0),
            ("max_canonical_file_bytes", 8193), ("max_session_bytes", 9000),
            ("accepted_mime_types", ["text/plain"]), ("accepted_extensions", ["../mov"]),
            ("accepted_video_codecs", []), ("canonical_mime_type", "video/webm")
        ] {
            var advertised = Self.advertisement
            var nested = advertised["video_normalization"] as! [String: Any]
            nested[key] = value; advertised["video_normalization"] = nested
            if key == "protocol_version" {
                #expect(VideoNormalizationPolicy.capability(try Self.decode(advertised), apiKey: "key") == nil)
            } else {
                let caps = try Self.decode(advertised)
                #expect(VideoNormalizationPolicy.capability(caps, apiKey: "key") == nil)
                try ReferenceUploadPolicy.validate(caps)
            }
        }
    }
    @Test func malformedAdditiveBlockPreservesUnrelatedCapabilitiesAndCanonicalV2() throws {
        var advertised = Self.advertisement
        advertised["video_normalization"] = ["available": "invalid"]
        let document: [String: Any] = ["reference_uploads": advertised,
            "gallery": ["organize": true, "trash": ["enabled": true]],
            "queue": ["durable_queue": true], "events": ["available": true]]
        let full = try MoldJSON.decoder.decode(Capabilities.self, from: JSONSerialization.data(withJSONObject: document))
        #expect(full.gallery?.organize == true)
        #expect(full.gallery?.trash?.enabled == true)
        #expect(full.queue?.durableQueue == true)
        #expect(full.events?.available == true)
        try ReferenceUploadPolicy.validate(try #require(full.referenceUploads))
        let encoded = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(full)) as! [String: Any]
        let uploads = encoded["reference_uploads"] as! [String: Any]
        #expect(uploads.keys.allSatisfy { !$0.contains("failure") })
        advertised["video_normalization"] = ["available": true, "protocol_version": 1]
        #expect(VideoNormalizationPolicy.capability(try Self.decode(advertised), apiKey: "key") == nil)
    }
    @Test func absentAndFutureAdditiveProtocolsPreserveCanonicalUploadV2() throws {
        var advertised = Self.advertisement
        advertised.removeValue(forKey: "video_normalization")
        let caps = try Self.decode(advertised)
        #expect(caps.videoNormalization == nil)
        try ReferenceUploadPolicy.validate(caps)
        advertised["video_normalization"] = ["protocol_version": 2, "future_schema": true]
        let future = try Self.decode(advertised)
        #expect(future.videoNormalization == nil)
        try ReferenceUploadPolicy.validate(future)
        advertised["video_normalization"] = ["available": true]
        let unversioned = try Self.decode(advertised)
        #expect(unversioned.videoNormalization == nil)
        try VideoNormalizationPolicy.requireValidAdvertisement(unversioned)
        try ReferenceUploadPolicy.validate(unversioned)
    }
}

struct FreshVideoImportTests {
    @Test func malformedAdvertisedV1RefusesBeforeCanonicalVideoFallback() async throws {
        var advertised = VideoNormalizationPolicyTests.advertisement
        advertised["video_normalization"] = ["available": true, "protocol_version": 1]
        let caps = try VideoNormalizationPolicyTests.decode(advertised)
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/endpoint-bframes.mp4")
        do {
            _ = try await GenerationReferenceImporter.loadAuthoring(url: file, capabilities: caps, apiKey: "key")
            Issue.record("Malformed known V1 fell through to canonical probing")
        } catch MoldClientError.http(_, let code, _) {
            #expect(code == "REFERENCE_NORMALIZATION_INVALID")
        } catch { Issue.record("Expected a specific normalization contract error, received \(error)") }
    }
    @Test func negotiatedImportKeepsOriginalWithoutTryingToDecodeCanonicalFacts() async throws {
        let caps = try VideoNormalizationPolicyTests.decode(VideoNormalizationPolicyTests.advertisement)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("webm")
        defer { try? FileManager.default.removeItem(at: file) }
        let bytes = Data([0x1a, 0x45, 0xdf, 0xa3, 1, 2, 3])
        try bytes.write(to: file)
        let imported = try await GenerationReferenceImporter.loadAuthoring(url: file, capabilities: caps, apiKey: "key")
        #expect(imported.freshVideo?.data == bytes)
        #expect(imported.freshVideo?.mimeType == "video/webm")
        #expect(imported.frameCount == nil)
        #expect(imported.canonicalReference == nil)
    }
    @Test func negotiatedOriginalLimitIsEnforcedBeforeCreatingCandidate() async throws {
        let caps = try VideoNormalizationPolicyTests.decode(VideoNormalizationPolicyTests.advertisement)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("webm")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data(repeating: 1, count: 4097).write(to: file)
        await #expect(throws: (any Error).self) {
            try await GenerationReferenceImporter.loadAuthoring(url: file, capabilities: caps, apiKey: "key")
        }
    }
    @Test func absentCapabilityDoesNotPromoteAnUnprobedFile() async throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("webm")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data([1, 2, 3]).write(to: file)
        await #expect(throws: (any Error).self) {
            try await GenerationReferenceImporter.loadAuthoring(url: file, capabilities: nil, apiKey: "key")
        }
    }
}
