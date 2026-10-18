import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

extension GenerateControllerTests {
    @Test func pendingFreshReferencesPinOnlyThisAttemptToEligibleAutoHost() async throws {
        let alternate = FakeBackend()
        alternate.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"test","busy":false,"uptime_secs":1}"#.utf8)))
        alternate.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data(Self.normalizationAdvertisement.utf8)))
        alternate.stub("models()", returning: [try model()])
        let (generate, _) = try await setUp { host, first in host.baseURL.host == "10.0.0.5" ? alternate : first }
        let original = generate.target!
        try generate.hosts.add(name: "normalizer", address: "10.0.0.5", apiKey: "key", makeDefault: false)
        await generate.hosts.refreshAll()
        let capable = generate.hosts.hosts.first { $0.id != original.id }!
        #expect(generate.referenceImportTarget?.id == capable.id)
        generate.draft.media.generationReferences = [.freshVideo(.init(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "a.mov"))]
        #expect(generate.target?.id == capable.id)
        #expect(generate.machine == .auto)
        generate.machine = .pinned(original.id)
        #expect(generate.target == nil)
        generate.machine = .pinned(capable.id)
        #expect(generate.target?.id == capable.id)
        generate.draft.media.generationReferences = []
        generate.machine = .auto
        #expect(generate.target?.id == original.id)
    }

    private static let normalizationAdvertisement = #"{"reference_uploads":{"available":true,"protocol_version":2,"requires_api_key":true,"session_path":"/api/reference-sessions","upload_path":"/api/reference-upload","session_handle_header":"x-session","upload_handle_header":"x-upload","max_file_bytes":4096,"max_session_bytes":8192,"max_active_sessions":4,"session_ttl_ms":60000,"video_normalization":{"available":true,"protocol_version":1,"input_roles":["reference_video"],"candidate_session_path":"/api/candidates","finalize_path":"/api/finalize","candidate_digest_header":"x-candidate","max_original_file_bytes":4096,"max_canonical_file_bytes":4096,"max_session_bytes":8192,"accepted_mime_types":["video/quicktime"],"accepted_extensions":["mov"],"accepted_video_codecs":["hevc"],"canonical_mime_type":"video/mp4"}}}"#
}
