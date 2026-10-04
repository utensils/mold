import Foundation
import Testing
@testable import MoldClient

private final class ReferenceUploadStub: StubTransport {
    // Transport happy paths must tolerate loaded CI runners; deterministic
    // session-validation tests cover expiry without a wall-clock deadline.
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var sessionInstance = "instance"
    nonisolated(unsafe) static var completed = true
    nonisolated(unsafe) static var sessionCount = 0
    nonisolated(unsafe) static var refuseAdmission = false
    nonisolated(unsafe) static var holdUpload = false
    nonisolated(unsafe) static var capsData = Data()
    nonisolated(unsafe) static var metadataDigest = RelayTransport.sha256(Data([1, 2, 3]))
    override class func response(for path: String) -> (status: Int, body: Data)? {
        let digest = RelayTransport.sha256(Data([1, 2, 3]))
        switch path {
        case "/api/capabilities": return (200, capsData)
        case "/api/status": return (200, Data(#"{"version":"test","busy":false,"instance_id":"instance","uptime_secs":1}"#.utf8))
        case "/api/generation-batches":
            return refuseAdmission ? (422, Data(#"{"error":"refused"}"#.utf8)) : (200, try! Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/batch-status.json")))
        case "/api/reference-sessions":
            return (200, Data("""
                {"instance_id":"\(sessionInstance)","expires_at_ms":\(HTTPBackend.referenceUploadNow + 3_600_000),
                 "request_scope_sha256":"\(digest)","session_handle":"session-secret-\(sessionCount)","uploads":[{"reference":1,"handle":"upload-secret-\(sessionCount)"}]}
                """.utf8))
        case "/api/reference-upload":
            return (200, Data("""
                {"instance_id":"instance","reference":1,"request_scope_sha256":"\(digest)","session_complete":\(completed),
                 "metadata":{"index":1,"kind":"image","mime_type":"image/png","sha256":"\(metadataDigest)","name":"sample.png","width":80,"height":60}}
                """.utf8))
        default: return (404, Data())
        }
    }
    override func startLoading() {
        Self.requests.append(request)
        if request.httpMethod == "POST" && request.url?.path == "/api/reference-sessions" { Self.sessionCount += 1 }
        if Self.holdUpload && request.httpMethod == "PUT" { return }
        super.startLoading()
    }
}

@Suite(.serialized)
struct ReferenceUploadTransportTests {
    func fixture() throws -> (HTTPBackend, GenerateRequest, ReferenceUploadCapabilities) {
        ReferenceUploadStub.requests = []; ReferenceUploadStub.sessionInstance = "instance"
        ReferenceUploadStub.completed = true; ReferenceUploadStub.sessionCount = 0; ReferenceUploadStub.refuseAdmission = false; ReferenceUploadStub.holdUpload = false; ReferenceUploadStub.metadataDigest = RelayTransport.sha256(Data([1, 2, 3]))
        let config = URLSessionConfiguration.ephemeral; config.protocolClasses = [ReferenceUploadStub.self]
        let backend = HTTPBackend(host: MoldHost(name: "test", baseURL: URL(string: "http://upload-test.local")!, apiKey: "key"), session: URLSession(configuration: config))
        var request = GenerateRequest(prompt: "scene", model: "minimax-h3-ref2va:official-bf16", width: 1280, height: 720, steps: 30, guidance: 1)
        request.references = [.init(kind: "image", media: .init(authority: "inline", data: "AQID"), mimeType: "image/png", provenance: .init(name: "sample.png"), width: 1, height: 1)]
        let caps = try MoldJSON.decoder.decode(ReferenceUploadCapabilities.self, from: Data(#"{"available":true,"protocol_version":2,"requires_api_key":true,"session_path":"/api/reference-sessions","upload_path":"/api/reference-upload","session_handle_header":"x-reference-session","upload_handle_header":"x-reference-upload","max_file_bytes":1024,"max_session_bytes":2048,"max_active_sessions":2,"session_ttl_ms":3600000}"#.utf8))
        var capabilitiesJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/capabilities.json"))) as! [String: Any]
        capabilitiesJSON["reference_uploads"] = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(caps))
        ReferenceUploadStub.capsData = try JSONSerialization.data(withJSONObject: capabilitiesJSON)
        return (backend, request, caps)
    }
    @Test func canonicalMetadataAndOneUseAuthorityDoNotMutateOriginal() async throws {
        let (backend, original, caps) = try fixture()
        let lease = try await backend.prepareReferenceUploads(original, capabilities: caps, expectedInstanceId: "instance")
        #expect(lease.request.references?.first?.width == 80)
        #expect(lease.request.references?.first?.media == .init(authority: "upload", handle: "upload-secret-1"))
        #expect(original.references?.first?.media.authority == "inline")
        let opening = try #require(ReferenceUploadStub.requests.first(where: { $0.httpMethod == "POST" }))
        let json = try JSONSerialization.jsonObject(with: requestBody(opening)) as? [String: Any]
        let scoped = try #require(json?["request"] as? [String: Any])
        let references = try #require(scoped["references"] as? [[String: Any]])
        #expect((references.first?["media"] as? [String: Any])?["authority"] as? String == "descriptor")
        #expect(ReferenceUploadStub.requests.first(where: { $0.httpMethod == "PUT" })?.value(forHTTPHeaderField: "x-reference-upload") == "upload-secret-1")
        await lease.cancel(); await lease.cancel()
        #expect(ReferenceUploadStub.requests.filter { $0.httpMethod == "DELETE" }.count == 1)
    }
    @Test func eachBatchSiblingGetsFreshLeaseAndAllAreReleasedOnRefusal() async throws {
        let (backend, original, _) = try fixture()
        ReferenceUploadStub.refuseAdmission = true
        await #expect(throws: MoldClientError.self) { try await backend.submit(BatchAdmission(requests: [original, original])) }
        #expect(ReferenceUploadStub.sessionCount == 2)
        let post = try #require(ReferenceUploadStub.requests.first(where: { $0.url?.path == "/api/generation-batches" }))
        let body = try MoldJSON.decoder.decode(BatchAdmission.self, from: requestBody(post))
        #expect(body.requests[0].references?.first?.media.handle == "upload-secret-1")
        #expect(body.requests[1].references?.first?.media.handle == "upload-secret-2")
        #expect(ReferenceUploadStub.requests.filter { $0.httpMethod == "DELETE" }.count == 2)
    }
    @Test func batchOverSessionLimitFailsBeforeOpeningSessions() async throws {
        let (backend, original, caps) = try fixture()
        #expect(ReferenceUploadPolicy.batchLimit(requests: [original], apiKey: "key", capabilities: caps, batchLimit: 64) == 2)
        await #expect(throws: MoldClientError.self) { try await backend.submit(BatchAdmission(requests: [original, original, original])) }
        #expect(ReferenceUploadStub.sessionCount == 0)
    }
    @Test func cancellingParentStillReleasesItsSession() async throws {
        let (backend, original, caps) = try fixture(); ReferenceUploadStub.holdUpload = true
        let preparation = Task { try await backend.prepareReferenceUploads(original, capabilities: caps, expectedInstanceId: "instance") }
        for _ in 0..<500 {
            if ReferenceUploadStub.requests.contains(where: { $0.httpMethod == "PUT" }) { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(ReferenceUploadStub.requests.contains(where: { $0.httpMethod == "PUT" }))
        preparation.cancel()
        do { _ = try await preparation.value; Issue.record("Cancelled preparation unexpectedly returned a lease") } catch {}
        #expect(ReferenceUploadStub.requests.filter { $0.httpMethod == "DELETE" }.count == 1)
    }
    @Test func wrongInstanceReleasesReturnedSession() async throws {
        let (backend, original, caps) = try fixture(); ReferenceUploadStub.sessionInstance = "foreign"
        await #expect(throws: MoldClientError.self) { try await backend.prepareReferenceUploads(original, capabilities: caps, expectedInstanceId: "instance") }
        #expect(ReferenceUploadStub.requests.filter { $0.httpMethod == "DELETE" }.count == 1)
        #expect(!ReferenceUploadStub.requests.contains { $0.httpMethod == "PUT" })
    }
    @Test func digestMismatchAndIncorrectCompletionReleaseSession() async throws {
        for mismatch in [true, false] {
            let (backend, original, caps) = try fixture()
            if mismatch { ReferenceUploadStub.metadataDigest = String(repeating: "0", count: 64) } else { ReferenceUploadStub.completed = false }
            await #expect(throws: MoldClientError.self) { try await backend.prepareReferenceUploads(original, capabilities: caps, expectedInstanceId: "instance") }
            #expect(ReferenceUploadStub.requests.filter { $0.httpMethod == "DELETE" }.count == 1)
        }
    }
    @Test func invalidDigestAndNoncanonicalBase64RefusedBeforeNetwork() async throws {
        for data in ["AQI=", "AQJ=", "AQID\n", "A==="] {
            let (backend, base, caps) = try fixture(); var original = base
            original.references?[0].media.data = data
            original.references?[0].provenance?.sha256 = String(repeating: "0", count: 64)
            await #expect(throws: MoldClientError.self) { try await backend.prepareReferenceUploads(original, capabilities: caps, expectedInstanceId: "instance") }
            #expect(ReferenceUploadStub.requests.isEmpty)
        }
    }
}

private func requestBody(_ request: URLRequest) -> Data {
    if let data = request.httpBody { return data }
    guard let stream = request.httpBodyStream else { return Data() }
    stream.open(); defer { stream.close() }
    var data = Data(); var buffer = [UInt8](repeating: 0, count: 4096)
    while stream.hasBytesAvailable {
        let count = stream.read(&buffer, maxLength: buffer.count)
        if count <= 0 { break }; data.append(buffer, count: count)
    }
    return data
}
