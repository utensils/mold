import Foundation
import Testing
import Synchronization
@testable import MoldClient

private actor NormalizationBytes: MoldBodyReader {
    var data: Data?
    init(_ value: Data) { data = value }
    func read() -> Data? { defer { data = nil }; return data }
    func cancel() { data = nil }
}
private final class NormalizationClock: Sendable {
    private let time = Mutex<Int64>(HTTPBackend.referenceUploadNow)
    func now() -> Int64 { time.withLock { $0 } }
    func advance() { time.withLock { $0 += 120000 } }
    func reset() { time.withLock { $0 = HTTPBackend.referenceUploadNow } }
}
private actor NormalizationAuthority {
    var host: MoldHost
    var instance = "instance"
    init(_ host: MoldHost) { self.host = host }
    func isCurrent(_ original: MoldHost) -> Bool { host == original && instance == "instance" }
    func change(_ field: String) {
        switch field {
        case "key": host.apiKey = "replacement"
        case "route": host.baseURL = URL(string: "http://replacement")!
        case "mode": host.connectionMode = .regular
        case "instance": instance = "replacement"
        default: host.secure = .init(binding: .init(signingKey: "changed", noiseKey: "changed", signature: "changed"), namespace: "changed", websocketURL: nil, grant: .init(id: "changed", expiresAt: 1, epoch: 1, signature: "changed"))
        }
    }
}
private actor NormalizationTransport: MoldHTTPTransport {
    var paths: [String] = []
    var fingerprint = ""
    var mismatch = false
    var boundary = ""
    var admitted: [GenerateRequest] = []
    var priced: GenerateRequest?
    var expireOnPricing: NormalizationClock?
    var requireLicense = false
    var malformedNormalization = false
    func state() -> ([String], String) { (paths, boundary) }
    func submittedRequests() -> [GenerateRequest] { admitted }
    func pricedRequest() -> GenerateRequest? { priced }
    func requireConsent() { requireLicense = true }
    func advertiseMalformedNormalization() { malformedNormalization = true }
    func expireWhilePricing(_ clock: NormalizationClock?) { expireOnPricing = clock }
    func setMismatch() { mismatch = true }
    func response(for request: URLRequest, body: MoldRequestBody) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        let path = request.url!.path
        paths.append(request.httpMethod! + " " + path)
        let original = TransportDigest.sha256(Data([1, 2, 3]))
        let digest = String(repeating: "a", count: 64)
        let receipt: [String: Any] = ["reference": 1, "kind": "video", "original_sha256": original, "canonical_sha256": digest]
        var value: [String: Any]
        if path == "/api/capabilities" {
            let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/capabilities.json")
            value = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as! [String: Any]
            value["reference_uploads"] = VideoNormalizationPolicyTests.advertisement
            if malformedNormalization {
                var uploads = VideoNormalizationPolicyTests.advertisement
                uploads["video_normalization"] = ["available": true, "protocol_version": 1]
                value["reference_uploads"] = uploads
            }
        } else if path == "/api/status" {
            value = ["version": "test", "busy": false, "instance_id": "instance", "uptime_secs": 1]
        } else if path == "/api/generate/placement-preview" {
            if case let .data(bytes) = body { priced = try MoldJSON.decoder.decode(PlacementRequest.self, from: bytes).request }
            expireOnPricing?.advance()
            value = ["outcome": "planned"]
            if requireLicense {
                value["pending_downloads"] = [["kind": "model", "name": "test", "licenses": [["id": "required", "name": "Required terms", "url": "https://example.com/license", "canonical": "https://example.com/license", "sha256": String(repeating: "c", count: 64), "summary": "Review the terms"]]]]
            }
        } else if path == "/api/generation-batches" {
            if case let .data(bytes) = body { admitted = try MoldJSON.decoder.decode(BatchAdmission.self, from: bytes).requests }
            let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/batch-status.json")
            let listing = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as! [String: Any]
            value = (listing["batches"] as! [[String: Any]]).first!
        } else if request.httpMethod == "DELETE" { boundary = "delete"; value = [:] }
        else if path == "/api/reference-candidates" {
            boundary = "session"; fingerprint = request.value(forHTTPHeaderField: "x-mold-candidate-sha256")!
            value = ["protocol_version": 1, "instance_id": "instance", "session_id": "session-id", "candidate_scope_sha256": fingerprint,
                "expires_at_ms": HTTPBackend.referenceUploadNow + 60000, "session_handle": "session-secret", "session_complete": false,
                "uploads": [["reference": 1, "handle": "upload-secret"]]]
        } else if path == "/api/reference-upload" {
            boundary = "upload"
            value = ["protocol_version": 1, "instance_id": "instance", "session_id": "session-id", "candidate_scope_sha256": fingerprint,
                "reference": 1, "session_complete": false,
                "receipt": ["version": 1, "kind": "video", "original_sha256": original, "original_byte_length": 3,
                    "original_mime_type": "video/quicktime", "canonical_sha256": digest, "canonical_byte_length": 9,
                    "canonical_mime_type": "video/mp4", "converted": true],
                "metadata": ["index": 1, "kind": "video", "mime_type": "video/mp4", "sha256": digest, "name": "a.mov",
                    "width": 32, "height": 32, "frame_count": 8, "duration_ms": 2000, "fps": 4]]
        } else {
            boundary = "finalize"
            value = ["protocol_version": 1, "instance_id": "instance", "session_id": "session-id", "candidate_scope_sha256": fingerprint,
                "request_scope_sha256": String(repeating: "b", count: 64), "session_complete": true, "receipts": [receipt],
                "references": [["kind": "video", "media": ["authority": "descriptor"], "mime_type": "video/mp4",
                    "provenance": ["name": "a.mov", "sha256": digest], "width": mismatch ? 64 : 32, "height": 32,
                    "frame_count": 8, "duration_ms": 2000, "fps": 4]]]
        }
        let data = try JSONSerialization.data(withJSONObject: value)
        return (MoldResponseBytes(reader: NormalizationBytes(data)), HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: [:])!)
    }
}

struct VideoNormalizationTransportTests {
    @Test func malformedKnownV1RefusesPendingAdmissionWithoutLosingCapabilitiesOrPosting() async throws {
        let transport = NormalizationTransport(); let (backend, request, _) = try fixture(transport)
        await transport.advertiseMalformedNormalization()
        do {
            _ = try await backend.submit(BatchAdmission(requests: [request]))
            Issue.record("Malformed normalization was admitted")
        } catch MoldClientError.http(_, let code, _) {
            #expect(code == "REFERENCE_NORMALIZATION_INVALID")
        } catch { Issue.record("Expected the specific normalization contract error, received \(error)") }
        #expect(await transport.state().0 == ["GET /api/capabilities"])
    }
    private func fixture(_ transport: NormalizationTransport) throws -> (HTTPBackend, GenerateRequest, ReferenceUploadCapabilities) {
        let host = MoldHost(name: "machine", baseURL: URL(string: "http://normalizer")!, apiKey: "key")
        var request = GenerateRequest(prompt: "scene", model: "minimax-h3-ref2va:official-bf16", width: 1280, height: 720, steps: 30, guidance: 1)
        request.pendingReferenceInputs = [.freshVideo(.init(data: Data([1, 2, 3]), mimeType: "video/quicktime", name: "a.mov"))]
        return (HTTPBackend(host: host, transport: transport), request, try VideoNormalizationPolicyTests.decode(VideoNormalizationPolicyTests.advertisement))
    }
    @Test func explicitFinalizePublishesCanonicalLeaseWithoutChangingOriginal() async throws {
        let transport = NormalizationTransport(); let (backend, original, caps) = try fixture(transport)
        let lease = try await backend.prepareNormalizedReferences(original, capabilities: caps, expectedInstanceId: "instance")
        #expect(lease.request.pendingReferenceInputs == nil)
        #expect(lease.request.references?.first?.media == .init(authority: "upload", handle: "upload-secret"))
        #expect(lease.request.references?.first?.provenance?.sha256 == String(repeating: "a", count: 64))
        #expect(original.pendingReferenceInputs?.first?.freshVideo?.data == Data([1, 2, 3]))
        #expect(await transport.state().0 == ["POST /api/reference-candidates", "PUT /api/reference-upload", "POST /api/reference-finalize"])
        await lease.cancel(); await lease.cancel()
        #expect(await transport.state().0.filter { $0.hasPrefix("DELETE") }.count == 1)
    }
    @Test func ordinaryAdmissionFinalizesThenPricesAndPostsExactCanonicalRequest() async throws {
        let transport = NormalizationTransport(); let (backend, original, _) = try fixture(transport)
        do { _ = try await backend.submit(BatchAdmission(requests: [original])) }
        catch { Issue.record("Admission failed at \(await transport.state().0): \(error)"); throw error }
        let paths = await transport.state().0
        #expect(paths.firstIndex(of: "POST /api/reference-finalize")! < paths.firstIndex(of: "POST /api/generate/placement-preview")!)
        #expect(paths.firstIndex(of: "POST /api/generate/placement-preview")! < paths.firstIndex(of: "POST /api/generation-batches")!)
        let submitted = try #require(await transport.submittedRequests().first)
        #expect(submitted.pendingReferenceInputs == nil)
        #expect(submitted.prompt == original.prompt)
        #expect(submitted.references?.first?.frameCount == 8)
        #expect(submitted.references?.first?.media.handle == "upload-secret")
        let priced = try #require(await transport.pricedRequest())
        #expect(priced.prompt == "")
        #expect(priced.references?.first?.media.authority == "descriptor")
        #expect(priced.references?.first?.frameCount == submitted.references?.first?.frameCount)
        #expect(paths.last == "DELETE /api/reference-sessions")
    }
    @Test func consentAfterPreparationReleasesLeaseWithoutHoldingItAcrossModalOrPosting() async throws {
        let transport = NormalizationTransport(); let (backend, original, _) = try fixture(transport)
        await transport.requireConsent()
        do {
            _ = try await backend.submit(BatchAdmission(requests: [original]))
            Issue.record("Prepared request must wait for consent without retaining authority")
        } catch MoldClientError.licenseRequired(let refusal, _) { #expect(refusal.id == "required") }
        catch { Issue.record("Expected the existing licence recovery contract: \(error)") }
        #expect(!(await transport.state().0).contains("POST /api/generation-batches"))
        #expect(await transport.state().0.last == "DELETE /api/reference-sessions")
    }
    @Test func leaseExpiryDuringPricingReleasesUnusedAuthorityAndRetryPreparesFreshOriginals() async throws {
        let transport = NormalizationTransport(); let (originalBackend, original, _) = try fixture(transport)
        let clock = NormalizationClock()
        var backend = originalBackend
        backend.referenceUploadClock = { clock.now() }
        await transport.expireWhilePricing(clock)
        await #expect(throws: (any Error).self) { try await backend.submit(BatchAdmission(requests: [original])) }
        let refused = await transport.state().0
        #expect(refused.contains("POST /api/generate/placement-preview"))
        #expect(!refused.contains("POST /api/generation-batches"))
        #expect(refused.last == "DELETE /api/reference-sessions")
        clock.reset(); await transport.expireWhilePricing(nil)
        _ = try await backend.submit(BatchAdmission(requests: [original]))
        #expect(await transport.state().0.filter { $0 == "POST /api/reference-candidates" }.count == 2)
        #expect(original.pendingReferenceInputs?.first?.freshVideo?.data == Data([1, 2, 3]))
    }
    @Test func mismatchedFinalFactsReleaseSessionWithoutPublishingAuthority() async throws {
        let transport = NormalizationTransport(); await transport.setMismatch()
        let (backend, original, caps) = try fixture(transport)
        await #expect(throws: (any Error).self) { try await backend.prepareNormalizedReferences(original, capabilities: caps, expectedInstanceId: "instance") }
        #expect(await transport.state().0.last == "DELETE /api/reference-sessions")
    }
    @Test(arguments: ["key", "route", "mode", "instance", "secure pins"])
    func sameHostUUIDDoesNotAuthorizeChangedCredentialsOrPairingBetweenUploadAndFinalize(_ field: String) async throws {
        let transport = NormalizationTransport(); let (originalBackend, request, caps) = try fixture(transport)
        let originalHost = originalBackend.host
        let authority = NormalizationAuthority(originalHost)
        let backend = originalBackend.checkingReferenceAuthority {
            if await transport.state().1 == "upload" { await authority.change(field) }
            return await authority.isCurrent(originalHost)
        }
        await #expect(throws: (any Error).self) { try await backend.prepareNormalizedReferences(request, capabilities: caps, expectedInstanceId: "instance") }
        #expect(await transport.state().0.last == "DELETE /api/reference-sessions")
        #expect(!(await transport.state().0).contains("POST /api/reference-finalize"))
    }
    @Test(arguments: ["session", "upload", "finalize"])
    func changedAuthorityAtEveryAsyncBoundaryReleasesSession(_ boundary: String) async throws {
        let transport = NormalizationTransport(); let (originalBackend, request, caps) = try fixture(transport)
        let backend = originalBackend.checkingReferenceAuthority { await transport.state().1 != boundary }
        await #expect(throws: (any Error).self) { try await backend.prepareNormalizedReferences(request, capabilities: caps, expectedInstanceId: "instance") }
        #expect(await transport.state().0.last == "DELETE /api/reference-sessions")
        #expect(!(await transport.state().0).contains { $0.contains("generation-batches") })
    }
}
