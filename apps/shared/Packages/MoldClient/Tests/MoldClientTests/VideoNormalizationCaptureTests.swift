import Foundation
import Testing
@testable import MoldClient

struct VideoNormalizationCaptureTests {
    func fixture() throws -> (GenerateRequest, ReferenceUploadCapabilities) {
        var request = GenerateRequest(prompt: "scene", model: "minimax-h3-ref2va:official-bf16", width: 1280, height: 720, steps: 30, guidance: 1)
        request.pendingReferenceInputs = [
            .canonical(.init(kind: "image", media: .init(authority: "inline", data: "AQID"), mimeType: "image/png", provenance: .init(name: "a.png"), width: 1, height: 1)),
            .freshVideo(.init(data: Data([4, 5, 6]), mimeType: "video/quicktime", name: "picked.mov"))]
        return (request, try VideoNormalizationPolicyTests.decode(VideoNormalizationPolicyTests.advertisement))
    }
    @Test func candidateBodyContainsOrderedDescriptorsAndExactOriginalFactsOnly() throws {
        let (request, caps) = try fixture()
        let captured = try VideoNormalizationCapture.capture(request, capabilities: caps, apiKey: "key")
        let json = try #require(JSONSerialization.jsonObject(with: captured.body) as? [String: Any])
        let wire = try #require(json["request"] as? [String: Any])
        #expect(wire["references"] == nil)
        let slots = try #require(json["reference_candidates"] as? [[String: Any]])
        #expect(slots.count == 2)
        #expect(slots[0]["type"] as? String == "canonical")
        #expect(slots[1]["type"] as? String == "fresh_video")
        #expect(slots[1]["width"] == nil)
        let original = try #require(slots[1]["original"] as? [String: Any])
        #expect(original["byte_length"] as? Int == 3)
        #expect(original["sha256"] as? String == TransportDigest.sha256(Data([4, 5, 6])))
        #expect(captured.entries.map(\.index) == [1, 2])
        var domain = Data("mold-reference-normalization-candidate-v1\0".utf8); domain.append(captured.body)
        #expect(captured.fingerprint == TransportDigest.sha256(domain))
    }
    @Test func spentHandlesAndChangedOriginalDigestsRefuseBeforeAnySession() throws {
        let (base, caps) = try fixture()
        var request = base
        request.pendingReferenceInputs?[0] = .canonical(.init(kind: "image", media: .init(authority: "upload", handle: "spent"), mimeType: "image/png"))
        #expect(throws: (any Error).self) { try VideoNormalizationCapture.capture(request, capabilities: caps, apiKey: "key") }
        request = base
        request.pendingReferenceInputs?[0] = .canonical(.init(kind: "image", media: .init(authority: "inline", data: "AQID"), mimeType: "image/png", provenance: .init(sha256: String(repeating: "0", count: 64))))
        #expect(throws: (any Error).self) { try VideoNormalizationCapture.capture(request, capabilities: caps, apiKey: "key") }
    }
}

@Test func nonUploadCandidatePreservesCapturedServerPathUntilFinalizedDescriptorValidation() throws {
    var (request, caps) = try VideoNormalizationCaptureTests().fixture()
    request.pendingReferenceInputs?[0] = .canonical(.init(kind: "image", media: .init(authority: "server_path", path: "/private/reference.png"),
        mimeType: "image/png", provenance: .init(sha256: String(repeating: "a", count: 64)), width: 1, height: 1))
    let capture = try VideoNormalizationCapture.capture(request, capabilities: caps, apiKey: "key")
    let json = try #require(JSONSerialization.jsonObject(with: capture.body) as? [String: Any])
    let candidate = try #require((json["reference_candidates"] as? [[String: Any]])?.first)
    let reference = try #require(candidate["reference"] as? [String: Any])
    #expect((reference["media"] as? [String: Any])?["authority"] as? String == "server_path")
    #expect((reference["media"] as? [String: Any])?["path"] as? String == "/private/reference.png")
    #expect(candidate["upload"] as? Bool == false)
}
