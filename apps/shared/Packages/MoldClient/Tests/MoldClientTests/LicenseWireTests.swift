import Foundation
import Testing

@testable import MoldClient

// Pins the licence wire contract: the listing a sheet renders from, the
// acceptance it sends back, and a 403/409 refusal decoded into something the
// app can resolve rather than a dead "answered with an error" sentence.

private func listing() throws -> LicenseListing {
    try MoldJSON.decoder.decode(LicenseListing.self, from: RepoFixtures.fixture("licenses-workstation.json"))
}

@Test func aLicenceListingDecodesEveryFieldIncludingTheFriendlyStyles() throws {
    let licenses = try listing().licenses
    #expect(licenses.count == 3)
    #expect(licenses.allSatisfy { !($0.requiredByStyles ?? []).isEmpty })
    let hunyuan = try #require(licenses.first { $0.id == "tencent-hunyuan3d-2.0" })
    #expect(hunyuan.accepted == true)
    #expect(hunyuan.requiredBy.contains("hunyuan3d:fp16"))
    #expect(hunyuan.requiredByStyles?.contains { $0.name == "hunyuan3d:fp16" } == true)
    #expect(!hunyuan.url.isEmpty)
    #expect(!hunyuan.canonical.isEmpty)
    #expect(!hunyuan.sha256.isEmpty)
    #expect(!hunyuan.summary.isEmpty)
}

@Test func anAcceptanceCarriesTheTermsAndNeverAnAcceptedFlag() throws {
    let license = try #require(listing().licenses.first)
    let data = try MoldJSON.encoder.encode(license.acceptance)
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(Set(object.keys) == ["id", "url", "sha256"])
}

@Test func anOlderHostWithoutTheFriendlyStylesStillDecodes() throws {
    let json = Data("""
    {"licenses":[{"id":"x","name":"X","url":"https://x","canonical":"https://x",
    "sha256":"abc","summary":"s","accepted":false,"required_by":["m"]}]}
    """.utf8)
    let decoded = try MoldJSON.decoder.decode(LicenseListing.self, from: json)
    #expect(decoded.licenses.first?.requiredByStyles == nil)
    #expect(decoded.licenses.first?.accepted == false)
}

// Stub transport, separate from `StreamTests`' `StubURLProtocol` -- that
// suite documents itself as the one place mutating its own static table, so
// a second suite touching the same one would race it under parallel tests.
private final class LicenseStubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var responses: [String: (status: Int, body: Data)] = [:]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let fixture = Self.responses[request.url?.path ?? ""] else {
            client?.urlProtocol(self, didFailWithError: MoldClientError.malformedResponse)
            return
        }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: fixture.status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: fixture.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private func stubbedBackend() -> HTTPBackend {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [LicenseStubURLProtocol.self]
    return HTTPBackend(
        host: MoldHost(name: "stub", baseURL: URL(string: "http://stub:7680")!),
        session: URLSession(configuration: config))
}

/// `ApiError::license_not_accepted`'s real shape (`routes.rs:136-148`).
private let refusalJSON = """
{"error":"accept the terms first","code":"LICENSE_NOT_ACCEPTED","license":{"id":"tencent-hunyuan3d-2.0",\
"name":"Tencent Hunyuan 3D 2.0 Community License","url":"https://raw.githubusercontent.com/x/LICENSE",\
"canonical":"https://github.com/x","sha256":"9425deadbeef","summary":"Tencent Hunyuan 3D 2.0 weights."}}
"""

@Suite(.serialized)
struct LicenseWireTests {
    @Test func aRefusedDownloadArrivesAsALicenceTheAppCanAccept() async throws {
        LicenseStubURLProtocol.responses["/api/downloads"] = (403, Data(refusalJSON.utf8))
        let backend = stubbedBackend()
        await #expect {
            _ = try await backend.startDownload(DownloadRequest(model: "hunyuan3d:fp16"))
        } throws: { error in
            guard case let MoldClientError.licenseRequired(refusal, mismatch) = error else { return false }
            return refusal.id == "tencent-hunyuan3d-2.0" && !refusal.sha256.isEmpty && !mismatch
        }
    }

    @Test func aTermsMismatchIsTheSamePayloadAndADifferentSentence() async throws {
        let mismatch = refusalJSON.replacingOccurrences(of: "LICENSE_NOT_ACCEPTED", with: "LICENSE_TERMS_MISMATCH")
        LicenseStubURLProtocol.responses["/api/downloads"] = (409, Data(mismatch.utf8))
        let backend = stubbedBackend()
        await #expect {
            _ = try await backend.startDownload(DownloadRequest(model: "hunyuan3d:fp16"))
        } throws: { error in
            guard case let MoldClientError.licenseRequired(refusal, mismatch) = error else { return false }
            return refusal.id == "tencent-hunyuan3d-2.0" && mismatch
        }
    }

    @Test func anErrorWithNoCodeKeepsTheMachinesOwnWords() async throws {
        let body = #"{"error":"unknown model 'x'. Run 'mold list' to see available models."}"#
        LicenseStubURLProtocol.responses["/api/downloads"] = (400, Data(body.utf8))
        let backend = stubbedBackend()
        await #expect {
            _ = try await backend.startDownload(DownloadRequest(model: "x"))
        } throws: { error in
            guard case let MoldClientError.http(status, code, message) = error else { return false }
            return status == 400 && code == nil
                && message == "unknown model 'x'. Run 'mold list' to see available models."
        }
    }

    @Test func aPlainTextRefusalIsQuotedAndAnHTMLPageIsNot() async throws {
        LicenseStubURLProtocol.responses["/api/downloads"] = (400, Data("id must be `cv:` or `hf:` prefixed".utf8))
        await #expect {
            _ = try await stubbedBackend().startDownload(DownloadRequest(model: "x"))
        } throws: { error in
            guard case let MoldClientError.http(_, _, message) = error else { return false }
            return message == "id must be `cv:` or `hf:` prefixed"
        }

        LicenseStubURLProtocol.responses["/api/downloads"] = (502, Data("<html><body>Bad Gateway</body></html>".utf8))
        await #expect {
            _ = try await stubbedBackend().startDownload(DownloadRequest(model: "x"))
        } throws: { error in
            guard case let MoldClientError.http(_, _, message) = error else { return false }
            return message == nil
        }
    }
}
