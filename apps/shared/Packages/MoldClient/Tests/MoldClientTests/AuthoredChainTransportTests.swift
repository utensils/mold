import Foundation
import Testing
@testable import MoldClient

private final class AuthoredChainTransport: StubTransport {
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path.hasSuffix("/operations/op-1/cancel") { return (202, Data()) }
        if path == "/api/chain-jobs" { return (202, Data(#"{"job_id":"op-1","request_warnings":["Check collection"]}"#.utf8)) }
        return (200, Data(#"{"id":"a/b","state":"completed","model":"h3-test","stage_count":1,"current_stage":1,"stages":[{"idx":0,"state":"completed","seed":18446744073709551615}],"finalizes":[{"gallery_filename":"older.mp4"},{"gallery_filename":"latest.mp4"}]}"#.utf8))
    }
    override func startLoading() {
        Self.requests.append(request)
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&bytes, maxLength: bytes.count)
                guard count > 0 else { break }; data.append(contentsOf: bytes.prefix(count))
            }
        }
        Self.bodies.append(data)
        super.startLoading()
    }
}

@Suite(.serialized) struct AuthoredChainTransportTests {
    @Test func durableCreateUsesCapturedHostAndIdempotencyHeader() async throws {
        AuthoredChainTransport.requests = []
        let backend: any MoldBackend = AuthoredChainTransport.backend(apiKey: "private-key", host: "workstation")
        let fixture = AuthoredChainTests()
        let request = try AuthoredChainRequest(fixture.base(), stages: [.init(prompt: "One", frames: 209)], limits: fixture.limits())
        let accepted = try await backend.createChainJob(request, operationId: "op-1")
        #expect(accepted.jobId == "op-1")
        #expect(accepted.requestWarnings == ["Check collection"])
        let sent = try #require(AuthoredChainTransport.requests.last)
        #expect(sent.httpMethod == "POST")
        #expect(sent.url?.host == "workstation")
        #expect(sent.value(forHTTPHeaderField: "x-mold-operation-id") == "op-1")
        #expect(sent.value(forHTTPHeaderField: "X-Api-Key") == "private-key")
        #expect(sent.url?.query == nil)
    }
    @Test func explicitSourceExportKeepsQueryAndNumericSeed() async throws {
        AuthoredChainTransport.requests = []
        let backend = AuthoredChainTransport.backend()
        let detail = try await backend.chainJob(id: "a/b", includeSourceMedia: true)
        #expect(detail.stages?.first?.seed == UInt64.max)
        #expect(detail.latestFinalize?.take == 2)
        let sent = try #require(AuthoredChainTransport.requests.last)
        #expect(sent.url?.absoluteString.contains("a%2Fb?include_source_media=true") == true)
    }
    @Test func stopFencesUnansweredCreateThroughOperationRoute() async throws {
        AuthoredChainTransport.requests = []
        let backend: any MoldBackend = AuthoredChainTransport.backend()
        try await backend.cancelChainOperation(id: "job-1", operationId: "op-1")
        let sent = try #require(AuthoredChainTransport.requests.last)
        #expect(sent.httpMethod == "POST")
        #expect(sent.url?.path == "/api/chain-jobs/job-1/operations/op-1/cancel")
    }
    @Test func authoredHTTPPreservesLiteralBackslashesForServerNormalization() async throws {
        AuthoredChainTransport.requests = []; AuthoredChainTransport.bodies = []
        let fixture = AuthoredChainTests()
        let backend = AuthoredChainTransport.backend()
        let request = try AuthoredChainRequest(fixture.base(), stages: [.init(prompt: #"C:\new\render"#, frames: 209)], limits: fixture.limits())
        _ = try await backend.createChainJob(request, operationId: "op-1")
        let data = try #require(AuthoredChainTransport.bodies.last)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let stages = try #require(json["stages"] as? [[String: Any]])
        #expect(stages[0]["prompt"] as? String == #"C:\\new\\render"#)
        #expect(request.stages[0].prompt == #"C:\new\render"#)
    }

}
