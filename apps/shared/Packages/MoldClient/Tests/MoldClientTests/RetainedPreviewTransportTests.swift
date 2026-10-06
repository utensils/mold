import Foundation
import Testing
@testable import MoldClient

private final class RetainedPreviewStub: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let declared = request.url!.path.contains("declared")
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: declared ? ["Content-Length": "2097153"] : [:])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(repeating: 0, count: declared ? 1 : 2097153))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

struct RetainedPreviewTransportTests {
    @Test(arguments: ["declared", "chunked"])
    func bothPreviewRoutesRefuseOversizedResponses(member: String) async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RetainedPreviewStub.self]
        let backend = HTTPBackend(host: .init(name: "Fixture", baseURL: URL(string: "http://fixture")!),
            session: URLSession(configuration: config))
        for thumbnail in [true, false] {
            do {
                if thumbnail { _ = try await backend.retainedSourceMediaThumbnail(for: "clip.mp4", member: member) }
                else { _ = try await backend.retainedSourceMediaPreviewBytes(for: "clip.mp4", member: member) }
                Issue.record("Oversized preview was accepted")
            } catch let error as ResponseCeiling.Exceeded {
                #expect(error.ceiling == 2 * 1024 * 1024)
            }
        }
    }
}
