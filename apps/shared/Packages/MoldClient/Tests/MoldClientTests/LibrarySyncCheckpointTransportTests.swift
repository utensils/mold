import Foundation
import Testing
@testable import MoldClient

private final class SyncCheckpointStub: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let headers = ["Content-Length": "16777217"]
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data([0]))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

struct LibrarySyncCheckpointTransportTests {
    @Test func oversizedCheckpointFallsBackToOrdinaryVerification() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SyncCheckpointStub.self]
        let backend = HTTPBackend(host: .init(name: "Fixture", baseURL: URL(string: "http://fixture")!), session: URLSession(configuration: configuration))
        #expect(try await backend.gallerySyncCheckpoint() == nil)
    }
}
