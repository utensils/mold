import Foundation

@testable import MoldClient

/// A canned HTTP response table behind a `URLProtocol`, so a test never opens
/// a socket.
///
/// The table has to be STATIC: `URLSession` instantiates the protocol class
/// itself, so there is nowhere to hang per-instance state. swift-testing
/// serializes within a `@Suite(.serialized)` but NOT between suites, so two
/// suites sharing one subclass would race on that table -- and both would be
/// planting different answers for `/api/events`. Every suite gets its own
/// subclass, which is what `response(for:)` is for.
class StubTransport: URLProtocol {
    /// The subclass's own table. Dispatched dynamically, so each suite
    /// answers from its own storage.
    class func response(for path: String) -> (status: Int, body: Data)? { nil }

    /// Most of these routes are `text/event-stream`; a JSON route decodes the
    /// same either way, since nothing here reads the header.
    class var contentType: String { "text/event-stream" }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        guard let fixture = Self.response(for: path) else {
            client?.urlProtocol(self, didFailWithError: MoldClientError.malformedResponse)
            return
        }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: fixture.status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": Self.contentType])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: fixture.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

extension StubTransport {
    /// A backend whose every request is answered by this subclass.
    static func backend(apiKey: String? = nil, host: String = "stub") -> HTTPBackend {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [self]
        return HTTPBackend(
            host: MoldHost(name: host, baseURL: URL(string: "http://\(host):7680")!,
                           apiKey: apiKey),
            session: URLSession(configuration: config))
    }
}
