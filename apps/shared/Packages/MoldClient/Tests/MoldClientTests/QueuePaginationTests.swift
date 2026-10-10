import Foundation
import Testing
@testable import MoldClient

@Suite(.serialized)
struct QueuePaginationTests {
    private func backend(_ bodies: [String]) -> HTTPBackend {
        QueuePageStub.bodies = bodies
        QueuePageStub.requests = []
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [QueuePageStub.self]
        return HTTPBackend(host: MoldHost(name: "box", baseURL: URL(string: "http://box/mold")!),
                           session: URLSession(configuration: configuration))
    }

    @Test func followsPagesAndDeduplicatesLiveRows() async throws {
        let client = backend([
            #"{"entries":[{"id":"one","state":"queued"},{"id":"live","state":"running"}],"page":{"next_cursor":"a+b/c="}}"#,
            #"{"entries":[{"id":"two","state":"held"}],"live_only_entries":[{"id":"live","state":"running"}],"page":{"next_cursor":"last"}}"#,
            #"{"entries":[{"id":"three","state":"paused"}],"live_only_entries":[{"id":"live","state":"cancelling"}],"page":{"next_cursor":null}}"#,
        ])
        let listing = try await client.queue()
        #expect(Set(listing.merged.map(\.id)) == ["one", "two", "three", "live"])
        #expect(listing.merged.count == 4)
        #expect(listing.merged.first { $0.id == "live" }?.state == .cancelling)
        #expect(QueuePageStub.requests.count == 3)
        let second = try #require(QueuePageStub.requests.dropFirst().first)
        #expect(URLComponents(url: second, resolvingAgainstBaseURL: false)?.queryItems?.first?.value == "a+b/c=")
    }

    @Test func legacyListingNeedsOneRequest() async throws {
        let listing = try await backend([#"{"entries":[{"id":"old","state":"queued"}]}"#]).queue()
        #expect(listing.merged.count == 1)
        #expect(QueuePageStub.requests.count == 1)
    }

    @Test func laterMalformedPageRejectsIncompleteListing() async {
        let client = backend([
            #"{"entries":[{"id":"one","state":"queued"}],"page":{"next_cursor":"next"}}"#,
            "invalid JSON",
        ])
        await #expect(throws: MoldClientError.self) { try await client.queue() }
    }

    @Test func repeatedCursorRejectsIncompleteListing() async {
        let client = backend([
            #"{"entries":[],"page":{"next_cursor":"same"}}"#,
            #"{"entries":[],"page":{"next_cursor":"same"}}"#,
        ])
        await #expect(throws: MoldClientError.self) { try await client.queue() }
        #expect(QueuePageStub.requests.count == 2)
    }
}

private final class QueuePageStub: URLProtocol {
    nonisolated(unsafe) static var bodies: [String] = []
    nonisolated(unsafe) static var requests: [URL] = []
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requests.append(request.url!)
        let body = Self.bodies.isEmpty ? #"{"entries":[]}"# : Self.bodies.removeFirst()
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
