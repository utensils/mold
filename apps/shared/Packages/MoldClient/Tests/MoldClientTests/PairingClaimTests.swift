import Foundation
import Testing

@testable import MoldClient

/// Redeeming a pairing code (`POST /api/pairing/claim`, `routes.rs`): the one
/// unauthenticated route that returns a durable key. It must go out with NO
/// key, read a 401 as "expired or used" rather than "needs a key", and refuse
/// a key from any machine other than the one the code names.
@Suite(.serialized)
struct PairingClaimTests {
    private let payload = try! MobilePairingPayload.parse(
        "mold://pair?version=1&base_url=http%3A%2F%2Fbox%3A7680&token=tok&expires_at=4000000000&instance_id=inst-1&name=box")

    private func session(status: Int, body: String) -> URLSession {
        ClaimStubURLProtocol.status = status
        ClaimStubURLProtocol.body = Data(body.utf8)
        ClaimStubURLProtocol.seen = nil
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ClaimStubURLProtocol.self]
        return URLSession(configuration: config)
    }

    @Test func aClaimReturnsTheKeyAndSendsNoKeyOfItsOwn() async throws {
        let session = session(status: 200, body: #"{"api_key":"mold_pair_x","instance_id":"inst-1","hostname":"box"}"#)
        let claim = try await HTTPBackend.claimPairing(payload, clientName: "James's iPhone",
                                                       clientKind: "iphone", session: session)
        #expect(claim.apiKey == "mold_pair_x")
        #expect(claim.hostname == "box")
        let request = try #require(ClaimStubURLProtocol.seen)
        #expect(request.url?.absoluteString == "http://box:7680/api/pairing/claim")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == nil)
        let body = try #require(ClaimStubURLProtocol.seenBody)
        let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: String])
        #expect(object == ["token": "tok", "client_name": "James's iPhone", "client_kind": "iphone"])
    }

    @Test func aUsedOrExpiredCodeSaysSo() async {
        let session = session(status: 401, body: #"{"error":"pairing token is missing, expired, or already used","code":"PAIRING_TOKEN_INVALID"}"#)
        await #expect(throws: PairingClaimError.expiredOrUsed) {
            _ = try await HTTPBackend.claimPairing(payload, clientName: "p", clientKind: "iphone", session: session)
        }
    }

    @Test func aKeyFromAnotherMachineIsRefused() async {
        let session = session(status: 200, body: #"{"api_key":"k","instance_id":"someone-else","hostname":"x"}"#)
        await #expect(throws: PairingClaimError.wrongMachine) {
            _ = try await HTTPBackend.claimPairing(payload, clientName: "p", clientKind: "iphone", session: session)
        }
    }

    @Test func aKeylessMachineAnswersWithNoKey() async throws {
        let session = session(status: 200, body: #"{"api_key":null,"instance_id":"inst-1","hostname":"box"}"#)
        let claim = try await HTTPBackend.claimPairing(payload, clientName: "p", clientKind: "ipad", session: session)
        #expect(claim.apiKey == nil)
    }

    @Test func anExpiredCodeIsRefusedWithoutAsking() async {
        let stale = try! MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox&token=t&expires_at=1&instance_id=i&name=b")
        let session = session(status: 200, body: "{}")
        await #expect(throws: PairingClaimError.expiredOrUsed) {
            _ = try await HTTPBackend.claimPairing(stale, clientName: "p", clientKind: "iphone", session: session)
        }
        #expect(ClaimStubURLProtocol.seen == nil)
    }
}

private final class ClaimStubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var status = 200
    nonisolated(unsafe) static var body = Data()
    nonisolated(unsafe) static var seen: URLRequest?
    nonisolated(unsafe) static var seenBody: Data?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.seen = request
        Self.seenBody = request.httpBody ?? request.httpBodyStream.map(Self.read)
        let response = HTTPURLResponse(url: request.url!, statusCode: Self.status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func read(_ stream: InputStream) -> Data {
        stream.open(); defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }
}
