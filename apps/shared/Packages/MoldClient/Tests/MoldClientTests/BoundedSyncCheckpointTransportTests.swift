import Foundation
import Synchronization
import Testing
@testable import MoldClient

/// Answers the bounded POST (or refuses it, as an older server does) and the
/// whole GET, recording which was asked.
private final class BoundedCheckpointStub: URLProtocol {
    struct State {
        var postStatus = 200
        var asked: [String] = []
        var postedNames: [[String]] = []
    }
    static let state = Mutex(State())

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let method = request.httpMethod ?? "GET"
        let body = request.httpBody ?? request.httpBodyStream.map(Self.read) ?? Data()
        let (status, data): (Int, Data) = Self.state.withLock { state in
            state.asked.append(method)
            guard method == "POST" else {
                return (200, Data(#"{"protocol_version":1,"instance_id":"i","revisions":{"a":"ra","b":"rb","c":"rc"}}"#.utf8))
            }
            let names = (try? JSONSerialization.jsonObject(with: body) as? [String: [String]])?["filenames"] ?? []
            state.postedNames.append(names)
            guard state.postStatus == 200 else { return (state.postStatus, Data(#"{"error":"no"}"#.utf8)) }
            let revisions = Dictionary(uniqueKeysWithValues: names.map { ($0, "r" + $0) })
            let json = try! JSONSerialization.data(withJSONObject: [
                "protocol_version": 1, "instance_id": "i", "revisions": revisions,
            ])
            return (200, json)
        }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
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

    static func backend() -> HTTPBackend {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BoundedCheckpointStub.self]
        return HTTPBackend.testing(host: .init(name: "Fixture", baseURL: URL(string: "http://fixture")!),
                                   session: URLSession(configuration: configuration))
    }
}

@Suite(.serialized)
struct BoundedSyncCheckpointTransportTests {
    @Test func touchedNamesAreAskedWithTheBoundedPost() async throws {
        BoundedCheckpointStub.state.withLock { $0 = .init() }
        let checkpoint = try #require(try await BoundedCheckpointStub.backend()
            .gallerySyncCheckpoint(filenames: ["b", "a", "b"]))
        #expect(checkpoint.revisions == ["a": "ra", "b": "rb"])
        #expect(BoundedCheckpointStub.state.withLock { $0.asked } == ["POST"])
        #expect(BoundedCheckpointStub.state.withLock { $0.postedNames } == [["a", "b"]])
    }

    @Test func manyNamesAreChunkedToWhatTheServerAccepts() async throws {
        BoundedCheckpointStub.state.withLock { $0 = .init() }
        let names = (0..<(HTTPBackend.boundedCheckpointLimit + 5)).map { "n\($0)" }
        let checkpoint = try #require(try await BoundedCheckpointStub.backend().gallerySyncCheckpoint(filenames: names))
        #expect(checkpoint.revisions.count == names.count)
        #expect(BoundedCheckpointStub.state.withLock { $0.postedNames.map(\.count) }
            == [HTTPBackend.boundedCheckpointLimit, 5])
    }

    @Test(arguments: [404, 405])
    func anOlderServerFallsBackToTheWholeGetRestrictedToTheNames(status: Int) async throws {
        BoundedCheckpointStub.state.withLock { $0 = .init(); $0.postStatus = status }
        let checkpoint = try #require(try await BoundedCheckpointStub.backend().gallerySyncCheckpoint(filenames: ["a", "c"]))
        #expect(checkpoint.revisions == ["a": "ra", "c": "rc"])
        #expect(BoundedCheckpointStub.state.withLock { $0.asked } == ["POST", "GET"])
    }

    @Test func anotherFailureIsNotAFallback() async throws {
        BoundedCheckpointStub.state.withLock { $0 = .init(); $0.postStatus = 500 }
        await #expect(throws: (any Error).self) {
            _ = try await BoundedCheckpointStub.backend().gallerySyncCheckpoint(filenames: ["a"])
        }
        #expect(BoundedCheckpointStub.state.withLock { $0.asked } == ["POST"])
    }
}
