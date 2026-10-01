import Foundation
import Testing

@testable import MoldClient

private final class RelayTransportStub: StubTransport {
    static let signedURL =
        "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com/_mold/objects/clip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature="
        + String(repeating: "a", count: 64) + "&X-Amz-Expires=900"
    nonisolated(unsafe) static var requests: [URLRequest] = []
    nonisolated(unsafe) static var infoStatus = 200
    nonisolated(unsafe) static var grantBody: Data?
    override class func response(for path: String) -> (status: Int, body: Data)? {
        let json: String
        switch path {
        case "/_mold/relay/info":
            if infoStatus != 200 { return (infoStatus, Data()) }
            json =
                #"{"protocol":2,"upload_threshold":2097152,"max_body_bytes":67108864,"object_origin":"https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com"}"#
        case "/_mold/relay/uploads":
            json =
                #"{"id":"up1","url":"https://mold-relay-123456789012-us-east-1.s3.us-east-1.amazonaws.com/object?signature=short","headers":{},"expires_at":9999999999}"#
        case "/api/gallery/media-token":
            json = #"{"auth_required":true,"relay":{"id":"media1","state":"pending"}}"#
        case "/_mold/relay/media/media1":
            json = #"{"state":"ready","url":"\#(signedURL)","expires_at":9999999999}"#
        default: json = "{}"
        }
        return (200, Data(json.utf8))
    }
    override func startLoading() {
        Self.requests.append(request)
        if request.url?.path == "/_mold/relay/uploads" {
            Self.grantBody = request.httpBody
            if Self.grantBody == nil, let stream = request.httpBodyStream {
                stream.open()
                defer { stream.close() }
                var body = Data()
                var buffer = [UInt8](repeating: 0, count: 4096)
                while stream.hasBytesAvailable {
                    let count = stream.read(&buffer, maxLength: buffer.count)
                    if count <= 0 { break }
                    body.append(buffer, count: count)
                }
                Self.grantBody = body
            }
        }
        if request.url?.path == "/api/object" {
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["x-mold-relay-object": "1"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(
                self,
                didLoad: try! JSONSerialization.data(withJSONObject: [
                    "url": Self.signedURL, "status": 206, "headers": ["content-range": "bytes 0-1/10"],
                ]))
            client?.urlProtocolDidFinishLoading(self)
        } else if request.url?.path == "/api/events" {
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/event-stream", "x-mold-relay-protocol": "2"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data("event: hello\ndata: {}\n\n".utf8))
            client?.urlProtocolDidFinishLoading(self)
        } else {
            super.startLoading()
        }
    }
}
@Suite(.serialized)
struct RelayTransportTests {
    private func backend() -> HTTPBackend {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RelayTransportStub.self]
        return HTTPBackend(
            host: MoldHost(name: "relay", baseURL: URL(string: "https://relay.example")!, apiKey: "secret"),
            session: URLSession(configuration: config))
    }
    @Test(arguments: ["PATCH", "DELETE"]) func arbitraryLargeBodiesStageWithoutSendingTheMoldKeyToS3(method: String) async throws {
        RelayTransportStub.requests = []
        let backend = backend()
        var request = backend.request("/api/any-future-route?value=1")
        request.httpMethod = method
        request.httpBody = Data(repeating: 7, count: 2_097_153)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer credential", forHTTPHeaderField: "Authorization")
        request.setValue("credential=secret", forHTTPHeaderField: "Cookie")
        request.setValue("keep-alive, x-hop-secret", forHTTPHeaderField: "Connection")
        request.setValue("connection-private", forHTTPHeaderField: "X-Hop-Secret")
        request.setValue("private", forHTTPHeaderField: "X-Mold-Viewer-Secret")
        request.setValue("operation-1", forHTTPHeaderField: "X-Mold-Operation-Id")
        _ = try await backend.send(request)
        #expect(
            RelayTransportStub.requests.map { $0.url!.path }.filter { $0 != "/_mold/relay/info" } == [
                "/_mold/relay/uploads", "/object", "/_mold/relay/request",
            ])
        let grantRequest = try #require(RelayTransportStub.requests.first { $0.url?.path == "/_mold/relay/uploads" })
        let grantBody = try #require(RelayTransportStub.grantBody)
        #expect(grantRequest.value(forHTTPHeaderField: "X-Api-Key") == "secret")
        let metadata = try #require(JSONSerialization.jsonObject(with: grantBody) as? [String: Any])
        let metadataHeaders = try #require(metadata["headers"] as? [String: String])
        let normalized = Dictionary(uniqueKeysWithValues: metadataHeaders.map { ($0.key.lowercased(), $0.value) })
        // The TypeScript integration test submits this same wire metadata to the actual cloud validator.
        #expect(normalized == ["content-type": "application/octet-stream", "x-mold-operation-id": "operation-1"])
        let upload = try #require(
            RelayTransportStub.requests.first {
                $0.url?.host == "mold-relay-123456789012-us-east-1.s3.us-east-1.amazonaws.com"
            })
        #expect(upload.value(forHTTPHeaderField: "X-Api-Key") == nil)
        #expect(upload.httpMethod == "PUT")
        let commit = try #require(RelayTransportStub.requests.last)
        #expect(commit.value(forHTTPHeaderField: "X-Api-Key") == "secret")
        #expect(commit.value(forHTTPHeaderField: "x-amz-content-sha256") != nil)
    }
    @Test func discoveryFallsBackOnlyOn404AndNeverOn503() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [RelayTransportStub.self]
        let session = URLSession(configuration: configuration)
        let discovery = RelayDiscovery()
        defer { RelayTransportStub.infoStatus = 200 }
        RelayTransportStub.infoStatus = 404
        let legacy = try await discovery.info(origin: URL(string: "https://legacy.example")!, session: session)
        #expect(legacy.protocol != 2)
        RelayTransportStub.infoStatus = 503
        await #expect(throws: (any Error).self) {
            try await discovery.info(origin: URL(string: "https://unavailable.example")!, session: session)
        }
    }
    @Test func pendingMediaResolvesToTheSignedSameOriginURL() async throws {
        let url = try await backend().playableURL(for: "clip.mp4")
        #expect(url.path == "/_mold/objects/clip")
        #expect(!url.absoluteString.contains("secret"))
    }
    @Test func gracefulRelayStreamEOFMakesLossVisibleAndReopensGET() async throws {
        RelayTransportStub.requests = []
        var iterator = backend().stream("/api/events", timeout: 3600).makeAsyncIterator()
        #expect(try await iterator.next()?.name == "hello")
        #expect(try await iterator.next()?.name == "resync_required")
        #expect(try await iterator.next()?.name == "hello")
        #expect(RelayTransportStub.requests.count == 2)
    }
    @Test func emptyHTTPSMutationCarriesOCAPayloadDigest() async throws {
        RelayTransportStub.requests = []
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RelayTransportStub.self]
        let backend = HTTPBackend(
            host: MoldHost(name: "relay", baseURL: URL(string: "https://relay.example")!, apiKey: "secret"),
            session: URLSession(configuration: config))
        var request = backend.request("/api/anything")
        request.httpMethod = "POST"
        _ = try await backend.send(request)
        let sent = try #require(RelayTransportStub.requests.first)
        #expect(
            sent.value(forHTTPHeaderField: "x-amz-content-sha256")
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        #expect(RelayTransportStub.requests.count == 1)
    }
    @Test func relayObjectsMustStayOnTheExactOriginAndReservedPath() throws {
        let origin = URL(string: "https://relay.example")!
        #expect(throws: (any Error).self) {
            try RelayTransport.objectURL("https://evil.example/_mold/objects/x", origin: origin)
        }
        #expect(throws: (any Error).self) {
            try RelayTransport.objectURL("https://relay.example/api/x", origin: origin)
        }
        #expect(
            try RelayTransport.objectURL(
                "https://relay.example/_mold/objects/x?Signature=short", origin: origin
            ).host == origin.host)
    }
    @Test func S3ObjectsMustBeSignedAndBelongToThePinnedBucket() throws {
        let origin = URL(string: "https://relay.example")!
        let objectOrigin = "https://mold-relay-123456789012-us-east-1.s3.dualstack.us-east-1.amazonaws.com"
        let signed =
            objectOrigin + "/_mold/objects/a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature="
            + String(repeating: "a", count: 64) + "&X-Amz-Expires=900"
        #expect(
            try RelayTransport.objectURL(signed, origin: origin, objectOrigin: objectOrigin).host
                == URL(string: objectOrigin)?.host)
        for url in [
            signed.replacingOccurrences(of: "123456789012", with: "999999999999"),
            signed.replacingOccurrences(of: "Expires=900", with: "Expires=901"),
            signed.replacingOccurrences(of: "X-Amz-Signature=", with: "unsigned="), signed + "#fragment",
        ] {
            #expect(throws: (any Error).self) {
                try RelayTransport.objectURL(url, origin: origin, objectOrigin: objectOrigin)
            }
        }
    }
    @Test func HTTPSRequestTargetsPreserveEncodingAndRepeatedQueries() async throws {
        RelayTransportStub.requests = []
        _ = try await backend().send(
            backend().request("/api/gallery/image/a%20b.png?media_token=x%2By&part=1&part=2"))
        let request = try #require(RelayTransportStub.requests.first)
        #expect(
            request.value(forHTTPHeaderField: "x-mold-request-target")
                == "/api/gallery/image/a%20b.png?media_token=x%2By&part=1&part=2")
    }
    @Test func signedS3ObjectHandshakeNeverForwardsMoldHeaders() async throws {
        RelayTransportStub.requests = []
        let backend = backend()
        let (data, response) = try await backend.send(backend.request("/api/object"))
        #expect(data == Data("{}".utf8))
        #expect(response.statusCode == 206)
        #expect(response.value(forHTTPHeaderField: "Content-Range") == "bytes 0-1/10")
        let downloaded = try #require(RelayTransportStub.requests.last)
        #expect(downloaded.url?.host?.contains("s3.dualstack") == true)
        #expect(downloaded.value(forHTTPHeaderField: "X-Api-Key") == nil)
        #expect(downloaded.value(forHTTPHeaderField: "x-mold-request-target") == nil)
        let (stream, streamedResponse) = try await backend.relayBytes(backend.request("/api/object"))
        #expect(streamedResponse.statusCode == 206)
        #expect(try await stream.collected(upTo: 100) == Data("{}".utf8))
    }
    @Test func fileBackedUploadsUseTheSameGenericStagingContract() async throws {
        RelayTransportStub.requests = []
        let backend = backend()
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mold-relay-upload-\(UUID().uuidString)")
        try Data(repeating: 9, count: 2_097_153).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        var request = backend.request("/api/future-file-upload")
        request.httpMethod = "PUT"
        let data = try await backend.upload(request, fromFile: file)
        #expect(data == Data("{}".utf8))
        let upload = try #require(
            RelayTransportStub.requests.first { $0.url?.host?.contains("s3.us-east-1") == true })
        #expect(upload.value(forHTTPHeaderField: "X-Api-Key") == nil)
        #expect(upload.value(forHTTPHeaderField: "x-mold-request-target") == nil)
        #expect(RelayTransportStub.requests.last?.url?.path == "/_mold/relay/request")
    }
}
