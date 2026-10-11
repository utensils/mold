import Foundation

extension HTTPBackend {
    /// Every response, including callers with a separate thumbnail cache, crosses this boundary.
    public static func responseBytes(for request: URLRequest, host: MoldHost, session: URLSession)
        async throws -> (MoldResponseBytes, HTTPURLResponse) {
        try await HTTPBackend(host: host, session: session).relayBytes(request)
    }

    func response(_ request: URLRequest, body: MoldRequestBody? = nil) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        let body = body ?? .data(request.httpBody ?? Data())
        if let injectedTransport { return try await injectedTransport.response(for: request, body: body) }
        if host.secure == nil, Self.isEmbeddedLoopback(host.baseURL) {
            return try await LoopbackHTTPTransport(session: session).response(for: request, body: body)
        }
        guard host.secure != nil else { throw SecureConnectionError.pairingRequired }
        return try await SecureTransportPool.shared.response(request, body: body, host: host)
    }

    static func isEmbeddedLoopback(_ url: URL) -> Bool {
        ["http", "https"].contains(url.scheme?.lowercased() ?? "") &&
            ["127.0.0.1", "[::1]", "::1"].contains(url.host?.lowercased() ?? "") &&
            url.user == nil && url.password == nil
    }

    func relayBytes(_ request: URLRequest) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        try await response(request)
    }

    /// Download straight into a caller-owned private temporary file, without staging a
    /// plaintext cloud object or retaining a complete response in memory.
    func relayDownload(_ request: URLRequest) async throws -> (URL, HTTPURLResponse) {
        let (bytes, http) = try await response(request)
        let file = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        guard FileManager.default.createFile(atPath: file.path, contents: nil, attributes: [.posixPermissions: 0o600])
        else { await bytes.cancel(); throw CocoaError(.fileWriteUnknown) }
        var succeeded = false
        defer { if !succeeded { try? FileManager.default.removeItem(at: file) } }
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        do {
            var chunk = Data(); chunk.reserveCapacity(65_536)
            for try await byte in bytes {
                chunk.append(byte)
                if chunk.count == 65_536 { try handle.write(contentsOf: chunk); chunk.removeAll(keepingCapacity: true) }
            }
            if !chunk.isEmpty { try handle.write(contentsOf: chunk) }
            succeeded = true
            return (file, http)
        } catch { await bytes.cancel(); throw error }
    }
}

/// Explicitly selected only for the private embedded engine or injected contract fixtures.
struct LoopbackHTTPTransport: MoldHTTPTransport {
    let session: URLSession
    func response(for original: URLRequest, body: MoldRequestBody) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        var request = original
        let stream: MoldResponseBytes
        let response: URLResponse
        switch body {
        case let .data(data):
            request.httpBody = data.isEmpty ? nil : data
            let (bytes, answer) = try await session.bytes(for: request, delegate: RelayNoRedirect())
            stream = MoldResponseBytes(bytes); response = answer
        case let .file(file):
            let (data, answer) = try await session.upload(for: request, fromFile: file, delegate: RelayNoRedirect())
            stream = MoldResponseBytes(reader: FiniteBodyReader(data)); response = answer
        }
        guard let http = response as? HTTPURLResponse else { throw MoldClientError.malformedResponse }
        return (stream, http)
    }
}

private actor FiniteBodyReader: MoldBodyReader {
    var data: Data
    init(_ data: Data) { self.data = data }
    func read() -> Data? {
        guard !data.isEmpty else { return nil }
        let result = Data(data.prefix(65_536)); data.removeFirst(result.count); return result
    }
    func cancel() { data = Data() }
}
