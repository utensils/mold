import Foundation

/// Talks to a `mold serve` over HTTP.
///
/// This is the only transport the app has, and it is the same one an embedded
/// engine will use -- see `MoldBackend`.
public struct HTTPBackend: MoldBackend {
    public let host: MoldHost
    private let session: URLSession

    public init(host: MoldHost, session: URLSession = .shared) {
        self.host = host
        self.session = session
    }

    public func status() async throws -> ServerStatus {
        try await get("/api/status")
    }

    public func capabilities() async throws -> Capabilities {
        try await get("/api/capabilities")
    }

    public func models() async throws -> [Model] {
        try await get("/api/models")
    }

    public func queue() async throws -> QueueListing {
        try await get("/api/queue")
    }

    public func placementPreview(
        _ request: GenerateRequest, copies: Int = 1
    ) async throws -> PlacementPreview {
        try await post("/api/generate/placement-preview",
                       body: PlacementRequest(request: request, copies: copies))
    }

    public func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        var request = self.request("/api/gallery")
        // The index is large and mostly unchanged between refreshes, so ask
        // the host whether it changed at all before it serializes 1.2 MB.
        if let etag { request.setValue(etag, forHTTPHeaderField: "If-None-Match") }
        // Listing a full gallery takes longer than a status probe.
        request.timeoutInterval = 60

        let (data, http) = try await send(request)
        if http.statusCode == 304 { return .notModified }
        try check(http, data)
        do {
            let prints = try MoldJSON.decoder.decode([GalleryPrint].self, from: data)
            return .fresh(prints, etag: http.value(forHTTPHeaderField: "ETag"))
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    // MARK: - Transport

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let data = try await bytes(for: request(path))
        do {
            return try MoldJSON.decoder.decode(T.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    private func post<Body: Encodable, T: Decodable>(_ path: String, body: Body) async throws -> T {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        let data = try await bytes(for: request)
        do {
            return try MoldJSON.decoder.decode(T.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    private func request(_ path: String) -> URLRequest {
        var request = URLRequest(url: host.baseURL.appending(path: path))
        // A keyless host is open by policy. Sending no key is the correct
        // request there, not a degraded one.
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        request.timeoutInterval = 10
        return request
    }

    private func bytes(for request: URLRequest) async throws -> Data {
        let (data, http) = try await send(request)
        try check(http, data)
        return data
    }

    private func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw MoldClientError.malformedResponse
            }
            return (data, http)
        } catch let error as URLError {
            throw MoldClientError.unreachable(error.localizedDescription)
        }
    }

    private func check(_ http: HTTPURLResponse, _ data: Data) throws {
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw MoldClientError.unauthorized }
            let api = try? MoldJSON.decoder.decode(APIError.self, from: data)
            throw MoldClientError.http(
                status: http.statusCode,
                code: api?.code,
                message: api?.error
            )
        }
    }
}

/// mold's error envelope. Every failing route answers with this shape, and the
/// `code` is the part to branch on -- the `error` prose is for humans and is
/// not stable.
struct APIError: Decodable, Sendable {
    let error: String
    let code: String
}
