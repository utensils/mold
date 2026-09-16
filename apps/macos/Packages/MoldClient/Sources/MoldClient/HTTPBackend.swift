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

    // MARK: - Transport

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let data = try await bytes(for: request(path))
        do {
            return try JSONDecoder().decode(T.self, from: data)
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
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError {
            throw MoldClientError.unreachable(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw MoldClientError.malformedResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw MoldClientError.unauthorized }
            let api = try? JSONDecoder().decode(APIError.self, from: data)
            throw MoldClientError.http(
                status: http.statusCode,
                code: api?.code,
                message: api?.error
            )
        }
        return data
    }
}

/// mold's error envelope. Every failing route answers with this shape, and the
/// `code` is the part to branch on -- the `error` prose is for humans and is
/// not stable.
struct APIError: Decodable, Sendable {
    let error: String
    let code: String
}
