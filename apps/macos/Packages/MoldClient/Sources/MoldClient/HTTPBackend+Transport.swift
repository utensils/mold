import Foundation

// Building a request and reading an answer. Split from the routes for
// size; `internal` throughout because every `HTTPBackend+` file shares it.
extension HTTPBackend {
    func get<T: Decodable>(_ path: String) async throws -> T {
        let data = try await bytes(for: request(path))
        do {
            return try MoldJSON.decoder.decode(T.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// `timeout` is the ordinary 10 s unless a route says otherwise -- a
    /// prompt rewrite may have to load its LLM first, and a 10 s idle limit
    /// turned every cold expansion into "the request timed out".
    func post<Body: Encodable, T: Decodable>(
        _ path: String, body: Body, timeout: TimeInterval = 10
    ) async throws -> T {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        let data = try await bytes(for: request)
        do {
            return try MoldJSON.decoder.decode(T.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// A request for a path, which may carry a query string.
    ///
    /// The path is taken as ALREADY PERCENT-ENCODED — interpolate a dynamic
    /// component through `escaped(_:)`. NOT `baseURL.appending(path:)`, which
    /// treats its whole argument as one component and encodes everything in
    /// it, `?` included: `/api/gallery?view=trash` became
    /// `/api/gallery%3Fview=trash`, a route no mold has, and Recently Deleted
    /// listed nothing on a machine holding 177 prints — silently, because the
    /// app asks conditionally and a shrug looks exactly like "nothing
    /// changed".
    func request(_ path: String) -> URLRequest {
        let parts = path.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        var components = URLComponents(url: host.baseURL, resolvingAgainstBaseURL: false)
        // A host behind a reverse proxy keeps the prefix in its base URL.
        var prefix = components?.percentEncodedPath ?? ""
        if prefix.hasSuffix("/") { prefix.removeLast() }
        components?.percentEncodedPath = prefix + String(parts[0])
        if parts.count == 2 { components?.percentEncodedQuery = String(parts[1]) }
        guard let url = components?.url else {
            return URLRequest(url: host.baseURL.appending(path: path))
        }
        var request = URLRequest(url: url)
        // A keyless host is open by policy. Sending no key is the correct
        // request there, not a degraded one.
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        request.timeoutInterval = 10
        return request
    }

    func bytes(for request: URLRequest) async throws -> Data {
        let (data, http) = try await send(request)
        try check(http, data)
        return data
    }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw MoldClientError.malformedResponse
            }
            return (data, http)
        } catch let error as URLError {
            throw Self.failure(for: error)
        }
    }

    /// A cancelled request is the app changing its mind -- a `.task(id:)`
    /// re-keying, a view going away -- not the machine failing, so it must
    /// never present as `.unreachable`. Every other `URLError` still becomes
    /// the same reachability failure as before.
    static func failure(for error: URLError) -> Error {
        error.code == .cancelled ? CancellationError() : MoldClientError.unreachable(error.localizedDescription)
    }

    func check(_ http: HTTPURLResponse, _ data: Data) throws {
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw MoldClientError.unauthorized }
            let api = try? MoldJSON.decoder.decode(APIError.self, from: data)
            if let refusal = api?.license,
               api?.code == LicenseCode.notAccepted || api?.code == LicenseCode.termsMismatch {
                throw MoldClientError.licenseRequired(refusal, mismatch: api?.code == LicenseCode.termsMismatch)
            }
            throw MoldClientError.http(
                status: http.statusCode,
                code: api?.code,
                message: api?.error ?? Self.plainMessage(data)
            )
        }
    }

}
