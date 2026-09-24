import Foundation

// Building a request and reading an answer. Split from the routes for
// size; `internal` throughout because every `HTTPBackend+` file shares it.
extension HTTPBackend {
    func get<T: Decodable>(_ path: String) async throws -> T {
        let data = try await bytes(for: request(path))
        return try decoded(T.self, from: data, route: path)
    }

    /// `timeout` is the ordinary 10 s unless a route says otherwise -- a
    /// prompt rewrite may have to load its LLM first, and a 10 s idle limit
    /// turned every cold expansion into "the request timed out".
    func post<Body: Encodable, T: Decodable>(
        _ path: String, body: Body, timeout: TimeInterval = 10,
        headers: [String: String] = [:]
    ) async throws -> T {
        var request = self.request(path)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // Never logged: `TransportLog` reports the route and the refusal, and
        // one of these is a one-use credential.
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        request.httpBody = try MoldJSON.encoder.encode(body)
        let data = try await bytes(for: request)
        return try decoded(T.self, from: data, route: path)
    }

    /// A request for a path, which may carry a query string. The path is taken
    /// as ALREADY PERCENT-ENCODED -- see `RouteRequest`, which owns the rule.
    func request(_ path: String) -> URLRequest { RouteRequest.build(path, for: host) }

    func bytes(for request: URLRequest) async throws -> Data {
        let (data, http) = try await send(request)
        do {
            try HTTPRefusal.check(http, data)
        } catch {
            TransportLog.refusal(error, for: request)
            throw error
        }
        return data
    }

    /// The task delegate every request carries.
    ///
    /// Its only job is to take `X-Api-Key` off a redirect that leaves this
    /// host's origin (`RedirectGuard`). Per TASK, so a caller's own session --
    /// a stub in tests, a configured one in the app -- keeps whatever
    /// delegate it already has.
    var redirectGuard: RedirectGuard { RedirectGuard(origin: host.baseURL) }

    func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (data, response) = try await session.data(
                for: request, delegate: redirectGuard)
            guard let http = response as? HTTPURLResponse else {
                throw MoldClientError.malformedResponse
            }
            return (data, http)
        } catch let error as URLError {
            throw TransportFailure.from(error)
        }
    }

    func upload(_ request: URLRequest, fromFile file: URL) async throws -> Data {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.upload(
                for: request, fromFile: file, delegate: redirectGuard)
        } catch let error as URLError {
            throw TransportFailure.from(error)
        }
        guard let http = response as? HTTPURLResponse else {
            throw MoldClientError.malformedResponse
        }
        do {
            try HTTPRefusal.check(http, data)
        } catch {
            TransportLog.refusal(error, for: request)
            throw error
        }
        return data
    }
}
