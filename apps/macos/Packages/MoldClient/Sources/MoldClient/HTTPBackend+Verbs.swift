import Foundation

// The verbs beyond GET and POST. Collections, tags, devices and settings all
// need PATCH, PUT and DELETE, and each of them wants the same three lines.
extension HTTPBackend {
    func request(_ path: String, method: String) -> URLRequest {
        var request = self.request(path)
        request.httpMethod = method
        return request
    }

    func body<Body: Encodable>(_ path: String, method: String, _ body: Body) throws -> URLRequest {
        var request = self.request(path, method: method)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try MoldJSON.encoder.encode(body)
        return request
    }

    /// A request with a body whose answer matters.
    func send<Body: Encodable, T: Decodable>(
        _ path: String, method: String, body: Body
    ) async throws -> T {
        let data = try await bytes(for: self.body(path, method: method, body))
        do {
            return try MoldJSON.decoder.decode(T.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// A request with a body whose answer does not.
    func send<Body: Encodable>(_ path: String, method: String, body: Body) async throws {
        _ = try await bytes(for: self.body(path, method: method, body))
    }

    func delete(_ path: String) async throws {
        _ = try await bytes(for: request(path, method: "DELETE"))
    }

    /// A path component that may contain anything a person typed -- a tag can
    /// hold a slash, a space or a `#`.
    func escaped(_ component: String) -> String {
        component.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? component
    }
}
