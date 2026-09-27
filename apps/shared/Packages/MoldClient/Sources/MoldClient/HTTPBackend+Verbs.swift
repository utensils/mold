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
        return try decoded(T.self, from: data, route: path)
    }

    /// A request with a body whose answer does not.
    func send<Body: Encodable>(_ path: String, method: String, body: Body) async throws {
        _ = try await bytes(for: self.body(path, method: method, body))
    }

    func delete(_ path: String) async throws {
        _ = try await bytes(for: request(path, method: "DELETE"))
    }

    /// One path component, whatever a person put in it. `RouteEscaping` owns
    /// the rule, and its query-value sibling, which is a DIFFERENT one.
    func escaped(_ component: String) -> String { RouteEscaping.escaped(component) }
}
