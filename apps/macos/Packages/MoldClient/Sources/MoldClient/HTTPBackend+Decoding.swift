import Foundation

// The one place a reply becomes a value, and the one place a reply that
// cannot become one is written down.
extension HTTPBackend {
    /// Decodes a reply, or says WHERE it could not.
    ///
    /// `.malformedResponse` is still the right thing to hand a caller --
    /// there is nothing an app can do about a field this build cannot read,
    /// and retrying will not help. But collapsing to it threw away the only
    /// fact that makes such a bug findable: which key, at which path, on
    /// which route. That answer goes to the log; the BODY never does, and
    /// neither does the concrete route, whose components ARE filenames and
    /// ids (`RouteTemplate`).
    func decoded<T: Decodable>(_ type: T.Type, from data: Data, route: String) throws -> T {
        do {
            return try MoldJSON.decoder.decode(type, from: data)
        } catch let error as DecodingError {
            log(route, type, DecodingFailure.summary(error))
            throw MoldClientError.malformedResponse
        } catch {
            log(route, type, "unreadable")
            throw MoldClientError.malformedResponse
        }
    }

    private func log(_ route: String, _ type: Any.Type, _ failure: String) {
        MoldLog.decoding.error(
            """
            \(RouteTemplate.redacted(route), privacy: .public) as \
            \(String(describing: type), privacy: .public): \(failure, privacy: .public)
            """)
    }

    /// A refusal, as a route and a status. Never the server's sentence, which
    /// names filenames and models, and never a header.
    func note(_ error: Error, for request: URLRequest) {
        let route = RouteTemplate.redacted(request.url?.path(percentEncoded: true) ?? "")
        let method = request.httpMethod ?? "GET"
        switch error {
        case let MoldClientError.http(status, code, _):
            MoldLog.transport.error(
                """
                \(method, privacy: .public) \(route, privacy: .public) \
                refused \(status, privacy: .public) \(code ?? "-", privacy: .public)
                """)
        case MoldClientError.unauthorized:
            MoldLog.transport.error(
                "\(method, privacy: .public) \(route, privacy: .public) refused 401")
        case MoldClientError.licenseRequired:
            MoldLog.transport.notice(
                "\(method, privacy: .public) \(route, privacy: .public) needs a licence accepted")
        default:
            break
        }
    }
}
