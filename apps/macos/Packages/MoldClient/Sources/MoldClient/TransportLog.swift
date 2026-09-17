import Foundation

/// What a refusal or an unreadable reply writes down.
///
/// Everything here goes through `RouteTemplate.redacted`, because a mold
/// route is made of the things a person owns -- a filename, a tag they typed,
/// a job id -- and `MoldLog`'s rule is that none of those is ever logged. The
/// body is never written at all, and neither is a header.
enum TransportLog {
    /// A reply this build could not read, as the route and the TYPE it was
    /// asked for. Together those name the bug without naming anything in it.
    static func decodeFailure(route: String, type: Any.Type, failure: String) {
        MoldLog.decoding.error(
            """
            \(RouteTemplate.redacted(route), privacy: .public) as \
            \(String(describing: type), privacy: .public): \(failure, privacy: .public)
            """)
    }

    /// A refusal, as a route and a status. Never the server's sentence, which
    /// names filenames and models, and never a header.
    static func refusal(_ error: Error, for request: URLRequest) {
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
