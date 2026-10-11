import Foundation
@testable import MoldClient

extension HTTPBackend {
    /// Contract fixtures opt in explicitly; shipping remote hosts cannot bypass Noise.
    static func testing(host: MoldHost, session: URLSession = APISession.api) -> HTTPBackend {
        HTTPBackend(host: host, transport: LoopbackHTTPTransport(session: session))
    }
}
