import Foundation

// Opening a `text/event-stream` route and turning it into frames.
extension HTTPBackend {
    /// A `text/event-stream` route as parsed frames. The status is checked
    /// before the first byte -- a 401 is `.unauthorized` like every other
    /// route, not a stream that opens and then goes silent.
    ///
    /// LAZY, like the two parsers under it, so that the whole path from the
    /// socket to a route's own stream holds nothing: there is exactly ONE
    /// buffer in this pipeline and it is the one the route declares
    /// (`StreamBuffering`). A `Task`-pumped stream here would have been a
    /// second one, stacked under it, with its own independent capacity.
    func stream(_ path: String, timeout: TimeInterval) -> SSEStream {
        SSEStream(backend: self, path: path, timeout: timeout)
    }

    /// The same refusal a plain GET would report, read off a stream's body.
    ///
    /// A refusal on an event route used to be built from the status code
    /// alone: `503 SERVER_RESTARTING` -- which names what is happening and is
    /// a wait rather than a fault -- became "The machine answered with an
    /// error (503)", and the reconnect loop then retried it forever with
    /// nothing to say. A licence refusal, which is the one refusal the app can
    /// RESOLVE, could not happen at all on a stream, because its payload rides
    /// in the body that was being dropped.
    func streamRefusal(_ http: HTTPURLResponse, _ bytes: URLSession.AsyncBytes) async -> Error {
        do {
            try HTTPRefusal.check(http, await RefusalBody.read(bytes))
        } catch {
            return error
        }
        // `check` throws on every non-2xx; this is only the compiler's due.
        return MoldClientError.http(status: http.statusCode, code: nil, message: nil)
    }
}
