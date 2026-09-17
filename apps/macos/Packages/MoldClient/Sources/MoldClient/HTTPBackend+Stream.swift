import Foundation

// Opening a `text/event-stream` route and turning it into frames. Split from
// the rest of the transport because a stream fails differently from a
// request: its body IS the answer, so a refusal has to be read before the
// body is handed to the parser, and only then.
extension HTTPBackend {
    /// A `text/event-stream` route as parsed frames. The status is checked
    /// before the first byte -- a 401 is `.unauthorized` like every other
    /// route, not a stream that opens and then goes silent.
    ///
    /// The one buffer in the pipeline: the parsers below it are lazy
    /// adapters that hold nothing, and this is where a consumer that cannot
    /// keep up stops costing memory (`StreamBuffering`).
    func stream(_ path: String, timeout: TimeInterval) -> AsyncThrowingStream<ServerSentEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingOldest(StreamBuffering.frames)) { continuation in
            let task = Task {
                do {
                    var request = self.request(path)
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    request.timeoutInterval = timeout

                    let (bytes, response) = try await session.bytes(
                        for: request, delegate: redirectGuard)
                    guard let http = response as? HTTPURLResponse
                    else { throw MoldClientError.malformedResponse }
                    if !(200..<300).contains(http.statusCode) {
                        throw await streamRefusal(http, bytes)
                    }

                    for try await frame in bytes.moldLines().serverSentEvents() {
                        continuation.yield(frame)
                    }
                    continuation.finish()
                } catch let error as URLError {
                    continuation.finish(throwing: Self.failure(for: error))
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
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
    private func streamRefusal(
        _ http: HTTPURLResponse, _ bytes: URLSession.AsyncBytes
    ) async -> Error {
        do {
            try check(http, await Self.refusalBody(bytes))
        } catch {
            return error
        }
        // `check` throws on every non-2xx; this is only the compiler's due.
        return MoldClientError.http(status: http.statusCode, code: nil, message: nil)
    }
}
