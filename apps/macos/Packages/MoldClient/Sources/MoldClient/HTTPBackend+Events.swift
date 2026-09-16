import Foundation

// The host-wide event stream. One connection carries everything a machine
// does, which is why this exists at all: the alternative is a poll per pane.
extension HTTPBackend {

    /// Everything this machine reports, as it happens.
    ///
    /// Yields only what this app understands. Job lifecycle, chain jobs and
    /// whatever mold adds next come down the same stream, and `MoldEvent`
    /// answers nil for all of them -- which is the contract, not a gap: a
    /// client that treats an unrecognised tag as a failure breaks the first
    /// time a machine is upgraded ahead of it.
    public func events() -> AsyncThrowingStream<MoldEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = self.request("/api/events")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    // An idle machine says nothing for hours, and that is the
                    // stream working rather than the stream stuck.
                    request.timeoutInterval = 86_400

                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse,
                          (200..<300).contains(http.statusCode)
                    else { throw eventStreamFailure(response) }

                    var parser = SSEParser()
                    for try await line in bytes.moldLines() {
                        guard let frame = parser.consume(line: line) else { continue }
                        if let event = MoldEvent(name: frame.name, data: frame.data) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func eventStreamFailure(_ response: URLResponse) -> MoldClientError {
        guard let http = response as? HTTPURLResponse else { return .malformedResponse }
        if http.statusCode == 401 { return .unauthorized }
        return .http(status: http.statusCode, code: nil, message: nil)
    }
}
