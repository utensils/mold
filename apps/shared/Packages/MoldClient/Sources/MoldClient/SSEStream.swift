import Foundation

/// One SSE route, opened on the first `next()` and read one frame at a time.
public struct SSEStream: AsyncSequence, Sendable {
    public typealias Element = ServerSentEvent

    let backend: HTTPBackend
    let path: String
    let timeout: TimeInterval

    public struct AsyncIterator: AsyncIteratorProtocol {
        let backend: HTTPBackend
        let path: String
        let timeout: TimeInterval
        var frames: ServerSentEventStream<MoldLines<URLSession.AsyncBytes>>.AsyncIterator?
        var done = false
        var relayStream = false

        public mutating func next() async throws -> ServerSentEvent? {
            guard !done else { return nil }
            do {
                try Task.checkCancellation()
                if frames == nil { frames = try await open() }
                guard let event = try await frames?.next() else {
                    if relayStream {
                        frames = nil
                        try await Task.sleep(for: .seconds(1))
                        return ServerSentEvent(name: "resync_required", data: "{}")
                    }
                    done = true
                    return nil
                }
                return event
            } catch let error as URLError {
                done = true
                throw TransportFailure.from(error)
            } catch {
                done = true
                throw error
            }
        }

        private mutating func open() async throws
            -> ServerSentEventStream<MoldLines<URLSession.AsyncBytes>>.AsyncIterator {
            var request = backend.request(path)
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            request.timeoutInterval = timeout

            let (bytes, http) = try await backend.relayBytes(request)
            relayStream = http.value(forHTTPHeaderField: "x-mold-relay-protocol") == "2"
            if !(200..<300).contains(http.statusCode) {
                throw await backend.streamRefusal(http, bytes)
            }
            return bytes.moldLines().serverSentEvents().makeAsyncIterator()
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(backend: backend, path: path, timeout: timeout)
    }
}
