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
        var frames: ServerSentEventStream<MoldLines<MoldResponseBytes>>.AsyncIterator?
        var done = false

        public mutating func next() async throws -> ServerSentEvent? {
            guard !done else { return nil }
            do {
                try Task.checkCancellation()
                if frames == nil { frames = try await open() }
                guard let event = try await frames?.next() else {
                    done = true
                    return nil
                }
                return event
            } catch {
                done = true
                throw TransportFailure.from(any: error)
            }
        }

        private mutating func open() async throws
            -> ServerSentEventStream<MoldLines<MoldResponseBytes>>.AsyncIterator {
            var request = backend.request(path)
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            request.timeoutInterval = timeout

            let (bytes, http) = try await backend.relayBytes(request)
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
