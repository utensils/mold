import Foundation

/// One `text/event-stream` message.
public struct ServerSentEvent: Hashable, Sendable {
    public let name: String?
    public let data: String
}

/// Turns a byte stream into events.
///
/// Written as a parser over lines rather than inline in the network call so it
/// can be tested against the exact framing mold emits -- including the
/// keep-alive comments it sends every 15 seconds, which are not events and
/// must not be delivered as empty ones.
public struct SSEParser: Sendable {
    private var name: String?
    private var data: [String] = []

    public init() {}

    /// Feeds one line. Returns an event when the line completes one.
    public mutating func consume(line: String) -> ServerSentEvent? {
        // A comment. The server uses these as keep-alives.
        if line.hasPrefix(":") { return nil }

        if line.isEmpty {
            defer { name = nil; data = [] }
            // A blank line with nothing buffered is just framing.
            guard !data.isEmpty else { return nil }
            return ServerSentEvent(name: name, data: data.joined(separator: "\n"))
        }

        guard let colon = line.firstIndex(of: ":") else {
            return nil  // A field with no value carries nothing we use.
        }
        let field = String(line[..<colon])
        var value = String(line[line.index(after: colon)...])
        // Exactly one leading space is part of the framing, not the value.
        if value.hasPrefix(" ") { value.removeFirst() }

        switch field {
        case "event": name = value
        case "data": data.append(value)
        default: break  // `id` and `retry` are not used here.
        }
        return nil
    }
}

/// A line sequence parsed into events.
///
/// Lazy for the same reason `MoldLines` is: a `Task` pumping an
/// `AsyncThrowingStream` here would be a second unbounded buffer between the
/// socket and a consumer that cannot keep up, and there is no `yield` that
/// blocks. Pulling, the whole way down, means one line is read for one
/// `next()`.
public struct ServerSentEventStream<Base: AsyncSequence & Sendable>: AsyncSequence, Sendable
where Base.Element == String {
    public typealias Element = ServerSentEvent

    let base: Base

    public struct AsyncIterator: AsyncIteratorProtocol {
        var base: Base.AsyncIterator
        var parser = SSEParser()

        public mutating func next() async throws -> ServerSentEvent? {
            while let line = try await base.next() {
                if let event = parser.consume(line: line) { return event }
            }
            return nil
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(base: base.makeAsyncIterator())
    }
}

public extension AsyncSequence where Element == String, Self: Sendable {
    /// Parses a line sequence into events.
    func serverSentEvents() -> ServerSentEventStream<Self> { ServerSentEventStream(base: self) }
}
