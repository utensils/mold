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

public extension AsyncSequence where Element == String, Self: Sendable {
    /// Parses a line sequence into events.
    func serverSentEvents() -> AsyncThrowingStream<ServerSentEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var parser = SSEParser()
                do {
                    for try await line in self {
                        if let event = parser.consume(line: line) {
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
}
