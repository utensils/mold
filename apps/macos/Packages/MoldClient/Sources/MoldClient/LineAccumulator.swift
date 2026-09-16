import Foundation

/// Splits a byte stream into lines, **including empty ones**.
///
/// `URLSession.AsyncBytes.lines` looks like exactly this and is not: it drops
/// zero-length lines. For server-sent events that is fatal and silent, because
/// the blank line is the frame terminator -- `event:` and `data:` arrive, the
/// parser buffers them, and nothing ever tells it the frame ended. The
/// connection is open, the server is sending, and the client reports nothing.
///
/// Byte-at-a-time rather than by chunk so that a multi-byte character split
/// across two network reads still arrives whole: the bytes are accumulated and
/// decoded only once a newline closes the line.
public struct LineAccumulator {
    private var buffer: [UInt8] = []

    public init() {}

    /// Feeds one byte. Returns the line it completed, if it completed one.
    ///
    /// Returns an array rather than an optional so a caller can write
    /// `out += accumulator.consume(byte)` without a branch; it is never longer
    /// than one.
    public mutating func consume(_ byte: UInt8) -> [String] {
        switch byte {
        case 0x0A:  // \n
            let line = String(decoding: buffer, as: UTF8.self)
            buffer.removeAll(keepingCapacity: true)
            return [line]
        case 0x0D:  // \r, framing in \r\n; a bare one is not a terminator here
            return []
        default:
            buffer.append(byte)
            return []
        }
    }
}

public extension AsyncSequence where Element == UInt8, Self: Sendable {
    /// The stream's lines, empty ones included. See `LineAccumulator`.
    func moldLines() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var accumulator = LineAccumulator()
                do {
                    for try await byte in self {
                        for line in accumulator.consume(byte) { continuation.yield(line) }
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
