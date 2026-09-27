import Foundation

/// A byte stream's lines, empty ones included. See `LineAccumulator`.
///
/// A LAZY adapter rather than a `Task` pumping an `AsyncThrowingStream`, and
/// that is the point of it. `AsyncStream` has no `yield` that blocks, so a
/// pumped stream reads the socket as fast as the socket will go regardless of
/// what the consumer can take: during a gallery burst -- a bulk import, an
/// `emptyTrash`, one `gallery_*` frame per print -- memory tracked the BURST
/// rather than the app, and the consumer (`@MainActor`, fanning out
/// synchronously) then spent seconds applying events that were already stale.
/// Pulling instead makes the consumer's pace the read pace, with the socket's
/// own window as the only buffer, and costs no cross-task hop per byte.
public struct MoldLines<Base: AsyncSequence & Sendable>: AsyncSequence, Sendable
where Base.Element == UInt8 {
    public typealias Element = String

    let base: Base

    public struct AsyncIterator: AsyncIteratorProtocol {
        var base: Base.AsyncIterator
        var accumulator = LineAccumulator()
        /// Lines one read completed but the consumer has not taken yet.
        /// Bounded by that read, not by the stream.
        var ready: [String] = []
        var next = 0

        public mutating func next() async throws -> String? {
            while true {
                if next < ready.count {
                    defer { next += 1 }
                    return ready[next]
                }
                ready.removeAll(keepingCapacity: true)
                next = 0
                guard let byte = try await base.next() else { return nil }
                accumulator.consume(byte, into: &ready)
            }
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(base: base.makeAsyncIterator())
    }
}

public extension AsyncSequence where Element == UInt8, Self: Sendable {
    /// The stream's lines, empty ones included. See `LineAccumulator`.
    func moldLines() -> MoldLines<Self> { MoldLines(base: self) }
}
