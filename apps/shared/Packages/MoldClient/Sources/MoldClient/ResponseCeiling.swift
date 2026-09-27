import Foundation

/// How much of an answer this app is willing to hold.
///
/// Two different guarantees live here, and they are not the same strength:
///
/// - **Bounded as it arrives** (`collected(upTo:)`), which is the real one: the
///   declared length is refused before a byte is read and the count is kept as
///   the bytes come in, so nothing large is ever allocated. The thumbnail
///   route reads this way -- it builds its own session, so it can.
/// - **Bounded on retention** (`checked(_:ceiling:what:)`), which is weaker and
///   says so: `HTTPBackend` buffers every route whole through
///   `session.data(for:)`, so by the time this runs the bytes are already in
///   memory. It stops this app KEEPING a hostile print, writing it to the
///   cache, and decoding ten of them into `NSImage` at once -- it does not stop
///   the allocation. Doing that means `download(for:)`/`bytes(for:)` in the
///   transport, which is `HTTPBackend+Transport.swift`'s to give.
public enum ResponseCeiling {
    /// What the server itself will serve for one member
    /// (`gallery_source_media.rs`'s own 512 MiB ceiling). Anything past that
    /// is not a print mold made.
    public static let media = 512 * 1_024 * 1_024

    /// A rendered thumbnail is tens of kilobytes at `?size=512`, and the
    /// answer is decoded into an `NSImage` on the main actor.
    public static let thumbnail = 32 * 1_024 * 1_024

    /// The body back, or a refusal naming the ceiling it broke.
    public static func checked(_ data: Data, ceiling: Int, what: String) throws -> Data {
        guard data.count <= ceiling else {
            throw Exceeded(bytes: data.count, ceiling: ceiling, what: what)
        }
        return data
    }

    public struct Exceeded: LocalizedError, Hashable, Sendable {
        public let bytes: Int
        public let ceiling: Int
        public let what: String

        public init(bytes: Int, ceiling: Int, what: String) {
            self.bytes = bytes
            self.ceiling = ceiling
            self.what = what
        }

        public var errorDescription: String? {
            let size = ByteCountFormatStyle().format(Int64(bytes))
            let cap = ByteCountFormatStyle().format(Int64(ceiling))
            return "It answered with \(size) of \(what), and Mold holds at most \(cap)."
        }
    }
}

public extension AsyncSequence where Element == UInt8 {
    /// The bytes, refused the moment they pass `ceiling`.
    ///
    /// Counted as it goes rather than measured afterwards, because a host that
    /// lies about `Content-Length` -- or sends none -- is precisely the one a
    /// ceiling exists for. The reserve is a guess at the answer's size and is
    /// itself bounded, so a declared 40 GB cannot be allocated here either.
    func collected(upTo ceiling: Int) async throws -> Data {
        var data = Data()
        data.reserveCapacity(Swift.min(ceiling, 1_024 * 1_024))
        for try await byte in self {
            guard data.count < ceiling else {
                throw ResponseCeiling.Exceeded(bytes: ceiling + 1, ceiling: ceiling,
                                               what: "that answer")
            }
            data.append(byte)
        }
        return data
    }
}
