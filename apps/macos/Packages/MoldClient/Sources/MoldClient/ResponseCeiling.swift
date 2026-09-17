import Foundation

/// How much of an answer this app is willing to hold.
///
/// Every route goes through `session.data(for:)`, which buffers the complete
/// body before anyone sees a byte of it, and nothing checks `Content-Length`
/// or caps the result. A compromised or simply broken host can therefore
/// answer a gallery listing or a print with an unbounded body and take the
/// process down -- and the person's only evidence is an app that quit.
///
/// The real fix is a streamed `download(for:)` on the media routes, writing
/// into the materializer's file instead of into memory. Until the transport
/// does that, the ceiling is applied where an unbounded body is actually
/// asked for, which turns an OOM into a refusal with a sentence.
public enum ResponseCeiling {
    /// What the server itself will serve for one member
    /// (`gallery_source_media.rs`'s own 512 MiB ceiling). Anything past that
    /// is not a print mold made.
    public static let media = 512 * 1_024 * 1_024

    /// A gallery index of ten thousand prints measures about 1.2 MB, and no
    /// JSON route in this app answers with more than one index.
    public static let json = 32 * 1_024 * 1_024

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
