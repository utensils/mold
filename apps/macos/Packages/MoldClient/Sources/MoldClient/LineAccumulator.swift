import Foundation

/// Splits a byte stream into lines, **including empty ones**.
///
/// `URLSession.AsyncBytes.lines` looks like exactly this and is not: it drops
/// zero-length lines. For server-sent events that is fatal and silent, because
/// the blank line is the frame terminator -- `event:` and `data:` arrive, the
/// parser buffers them, and nothing ever tells it the frame ended. The
/// connection is open, the server is sending, and the client reports nothing.
///
/// Byte-wise internally rather than by chunk so that a multi-byte character
/// split across two network reads still arrives whole: the bytes are
/// accumulated and decoded only once a newline closes the line. That is about
/// the guarantee, not about how a caller feeds it -- `consume(contentsOf:)`
/// takes whatever arrived.
public struct LineAccumulator {
    private var buffer: [UInt8] = []

    public init() {}

    /// Feeds one byte, appending the line it completed (if any) to `lines`.
    ///
    /// The `inout` form rather than a return value: on an SSE burst this runs
    /// once per byte, and an `[String]` allocated and thrown away per byte is
    /// the whole cost of the loop.
    public mutating func consume(_ byte: UInt8, into lines: inout [String]) {
        switch byte {
        case 0x0A:  // \n
            lines.append(String(decoding: buffer, as: UTF8.self))
            buffer.removeAll(keepingCapacity: true)
        case 0x0D:  // \r, framing in \r\n; a bare one is not a terminator here
            break
        default:
            buffer.append(byte)
        }
    }

    /// Feeds however many bytes arrived together, appending every line they
    /// completed to `lines`. A line split across two chunks -- or a character
    /// split across two chunks -- is still whole when it comes out.
    public mutating func consume(
        contentsOf chunk: some Sequence<UInt8>, into lines: inout [String]
    ) {
        for byte in chunk { consume(byte, into: &lines) }
    }

    /// Feeds one byte. Returns the line it completed, if it completed one.
    ///
    /// Returns an array rather than an optional so a caller can write
    /// `out += accumulator.consume(byte)` without a branch; it is never longer
    /// than one.
    public mutating func consume(_ byte: UInt8) -> [String] {
        var lines: [String] = []
        consume(byte, into: &lines)
        return lines
    }
}
