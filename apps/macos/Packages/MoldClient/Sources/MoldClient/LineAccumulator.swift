import Foundation

/// Splits a byte stream into lines, **including empty ones**.
///
/// `URLSession.AsyncBytes.lines` looks like exactly this and is not: it drops
/// zero-length lines. For server-sent events that is fatal and silent, because
/// the blank line is the frame terminator -- `event:` and `data:` arrive, the
/// parser buffers them, and nothing ever tells it the frame ended. The
/// connection is open, the server is sending, and the client reports nothing.
///
/// Byte-wise so that a multi-byte character split across two network reads
/// still arrives whole: the bytes are accumulated and decoded only once a
/// newline closes the line.
public struct LineAccumulator {
    private var buffer: [UInt8] = []

    public init() {}

    /// Feeds one byte, appending the line it completed (if any) to `lines`.
    ///
    /// The `inout` form rather than a return value, and one method rather
    /// than several: on an SSE burst this runs once per byte, and an
    /// `[String]` allocated and thrown away per byte to say "no line yet" was
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
}
