import Foundation

/// The `.glb` container: a 12-byte header, then length-prefixed chunks padded
/// to four bytes.
///
/// Port of `splitContainer` in `studio/lib/glb.ts:164-227`. Every offset is
/// checked against the buffer BEFORE anything is read out of it, and the walk
/// cannot stall: `start` always grows, so a zero-length chunk still advances.
enum GLBContainer {
    /// "glTF", read as one little-endian u32.
    static let magic: UInt32 = 0x4654_6C67
    static let jsonChunkType: UInt32 = 0x4E4F_534A
    static let binChunkType: UInt32 = 0x004E_4942

    struct Split {
        let json: [String: Any]
        let bin: [UInt8]
    }

    static func split(_ buffer: [UInt8]) throws -> Split {
        guard buffer.count >= 12 else {
            throw GLBParseError(
                "not a GLB: a 12-byte header needs 12 bytes, got \(buffer.count)")
        }
        let magic = try readUInt32(buffer, at: 0)
        guard magic == Self.magic else {
            throw GLBParseError(
                "not a GLB: bad magic 0x\(hex(magic)), expected \"glTF\"")
        }
        let version = try readUInt32(buffer, at: 4)
        guard version == 2 else {
            throw GLBParseError(
                "unsupported GLB version \(version): only glTF 2 binary is supported")
        }
        let declared = Int(try readUInt32(buffer, at: 8))
        guard declared == buffer.count else {
            throw GLBParseError(
                "GLB length mismatch: the header declares \(declared) bytes but the "
                    + "buffer holds \(buffer.count)")
        }

        var offset = 12
        var jsonChunk: ArraySlice<UInt8>?
        var binChunk: ArraySlice<UInt8>?
        while offset + 8 <= declared {
            let length = Int(try readUInt32(buffer, at: offset))
            let type = try readUInt32(buffer, at: offset + 4)
            let start = offset + 8
            guard length <= declared - start else {
                throw GLBParseError(
                    "truncated GLB \(name(of: type)) chunk: it declares \(length) bytes "
                        + "but only \(declared - start) remain")
            }
            if type == jsonChunkType, jsonChunk == nil {
                jsonChunk = buffer[start..<(start + length)]
            } else if type == binChunkType, binChunk == nil {
                binChunk = buffer[start..<(start + length)]
            }
            offset = start + length + ((4 - (length % 4)) % 4)
        }

        guard let jsonChunk else { throw GLBParseError("GLB has no JSON chunk") }
        return Split(json: try parseJSON(jsonChunk), bin: Array(binChunk ?? []))
    }

    /// The JSON chunk is padded with spaces to a four-byte boundary, which
    /// `JSON.parse` skips and `JSONSerialization` will not, so the padding is
    /// trimmed rather than handed over.
    private static func parseJSON(_ chunk: ArraySlice<UInt8>) throws -> [String: Any] {
        var bytes = chunk
        while let last = bytes.last, last == 0x20 || last == 0x00 || last == 0x0A {
            bytes = bytes.dropLast()
        }
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: Data(bytes), options: [])
        } catch {
            throw GLBParseError(
                "GLB JSON chunk is not valid JSON: \(error.localizedDescription)")
        }
        guard let document = parsed as? [String: Any] else {
            throw GLBParseError("GLB JSON chunk is not a glTF object")
        }
        return document
    }

    private static func name(of type: UInt32) -> String {
        if type == jsonChunkType { return "JSON" }
        if type == binChunkType { return "BIN" }
        return "0x\(hex(type))"
    }

    private static func hex(_ value: UInt32) -> String {
        String(format: "%08x", value)
    }
}

/// One little-endian `u32`, refused rather than read past the end.
///
/// Bounds-checked slicing throughout: this reader is handed hostile bytes by
/// design, so there is no unchecked pointer arithmetic anywhere in it.
func readUInt32(_ bytes: [UInt8], at offset: Int) throws -> UInt32 {
    guard offset >= 0, offset + 4 <= bytes.count else {
        throw GLBParseError("GLB read of 4 bytes at \(offset) is past the end of the file")
    }
    return UInt32(bytes[offset])
        | UInt32(bytes[offset + 1]) << 8
        | UInt32(bytes[offset + 2]) << 16
        | UInt32(bytes[offset + 3]) << 24
}
