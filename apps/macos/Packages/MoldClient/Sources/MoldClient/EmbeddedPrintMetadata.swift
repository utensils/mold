import Foundation

/// The recipe in the bytes is the immutable authority for a mirrored print.
/// The gallery listing can have a newer DB recipe for the same old PNG/JPEG.
public enum EmbeddedPrintMetadata {
    public static func json(in file: Data, named filename: String) -> Data? {
        switch URL(fileURLWithPath: filename).pathExtension.lowercased() {
        case "png": return png(file)
        case "jpg", "jpeg": return jpeg(file)
        default: return nil
        }
    }

    private static func validJSON(_ data: Data) -> Data? {
        (try? JSONSerialization.jsonObject(with: data)) is [String: Any] ? data : nil
    }

    private static func png(_ file: Data) -> Data? {
        file.withUnsafeBytes { raw in
        let bytes = raw.bindMemory(to: UInt8.self)
        guard bytes.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]) else { return nil }
        var cursor = 8
        while cursor + 12 <= bytes.count {
            let length = Int(bytes[cursor]) << 24 | Int(bytes[cursor + 1]) << 16
                | Int(bytes[cursor + 2]) << 8 | Int(bytes[cursor + 3])
            guard length >= 0, length <= bytes.count - cursor - 12 else { return nil }
            let kind = String(bytes: bytes[(cursor + 4)..<(cursor + 8)], encoding: .ascii)
            if kind == "tEXt" || kind == "iTXt" {
                let payload = Array(bytes[(cursor + 8)..<(cursor + 8 + length)])
                if let separator = payload.firstIndex(of: 0),
                   String(bytes: payload[..<separator], encoding: .ascii) == "mold:parameters" {
                    if kind == "tEXt" {
                        if let json = validJSON(Data(payload.dropFirst(separator + 1))) { return json }
                        cursor += length + 12
                        continue
                    }
                    // iTXt: keyword\0, compression flag, compression method,
                    // language\0, translated keyword\0, UTF-8 text.
                    let start = separator + 3
                    guard start <= payload.count, payload[separator + 1] == 0,
                          let languageEnd = payload[start...].firstIndex(of: 0),
                          let translatedEnd = payload[(languageEnd + 1)...].firstIndex(of: 0)
                    else { return nil }
                    if let json = validJSON(Data(payload.dropFirst(translatedEnd + 1))) { return json }
                }
            }
            cursor += length + 12
        }
        return nil
        }
    }

    private static func jpeg(_ file: Data) -> Data? {
        file.withUnsafeBytes { raw in
        let bytes = raw.bindMemory(to: UInt8.self)
        guard bytes.starts(with: [0xFF, 0xD8]) else { return nil }
        var cursor = 2
        while cursor + 4 <= bytes.count, bytes[cursor] == 0xFF {
            let marker = bytes[cursor + 1]
            if marker == 0xD9 || marker == 0xDA { break }
            if marker == 0xD8 || marker == 0x01 || (0xD0...0xD7).contains(marker) {
                cursor += 2
                continue
            }
            let length = Int(bytes[cursor + 2]) << 8 | Int(bytes[cursor + 3])
            guard length >= 2, length <= bytes.count - cursor - 2 else { return nil }
            if marker == 0xFE {
                let comment = Data(bytes[(cursor + 4)..<(cursor + 2 + length)])
                let prefix = Data("mold:parameters ".utf8)
                if comment.starts(with: prefix) {
                    if let json = validJSON(comment.dropFirst(prefix.count)) { return json }
                }
            }
            cursor += 2 + length
        }
        return nil
        }
    }
}
