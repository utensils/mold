import Foundation

/// The recipe in the bytes is the immutable authority for a mirrored print.
/// The gallery listing can have a newer DB recipe for the same old PNG/JPEG.
public enum EmbeddedPrintMetadata {
    public static func json(in url: URL, named filename: String) throws -> Data? {
        let suffix = URL(fileURLWithPath: filename).pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "gif"].contains(suffix) else { return nil }
        // Mapping avoids copying a full-size picture just to inspect its
        // small metadata chunk before a file-backed upload.
        return json(in: try Data(contentsOf: url, options: .mappedIfSafe), named: filename)
    }

    public static func json(in file: Data, named filename: String) -> Data? {
        switch URL(fileURLWithPath: filename).pathExtension.lowercased() {
        case "png": return png(file)
        case "jpg", "jpeg": return jpeg(file)
        case "gif": return gif(file)
        default: return nil
        }
    }

    private static func gif(_ file: Data) -> Data? {
        guard file.starts(with: Data("GIF8".utf8)) else { return nil }
        // GIF application metadata uses a comment extension: 21 FE, followed
        // by one or more length-prefixed sub-blocks and a zero terminator.
        let prefix = Data("mold:parameters ".utf8)
        var cursor = 6
        while cursor + 2 < file.count {
            guard file[cursor] == 0x21, file[cursor + 1] == 0xFE else {
                cursor += 1
                continue
            }
            cursor += 2
            var comment = Data()
            while cursor < file.count {
                let size = Int(file[cursor]); cursor += 1
                if size == 0 { break }
                guard size <= file.count - cursor else { return nil }
                comment.append(file[cursor..<(cursor + size)])
                cursor += size
            }
            if comment.starts(with: prefix),
               let json = validJSON(comment.dropFirst(prefix.count)) { return json }
        }
        return nil
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
