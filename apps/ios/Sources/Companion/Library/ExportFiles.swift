import CryptoKit
import Foundation
import ImageIO
import MoldClient
import UniformTypeIdentifiers

/// Temporary output belongs to one operation; persistent copies belong to the user.
enum ExportFiles {
    static func stage(_ data: Data, filename: String, asset: GenerationAsset? = nil) throws -> URL {
        if let asset {
            guard data.count == asset.sizeBytes,
                  SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == asset.sha256.lowercased()
            else { throw MoldClientError.malformedResponse }
        }
        let directory = FileManager.default.temporaryDirectory.appending(path: "mold-print-export-\(UUID())")
        guard let named = SafeFilename.url(filename, in: directory), !data.isEmpty else {
            throw MoldClientError.malformedResponse
        }
        try verify(data, extension: named.pathExtension)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do { try data.write(to: named, options: .atomic) }
        catch { try? FileManager.default.removeItem(at: directory); throw error }
        return named
    }

    static func verify(_ data: Data, extension ext: String) throws {
        switch ext.lowercased() {
        case "gif", "png", "webp", "jpg", "jpeg":
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  CGImageSourceGetCount(source) > 0,
                  let type = CGImageSourceGetType(source),
                  let actual = UTType(type as String),
                  let expected = UTType(filenameExtension: ext), actual.conforms(to: expected),
                  CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else { throw MoldClientError.malformedResponse }
        case "glb": guard data.starts(with: Data("glTF".utf8)) else { throw MoldClientError.malformedResponse }
        case "zip": guard data.starts(with: [0x50, 0x4b, 0x03, 0x04]) else { throw MoldClientError.malformedResponse }
        case "ply": guard data.starts(with: Data("ply".utf8)) else { throw MoldClientError.malformedResponse }
        case "obj": guard String(data: data, encoding: .utf8)?.contains("\nv ") == true else { throw MoldClientError.malformedResponse }
        case "stl":
            let binaryCount = data.count >= 84 ? data[80..<84].enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << ($1.offset * 8) } : 0
            guard data.starts(with: Data("solid".utf8)) || (binaryCount > 0 && UInt64(data.count) == 84 + UInt64(binaryCount) * 50) else { throw MoldClientError.malformedResponse }
        default: break // Original audio/video is delivered without re-encoding.
        }
    }

    static func saveToFolder(_ url: URL, root: URL? = nil) throws -> URL {
        let base = try root ?? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let folder = base.appending(path: "Mold", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stem = url.deletingPathExtension().lastPathComponent
        var target = folder.appending(path: url.lastPathComponent)
        var suffix = 2
        while FileManager.default.fileExists(atPath: target.path) {
            target = folder.appending(path: "\(stem) (\(suffix)).\(url.pathExtension)"); suffix += 1
        }
        // Publish a complete copy with one rename. Both moves are exclusive;
        // a raced collision never overwrites another export.
        let staging = folder.appending(path: ".mold-export-\(UUID())")
        defer { try? FileManager.default.removeItem(at: staging) }
        try FileManager.default.copyItem(at: url, to: staging)
        try FileManager.default.moveItem(at: staging, to: target)
        return target
    }
}
