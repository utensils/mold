import Foundation
import ImageIO
import UniformTypeIdentifiers

/// A picture a well is holding, already in a form the host can read and
/// already base64 -- because mold takes every byte field as base64 on the
/// wire, so the draft holds exactly what will be sent.
public struct ImportedPicture: Sendable {
    public let encoded: String
    public let name: String
    public let data: Data

    public init(encoded: String, name: String, data: Data) {
        self.encoded = encoded
        self.name = name
        self.data = data
    }
}

/// Reading a picture in, off the main actor and in a format the engine reads.
///
/// Two findings meet here. `Data(contentsOf:)` + `base64EncodedString()` +
/// `NSImage(data:)` for a 50 MB photograph all ran on the main thread from a
/// `View` method (02#10) -- the app defaults to MainActor isolation, so the
/// only way off it is an explicit `Task.detached`. And the open panels offer
/// HEIC, the default format of every photograph an iPhone syncs to a Mac,
/// which no decoder behind mold reads: the identity path walks a PNG
/// signature and then JPEG markers (`crates/mold-core/src/identity.rs:831-880`)
/// and the rest goes through the `image` crate, which has no HEIC at all. The
/// 422 arrived after the whole request had been uploaded (02#7).
/// `nonisolated` throughout: the whole point is to run off the main actor,
/// and the app's `SWIFT_DEFAULT_ACTOR_ISOLATION: MainActor` would otherwise
/// pull every one of these back onto it.
public nonisolated enum PictureImport {
    /// What mold's general picture inputs decode (the `image` crate).
    public static let engineReadable: Set<String> = [
        UTType.png.identifier, UTType.jpeg.identifier, UTType.webP.identifier,
        UTType.tiff.identifier, UTType.gif.identifier, UTType.bmp.identifier,
    ]

    /// The type a wire image-format name (`ImageInputFormat`) names, or nil
    /// for one this build does not know.
    public static func typeIdentifier(forFormat format: String) -> String? {
        switch format.lowercased() {
        case "png": UTType.png.identifier
        case "jpeg", "jpg": UTType.jpeg.identifier
        case "webp": UTType.webP.identifier
        default: nil
        }
    }

    /// What the identity path decodes, and nothing else.
    public static let identityReadable: Set<String> = [UTType.png.identifier, UTType.jpeg.identifier]

    /// Reads, conforms and encodes a file, entirely off the main actor.
    public static func load(_ url: URL, accepting: Set<String>) async throws -> ImportedPicture {
        try await Task.detached(priority: .userInitiated) {
            let data = try Data(contentsOf: url)
            return try conform(data, name: url.lastPathComponent, accepting: accepting)
        }.value
    }

    /// Conforms and encodes bytes that are already in hand -- a Library
    /// print's, fetched from the machine that holds it -- off the main actor.
    /// Same policy as a file: bytes mold MADE are not necessarily bytes the
    /// well asking for them can decode.
    public static func conforming(
        _ data: Data, name: String, accepting: Set<String>
    ) async throws -> ImportedPicture {
        try await Task.detached(priority: .userInitiated) {
            try conform(data, name: name, accepting: accepting)
        }.value
    }

    /// Base64 for bytes that already came from a machine -- no transcode, but
    /// still off the main actor: the encode alone is a third of a second on a
    /// large still.
    public static func encoded(_ data: Data, name: String) async -> ImportedPicture {
        await Task.detached(priority: .userInitiated) {
            ImportedPicture(encoded: data.base64EncodedString(), name: name, data: data)
        }.value
    }

    /// A picture's pixel dimensions, read from its HEADER -- no decode, so
    /// this is cheap enough to call on the main actor right after an import.
    public static func pixelSize(of data: Data) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        return (5...8).contains(orientation) ? (height, width) : (width, height)
    }

    /// Passes acceptable bytes through untouched; re-encodes anything else as
    /// PNG, which every decoder behind mold reads.
    public static func conform(
        _ data: Data, name: String, accepting: Set<String>
    ) throws -> ImportedPicture {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let type = CGImageSourceGetType(source) as String?
        else { throw PictureImportError.undecodable(name: name) }

        guard !accepting.contains(type) else {
            return ImportedPicture(encoded: data.base64EncodedString(), name: name, data: data)
        }
        let png = try transcodeToPNG(source, name: name)
        return ImportedPicture(
            encoded: png.base64EncodedString(),
            name: (name as NSString).deletingPathExtension + ".png",
            data: png)
    }

    /// The thumbnail route with no size limit is the one that APPLIES the
    /// EXIF orientation: an iPhone photograph is stored landscape with a
    /// rotation flag, and `NSBitmapImageRep(cgImage:)` alone would send it
    /// sideways.
    private static func transcodeToPNG(_ source: CGImageSource, name: String) throws -> Data {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxTranscodePixels,
        ]
        let png = NSMutableData()
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
              let destination = CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil)
        else { throw PictureImportError.undecodable(name: name) }
        // ImageIO rather than AppKit's bitmap rep, so the phone converts a
        // HEIC exactly as the Mac does.
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw PictureImportError.undecodable(name: name) }
        return png as Data
    }

    /// Well past any canvas mold renders, so a transcode is a format change
    /// and not a downscale anybody would notice.
    private static let maxTranscodePixels = 16_384
}

public enum PictureImportError: LocalizedError {
    case undecodable(name: String)

    public var errorDescription: String? {
        switch self {
        case let .undecodable(name):
            "\(name) isn't a picture this app can read."
        }
    }
}
