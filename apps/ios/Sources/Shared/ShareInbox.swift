import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Photos the Share extension hands to the app (DESIGN.md §5.9). The
/// extension never networks: it downsamples the picture to at most 2048 px --
/// straight from the file, never decoding a 48 MP photo whole, since an
/// extension is killed past about 120 MB -- writes it and a small manifest
/// into the App Group, and the app offers it in Generate.
nonisolated enum ShareInbox {
    enum Use: String, Codable, CaseIterable, Identifiable {
        /// "Start from": the source picture.
        case source
        /// A reference image.
        case reference
        /// Imported into the Library as it is.
        case library

        var id: Self { self }
    }

    struct Item: Codable, Equatable, Identifiable {
        var id: String
        var use: Use
        /// The picture's file name in the inbox.
        var file: String
        var created: Date
        var width: Int
        var height: Int
    }

    enum Failure: Error, Equatable { case unreadable, unwritable }

    static let maxPixels = 2048

    /// Downsamples and stages one picture. Alpha is kept (PNG); everything
    /// else, HEIC included, becomes JPEG.
    static func stage(_ source: URL, use: Use, in directory: URL = AppGroup.shareInbox, now: Date = .now) throws -> Item {
        guard let image = CGImageSourceCreateWithURL(source as CFURL, nil) else { throw Failure.unreadable }
        return try stage(image, use: use, in: directory, now: now)
    }

    static func stage(data: Data, use: Use, in directory: URL = AppGroup.shareInbox, now: Date = .now) throws -> Item {
        guard let image = CGImageSourceCreateWithData(data as CFData, nil) else { throw Failure.unreadable }
        return try stage(image, use: use, in: directory, now: now)
    }

    private static func stage(_ source: CGImageSource, use: Use, in directory: URL, now: Date) throws -> Item {
        guard let picture = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixels,
            kCGImageSourceShouldCacheImmediately: false,
        ] as CFDictionary) else { throw Failure.unreadable }
        let alpha = ![.none, .noneSkipFirst, .noneSkipLast].contains(picture.alphaInfo)
        let id = UUID().uuidString
        let file = id + (alpha ? ".png" : ".jpg")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let type = alpha ? UTType.png : UTType.jpeg
        guard let out = CGImageDestinationCreateWithURL(directory.appending(path: file) as CFURL,
                                                        type.identifier as CFString, 1, nil)
        else { throw Failure.unwritable }
        CGImageDestinationAddImage(out, picture, alpha ? nil : [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        guard CGImageDestinationFinalize(out) else { throw Failure.unwritable }
        let item = Item(id: id, use: use, file: file, created: now, width: picture.width, height: picture.height)
        try JSONEncoder().encode(item).write(to: directory.appending(path: id + ".json"), options: .atomic)
        return item
    }

    /// What is waiting, oldest first.
    static func items(in directory: URL = AppGroup.shareInbox) -> [Item] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasSuffix(".json") }
            .compactMap { try? JSONDecoder().decode(Item.self, from: Data(contentsOf: directory.appending(path: $0))) }
            .filter { FileManager.default.fileExists(atPath: directory.appending(path: $0.file).path) }
            .sorted { $0.created < $1.created }
    }

    static func url(of item: Item, in directory: URL = AppGroup.shareInbox) -> URL {
        directory.appending(path: item.file)
    }

    static func remove(_ item: Item, in directory: URL = AppGroup.shareInbox) {
        try? FileManager.default.removeItem(at: directory.appending(path: item.file))
        try? FileManager.default.removeItem(at: directory.appending(path: item.id + ".json"))
    }
}
