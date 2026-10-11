import Foundation
import ImageIO

/// Immutable decoded pixels cross from the bounded worker to either native UI.
public struct SourceMediaOriginalImage: @unchecked Sendable {
    public let image: CGImage
    public let dimensions: String?

    public static func decode(_ original: SourceMediaFile) async throws -> Self {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .utility) {
            try Task.checkCancellation()
            guard let source = CGImageSourceCreateWithURL(original.url as CFURL,
                    [kCGImageSourceShouldCache: false] as CFDictionary),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                  let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
                  let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
                  width > 0, height > 0 else { throw MoldClientError.malformedResponse }
            try SourceMediaOriginal.checkDecodedDimensions(width: Double(width), height: Double(height))
            guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: max(width, height),
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceShouldCacheImmediately: true
                  ] as CFDictionary) else { throw MoldClientError.malformedResponse }
            try Task.checkCancellation()
            let dimensions: String? = "\(width) × \(height) pixels"
            return Self(image: image, dimensions: dimensions)
        }
        return try await withTaskCancellationHandler {
            let decoded = try await worker.value
            try Task.checkCancellation()
            return decoded
        } onCancel: { worker.cancel() }
    }
}
