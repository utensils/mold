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
                  let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: 4096,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceShouldCacheImmediately: true
                  ] as CFDictionary) else { throw MoldClientError.malformedResponse }
            try Task.checkCancellation()
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            let dimensions: String?
            if let width = properties?[kCGImagePropertyPixelWidth], let height = properties?[kCGImagePropertyPixelHeight] {
                let reduced = ((width as? NSNumber)?.intValue ?? 0) > 4096 || ((height as? NSNumber)?.intValue ?? 0) > 4096
                dimensions = "\(width) × \(height) pixels" + (reduced ? " · Reduced for display" : "")
            } else { dimensions = nil }
            return Self(image: image, dimensions: dimensions)
        }
        return try await withTaskCancellationHandler {
            let decoded = try await worker.value
            try Task.checkCancellation()
            return decoded
        } onCancel: { worker.cancel() }
    }
}
