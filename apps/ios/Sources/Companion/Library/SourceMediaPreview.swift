import Foundation
import ImageIO
import MoldClient
import UIKit

/// Decoded pixels are immutable and safe to move between the decode worker and UI.
private struct SourcePreviewImage: @unchecked Sendable {
    let image: CGImage
}
@MainActor enum SourceMediaPreview {
    private static let cache = SourceMediaPreviewCache<SourcePreviewImage>(
        maximumCost: 32 * 1024 * 1024, concurrency: 4,
        cost: { $0.image.bytesPerRow * $0.image.height })

    static func peek(key: SourcePreviewKey) -> UIImage? {
        guard let decoded = cache.peek(for: key) else { return nil }
        return UIImage(cgImage: decoded.image)
    }

    static func image(key: SourcePreviewKey, backend: any MoldBackend) async -> UIImage? {
        let decoded = await cache.value(for: key) {
            let bytes = try await backend.retainedSourceMediaThumbnail(for: key.authority.owner.filename, member: key.member.memberId)
            return try await Task.detached(priority: .utility) {
                guard let source = CGImageSourceCreateWithData(bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
                      let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceThumbnailMaxPixelSize: 320,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceShouldCacheImmediately: true
                      ] as CFDictionary) else { throw MoldClientError.malformedResponse }
                return SourcePreviewImage(image: image)
            }.value
        }
        guard !Task.isCancelled, let decoded else { return nil }
        return UIImage(cgImage: decoded.image)
    }
    static func invalidate(host: UUID, filename: String? = nil) async {
        await cache.invalidate { $0.authority.owner.host == host && (filename == nil || $0.authority.owner.filename == filename) }
    }
    static func invalidate(authorities: Set<SourceMediaAuthority>) async {
        await cache.invalidate { authorities.contains($0.authority) }
    }
    static func retain(authorities: Set<SourceMediaAuthority>) async {
        await cache.invalidate { !authorities.contains($0.authority) }
    }
}
