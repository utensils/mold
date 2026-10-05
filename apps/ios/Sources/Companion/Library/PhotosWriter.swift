import Foundation
import Photos

/// PhotoKit invokes its change block on an arbitrary serial queue. Neither
/// that block nor its captures may inherit the app's MainActor isolation.
nonisolated enum PhotosWriter {
    struct Resource: Sendable {
        let url: URL
        let video: Bool
    }
    static func save(_ resources: [Resource]) async throws {
        try await PHPhotoLibrary.shared().performChanges(changeBlock(resources))
    }
    static func changeBlock(_ resources: [Resource]) -> @Sendable () -> Void {
        {
            for resource in resources {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: resource.video ? .video : .photo, fileURL: resource.url, options: nil)
            }
        }
    }
}
