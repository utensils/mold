import AppKit
import Foundation

/// Turning a draft's base64 back into something to look at, OFF the main
/// actor.
///
/// `Data(base64Encoded:)` and `NSImage(data:)` on a 50 MB still are tens of
/// milliseconds each, and every well used to do both inside a `View` method
/// under the app's default MainActor isolation (finding 02#10). One place, so
/// the source well, the reference thumbnails and the mask preview all get it.
enum PicturePreview {
    static func decode(_ encoded: String?) async -> NSImage? {
        guard let encoded else { return nil }
        return await Task.detached(priority: .userInitiated) {
            Data(base64Encoded: encoded).flatMap(NSImage.init(data:))
        }.value
    }
}
