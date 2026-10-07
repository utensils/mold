import AppKit
import SwiftUI

enum MediaDisplayMode: String {
    case fit
    case actualSize
}

/// Processing pixels are independent of AppKit points, especially on Retina.
enum MediaViewportLayout {
    static func size(pixels: CGSize, viewport: CGSize, mode: MediaDisplayMode,
                     displayScale: CGFloat) -> CGSize {
        guard pixels.width.isFinite, pixels.height.isFinite,
              pixels.width > 0, pixels.height > 0 else { return .zero }
        if mode == .actualSize {
            let scale = displayScale.isFinite && displayScale > 0 ? displayScale : 1
            return CGSize(width: pixels.width / scale, height: pixels.height / scale)
        }
        guard viewport.width > 0, viewport.height > 0 else { return .zero }
        let scale = min(viewport.width / pixels.width, viewport.height / pixels.height)
        return CGSize(width: pixels.width * scale, height: pixels.height * scale)
    }

    static func pixels(of image: NSImage) -> CGSize {
        let bitmap = image.representations.filter { $0.pixelsWide > 0 && $0.pixelsHigh > 0 }
            .max { $0.pixelsWide < $1.pixelsWide }
        return bitmap.map { CGSize(width: $0.pixelsWide, height: $0.pixelsHigh) } ?? image.size
    }
}

struct MediaSizeControls: View {
    let select: (MediaDisplayMode) -> Void

    var body: some View {
        HStack(spacing: 8) {
            Button("Fit") { select(.fit) }
                .help("Fit the entire picture in the available space")
                .accessibilityIdentifier("media-fit")
            Button("Actual Size") { select(.actualSize) }
                .help("Show one image pixel per display pixel. Scroll to see larger pictures.")
                .accessibilityIdentifier("media-actual-size")
        }
        .buttonStyle(.bordered)
    }
}
