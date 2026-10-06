import AppKit
import MoldClient

/// Native macOS menus flatten the label; use a template image with the offered ratio.
enum AspectRatioIcon {
    static func image(width: Int, height: Int) -> NSImage {
        let bounds = NSSize(width: 26, height: 26)
        let size = AspectRatioGeometry.size(width: width, height: height, bound: 22)
        let image = NSImage(size: bounds, flipped: false) { _ in
            let rect = NSRect(x: (bounds.width - size.width) / 2,
                y: (bounds.height - size.height) / 2, width: size.width, height: size.height)
            let outline = NSBezierPath(roundedRect: rect, xRadius: 2, yRadius: 2)
            outline.lineWidth = 1.5
            NSColor.labelColor.setStroke()
            outline.stroke()
            return true
        }
        image.isTemplate = true
        return image
    }
}
