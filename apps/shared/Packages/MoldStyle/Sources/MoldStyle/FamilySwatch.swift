import SwiftUI
#if os(macOS)
import AppKit
#endif

// The ONE place a family colour becomes a colour: the values are the
// generated table (`mold_core::model_presentation::FAMILY_COLORS`), never
// picked here, which is why `swift-lint.sh color` exempts this file alone.

public extension Color {
    /// An sRGB `0xRRGGBB` value, as the generated family colour table holds it.
    init(rgb: UInt32) {
        self.init(.sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255)
    }
}

/// A model family's colour, as a dot beside its name -- the way Finder shows
/// a tag. Decorative: the name beside it carries the meaning, so it is hidden
/// from accessibility and never the only signal.
public struct FamilySwatch: View {
    @ScaledMetric(relativeTo: .caption) private var diameter: CGFloat = 8
    let rgb: UInt32

    public init(rgb: UInt32) { self.rgb = rgb }

    public var body: some View {
        Circle()
            .fill(Color(rgb: rgb))
            .overlay(Circle().strokeBorder(.primary.opacity(0.15), lineWidth: 0.5))
            .frame(width: diameter, height: diameter)
            .accessibilityHidden(true)
    }
}

#if os(macOS)
public extension FamilySwatch {
    /// The dot as a menu image. A menu draws SwiftUI images as templates,
    /// which would paint every family dot the text colour; this bitmap is
    /// not a template, so it keeps the family's colour, as Finder's tag
    /// dots do. One image per colour.
    @MainActor static func menuImage(rgb: UInt32) -> NSImage {
        if let cached = menuImages[rgb] { return cached }
        let image = NSImage(size: NSSize(width: 12, height: 12), flipped: false) { rect in
            let dot = NSBezierPath(ovalIn: rect.insetBy(dx: 1.5, dy: 1.5))
            NSColor(srgbRed: CGFloat((rgb >> 16) & 0xFF) / 255, green: CGFloat((rgb >> 8) & 0xFF) / 255,
                    blue: CGFloat(rgb & 0xFF) / 255, alpha: 1).setFill()
            dot.fill()
            NSColor.labelColor.withAlphaComponent(0.15).setStroke()
            dot.lineWidth = 0.5
            dot.stroke()
            return true
        }
        image.isTemplate = false
        menuImages[rgb] = image
        return image
    }

    @MainActor private static var menuImages: [UInt32: NSImage] = [:]
}
#endif
