import AppKit
import MoldStyle

/// Preserve Generate's single-click action, without firing it after a pan.
final class PannableImageView: NSImageView {
    var onClick: (() -> Void)?
    /// Paints `AlphaBed`'s checkerboard under the picture's own fitted
    /// rectangle -- never the letterbox around it -- for a print that
    /// carries alpha.
    var showsAlphaBed = false {
        didSet { if showsAlphaBed != oldValue { needsDisplay = true } }
    }
    private var previousPoint = NSPoint.zero
    private var startPoint = NSPoint.zero
    private var dragged = false

    override func draw(_ dirtyRect: NSRect) {
        if showsAlphaBed, let image {
            let picture = AlphaBed.fittedRect(content: image.size, in: bounds)
            // `labelColor` resolves against this view's appearance while it
            // draws, exactly as `.primary` does in the SwiftUI board.
            for square in AlphaBed.squares(in: picture) {
                NSColor.labelColor.withAlphaComponent(
                    square.isLight ? AlphaBed.lightOpacity : AlphaBed.darkOpacity).setFill()
                square.rect.fill()
            }
        }
        super.draw(dirtyRect)
    }

    override func mouseDown(with event: NSEvent) {
        startPoint = event.locationInWindow
        previousPoint = startPoint
        dragged = false
    }

    override func mouseDragged(with event: NSEvent) {
        let point = event.locationInWindow
        if hypot(point.x - startPoint.x, point.y - startPoint.y) > 3 { dragged = true }
        defer { previousPoint = point }
        guard dragged, let scroll = enclosingScrollView, scroll.magnification > 1 else { return }
        let clip = scroll.contentView
        let scale = scroll.magnification
        var origin = clip.bounds.origin
        origin.x -= (point.x - previousPoint.x) / scale
        origin.y -= (point.y - previousPoint.y) / scale
        clip.scroll(to: clip.constrainBoundsRect(NSRect(origin: origin, size: clip.bounds.size)).origin)
        scroll.reflectScrolledClipView(clip)
    }

    override func mouseUp(with event: NSEvent) {
        if !dragged { onClick?() }
    }

    override func accessibilityPerformPress() -> Bool {
        guard let onClick else { return false }
        onClick()
        return true
    }
}
