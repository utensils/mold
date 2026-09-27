import AppKit

/// Preserve Generate's single-click action, without firing it after a pan.
final class PannableImageView: NSImageView {
    var onClick: (() -> Void)?
    private var previousPoint = NSPoint.zero
    private var startPoint = NSPoint.zero
    private var dragged = false

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
