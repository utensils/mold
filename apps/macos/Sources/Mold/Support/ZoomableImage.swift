import AppKit
import SwiftUI

/// A shared still-image viewport. AppKit owns pinch anchoring, scroll momentum
/// and clipping; SwiftUI owns the image identity and the surrounding actions.
struct ZoomableImage: View {
    let image: NSImage
    let identity: AnyHashable
    var onClick: (() -> Void)?
    @State private var fitRequest = 0

    var body: some View {
        ImageViewport(image: image, identity: identity, fitRequest: fitRequest, onClick: onClick)
            .overlay(alignment: .bottomTrailing) {
                Button("Fit", systemImage: "arrow.down.right.and.arrow.up.left") { fitRequest += 1 }
                    .help("Fit image to window. Pinch to zoom; drag to pan.")
                    .padding(8)
            }
    }
}

private struct ImageViewport: NSViewRepresentable {
    let image: NSImage
    let identity: AnyHashable
    let fitRequest: Int
    let onClick: (() -> Void)?

    func makeNSView(context: Context) -> ImageScrollView { ImageScrollView() }

    func updateNSView(_ view: ImageScrollView, context: Context) {
        view.update(image: image, identity: identity, fitRequest: fitRequest, onClick: onClick)
    }
}

/// The document always starts at the viewport size, with the image fitted
/// inside it. Magnification then enlarges that document without resampling.
final class ImageScrollView: NSScrollView {
    let picture = PannableImageView()
    private var identity: AnyHashable?
    private var fitRequest = 0
    private var viewportSize = NSSize.zero

    init() {
        super.init(frame: .zero)
        drawsBackground = false
        allowsMagnification = true
        minMagnification = 1
        maxMagnification = 8
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = true
        scrollerStyle = .overlay
        picture.imageScaling = .scaleProportionallyUpOrDown
        picture.imageAlignment = .alignCenter
        picture.setAccessibilityLabel("Image preview")
        documentView = picture
    }

    required init?(coder: NSCoder) { nil }

    func update(image: NSImage, identity: AnyHashable, fitRequest: Int, onClick: (() -> Void)?) {
        picture.image = image
        picture.onClick = onClick
        picture.setAccessibilityRole(onClick == nil ? .image : .button)
        picture.setAccessibilityHelp(onClick == nil
            ? "Pinch to zoom, drag to pan. Use Fit to reset the view."
            : "Show or hide the prompt. Pinch to zoom, drag to pan. Use Fit to reset the view.")
        if self.identity != identity || self.fitRequest != fitRequest { fit() }
        self.identity = identity
        self.fitRequest = fitRequest
    }

    override func layout() {
        super.layout()
        let size = contentView.frame.size
        guard size.width > 0, size.height > 0, size != viewportSize else { return }
        viewportSize = size
        picture.setFrameSize(size)
    }

    func fit() {
        magnification = 1
        contentView.scroll(to: .zero)
        reflectScrolledClipView(contentView)
    }
}
