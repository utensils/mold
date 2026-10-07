import AppKit
import SwiftUI

/// A shared still-image viewport. AppKit owns pinch anchoring, scroll momentum
/// and clipping; SwiftUI owns the image identity and the surrounding actions.
struct ZoomableImage: View {
    let image: NSImage
    let identity: AnyHashable
    var onClick: (() -> Void)?
    /// Draws the checkerboard under the picture: a print whose metadata says
    /// it carries alpha (`OutputMetadata.showsAlphaBed`).
    var alphaBed = false
    @State private var fitRequest = 0
    @State private var mode = MediaDisplayMode.fit
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        VStack(spacing: 0) {
            ImageViewport(image: image, identity: identity, fitRequest: fitRequest,
                          mode: mode, displayScale: displayScale,
                          alphaBed: alphaBed, onClick: onClick)
            HStack {
                Spacer()
                MediaSizeControls { selection in
                    mode = selection
                    fitRequest += 1
                }
            }
            .padding(8)
        }
    }
}

private struct ImageViewport: NSViewRepresentable {
    let image: NSImage
    let identity: AnyHashable
    let fitRequest: Int
    let mode: MediaDisplayMode
    let displayScale: CGFloat
    let alphaBed: Bool
    let onClick: (() -> Void)?

    func makeNSView(context: Context) -> ImageScrollView { ImageScrollView() }

    func updateNSView(_ view: ImageScrollView, context: Context) {
        view.update(image: image, identity: identity, fitRequest: fitRequest, onClick: onClick,
                    mode: mode, displayScale: displayScale)
        view.picture.showsAlphaBed = alphaBed
    }
}

/// The document always starts at the viewport size, with the image fitted
/// inside it. Magnification then enlarges that document without resampling.
final class ImageScrollView: NSScrollView {
    let picture = PannableImageView()
    private var identity: AnyHashable?
    private var fitRequest = 0
    private var viewportSize = NSSize.zero
    private var mode = MediaDisplayMode.fit
    private var displayScale: CGFloat = 1

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

    func update(image: NSImage, identity: AnyHashable, fitRequest: Int, onClick: (() -> Void)?,
                mode: MediaDisplayMode = .fit, displayScale: CGFloat = 1) {
        let reset = self.identity != identity || self.fitRequest != fitRequest || self.mode != mode
        self.mode = mode
        self.displayScale = displayScale
        if mode == .actualSize, let copy = image.copy() as? NSImage {
            copy.size = MediaViewportLayout.size(pixels: MediaViewportLayout.pixels(of: image),
                viewport: viewportSize, mode: mode, displayScale: displayScale)
            picture.image = copy
        } else {
            picture.image = image
        }
        picture.imageScaling = mode == .fit ? .scaleProportionallyUpOrDown : .scaleNone
        picture.onClick = onClick
        picture.setAccessibilityRole(onClick == nil ? .image : .button)
        picture.setAccessibilityHelp(onClick == nil
            ? "Pinch to zoom, drag to pan. Use Fit to reset the view."
            : "Show or hide the prompt. Pinch to zoom, drag to pan. Use Fit to reset the view.")
        if reset { fit() }
        self.identity = identity
        self.fitRequest = fitRequest
        sizeDocument()
    }

    override func layout() {
        super.layout()
        let size = contentView.frame.size
        guard size.width > 0, size.height > 0, size != viewportSize else { return }
        viewportSize = size
        sizeDocument()
    }

    private func sizeDocument() {
        let size = contentView.frame.size
        let actual = mode == .actualSize ? picture.image?.size ?? .zero : .zero
        picture.setFrameSize(NSSize(width: max(size.width, actual.width), height: max(size.height, actual.height)))
    }

    func fit() {
        magnification = 1
        contentView.scroll(to: .zero)
        reflectScrolledClipView(contentView)
    }
}
