import AVKit
import SwiftUI

/// AppKit gets a bounded scroll viewport, never an aspect-ratio-sized parent.
/// The same player survives Fit/Actual Size and window resizing.
struct VideoViewport: NSViewRepresentable {
    let player: AVPlayer
    let pixels: CGSize
    let mode: MediaDisplayMode
    let displayScale: CGFloat

    func makeNSView(context: Context) -> VideoScrollView { VideoScrollView() }

    func updateNSView(_ view: VideoScrollView, context: Context) {
        view.update(player: player, pixels: pixels, mode: mode, displayScale: displayScale)
    }

    static func dismantleNSView(_ view: VideoScrollView, coordinator: ()) {
        view.playerView.player?.pause()
        view.playerView.player = nil
    }
}

final class UndimmedPlayerView: AVPlayerView {
    init() {
        super.init(frame: .zero)
        // AVKit's hover transport paints a scrim even with inline controls.
        // Accessible transport controls live below the picture instead.
        controlsStyle = .none
        videoGravity = .resizeAspect
    }

    required init?(coder: NSCoder) { nil }
}

final class VideoScrollView: NSScrollView {
    let playerView = UndimmedPlayerView()
    private let canvas = NSView()
    private var pixels = CGSize.zero
    private var mode = MediaDisplayMode.fit
    private var displayScale: CGFloat = 1

    init() {
        super.init(frame: .zero)
        drawsBackground = false
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = true
        scrollerStyle = .overlay
        canvas.addSubview(playerView)
        documentView = canvas
    }

    required init?(coder: NSCoder) { nil }

    func update(player: AVPlayer, pixels: CGSize, mode: MediaDisplayMode, displayScale: CGFloat) {
        let reset = self.mode != mode || playerView.player !== player
        if playerView.player !== player { playerView.player = player }
        self.pixels = pixels
        self.mode = mode
        self.displayScale = displayScale
        needsLayout = true
        layoutSubtreeIfNeeded()
        if reset {
            contentView.scroll(to: .zero)
            reflectScrolledClipView(contentView)
        }
    }

    override func layout() {
        super.layout()
        let viewport = contentView.frame.size
        let content = MediaViewportLayout.size(pixels: pixels, viewport: viewport,
                                               mode: mode, displayScale: displayScale)
        let size = CGSize(width: max(viewport.width, content.width),
                          height: max(viewport.height, content.height))
        if canvas.frame.size != size { canvas.setFrameSize(size) }
        playerView.frame = CGRect(x: (size.width - content.width) / 2,
                                  y: (size.height - content.height) / 2,
                                  width: content.width, height: content.height)
    }
}
