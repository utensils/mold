import AppKit
import AVKit
import Testing
@testable import Mold

@MainActor
struct MediaViewportTests {
    @Test func portraitAndLandscapeFitInsideBothViewportAxes() {
        let viewport = CGSize(width: 1100, height: 650)
        for pixels in [CGSize(width: 1080, height: 1920), CGSize(width: 3840, height: 1080),
                       CGSize(width: 100, height: 100)] {
            let size = MediaViewportLayout.size(pixels: pixels, viewport: viewport, mode: .fit, displayScale: 2)
            #expect(size.width <= viewport.width)
            #expect(size.height <= viewport.height)
            #expect(abs(size.width / size.height - pixels.width / pixels.height) < 0.000_001)
            #expect(size.width == viewport.width || size.height == viewport.height)
        }
    }

    @Test func actualSizeMeansOneImagePixelPerDisplayPixel() {
        let pixels = CGSize(width: 3840, height: 2160)
        let viewport = CGSize(width: 600, height: 400)
        #expect(MediaViewportLayout.size(pixels: pixels, viewport: viewport, mode: .actualSize, displayScale: 2)
            == CGSize(width: 1920, height: 1080))
        #expect(MediaViewportLayout.size(pixels: pixels, viewport: viewport, mode: .actualSize, displayScale: 1) == pixels)
    }

    @Test func actualSizeImagesCanScrollAndFitResetsThem() {
        let view = ImageScrollView()
        view.frame = CGRect(x: 0, y: 0, width: 600, height: 400)
        view.layoutSubtreeIfNeeded()
        let image = NSImage(size: CGSize(width: 2400, height: 1600))
        view.update(image: image, identity: "large", fitRequest: 0, onClick: nil,
                    mode: .actualSize, displayScale: 2)
        view.layoutSubtreeIfNeeded()
        #expect(view.picture.frame.width == 1200)
        #expect(view.picture.frame.height == 800)
        #expect(view.picture.imageScaling == .scaleNone)
        view.update(image: image, identity: "large", fitRequest: 1, onClick: nil,
                    mode: .fit, displayScale: 2)
        view.layoutSubtreeIfNeeded()
        #expect(view.picture.frame.size == view.contentView.frame.size)
        #expect(view.picture.imageScaling == .scaleProportionallyUpOrDown)
        #expect(view.magnification == 1)
    }

    @Test func playerDoesNotInstallHoverDimmingControls() {
        let view = UndimmedPlayerView()
        #expect(view.controlsStyle == .none)
        #expect(view.videoGravity == .resizeAspect)
    }

    @Test func playRestartsOnlyAFinishedFiniteClip() {
        #expect(VideoTransportBar.shouldRestart(position: 4, duration: 4))
        #expect(VideoTransportBar.shouldRestart(position: 4.1, duration: 4))
        #expect(!VideoTransportBar.shouldRestart(position: 2, duration: 4))
        #expect(!VideoTransportBar.shouldRestart(position: 0, duration: 0))
        #expect(!VideoTransportBar.shouldRestart(position: 4, duration: .infinity))
    }

    @Test func portraitVideoFitsAfterResizeAndActualSizePreservesPlayer() {
        let view = VideoScrollView()
        let player = AVPlayer()
        let pixels = CGSize(width: 1080, height: 1920)
        view.frame = CGRect(x: 0, y: 0, width: 1100, height: 650)
        view.update(player: player, pixels: pixels, mode: .fit, displayScale: 2)
        view.layoutSubtreeIfNeeded()
        #expect(view.playerView.frame.height <= view.contentView.frame.height)
        #expect(view.playerView.frame.width <= view.contentView.frame.width)
        view.frame.size = CGSize(width: 450, height: 300)
        view.layoutSubtreeIfNeeded()
        #expect(view.playerView.frame.height <= 300)
        view.update(player: player, pixels: pixels, mode: .actualSize, displayScale: 2)
        #expect(view.playerView.frame.size == CGSize(width: 540, height: 960))
        #expect(view.documentView!.frame.height > view.contentView.frame.height)
        #expect(view.playerView.player === player)
        view.update(player: player, pixels: pixels, mode: .fit, displayScale: 2)
        #expect(view.playerView.player === player)
        #expect(view.documentView!.frame.size == view.contentView.frame.size)
    }
}
