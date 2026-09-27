import AppKit
import Testing
@testable import Mold

@MainActor
struct ImageZoomTests {
    @Test func generateImageRetainsAccessiblePromptAction() {
        let view = ImageScrollView()
        let image = NSImage(size: NSSize(width: 1200, height: 800))
        var presses = 0
        view.update(image: image, identity: "first", fitRequest: 0, onClick: { presses += 1 })
        #expect(view.picture.accessibilityRole() == .button)
        #expect(view.picture.accessibilityPerformPress())
        #expect(presses == 1)
        view.update(image: image, identity: "first", fitRequest: 0, onClick: nil)
        #expect(view.picture.accessibilityRole() == .image)
        #expect(!view.picture.accessibilityPerformPress())
        #expect(presses == 1)
    }

    @Test func fitAndIdentityResetMagnification() {
        let view = ImageScrollView()
        view.frame = NSRect(x: 0, y: 0, width: 600, height: 400)
        view.layoutSubtreeIfNeeded()
        let image = NSImage(size: NSSize(width: 1200, height: 800))
        view.update(image: image, identity: "first", fitRequest: 0, onClick: nil)
        view.magnification = 3
        // A full-size image arriving after its thumbnail must keep the zoom.
        view.update(image: image, identity: "first", fitRequest: 0, onClick: nil)
        #expect(view.magnification == 3)
        view.update(image: image, identity: "second", fitRequest: 0, onClick: nil)
        #expect(view.magnification == 1)
        view.magnification = 2
        view.update(image: image, identity: "second", fitRequest: 1, onClick: nil)
        #expect(view.magnification == 1)
        #expect(view.contentView.bounds.origin == .zero)
    }

    @Test func viewportFitsAndBoundsZoom() {
        let view = ImageScrollView()
        view.frame = NSRect(x: 0, y: 0, width: 600, height: 400)
        view.layoutSubtreeIfNeeded()
        #expect(view.picture.frame.size == view.contentView.frame.size)
        #expect(view.allowsMagnification)
        #expect(view.minMagnification == 1)
        #expect(view.maxMagnification == 8)
        view.frame.size = NSSize(width: 400, height: 300)
        view.layoutSubtreeIfNeeded()
        #expect(view.picture.frame.size == view.contentView.frame.size)
    }
}
