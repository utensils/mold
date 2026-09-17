import AppKit
import MetalKit
import MoldClient

/// The `MTKView` the mesh is drawn into, and the thing the mouse and keyboard
/// talk to.
///
/// Paused and redraw-on-demand: nothing is rendered until the camera moves or
/// the view is resized, so a still mesh on screen costs no frames at all --
/// which is also how "render only while visible" is satisfied, since the tour
/// below is the only thing that marks it dirty on its own and it stops the
/// moment the view leaves a window.
final class MeshMetalView: MTKView {
    /// The first drag, key or wheel ends the tour, permanently, for this view.
    private(set) var interacted = false
    /// True only while the mesh is actually turning on its own.
    private(set) var autoRotating = false
    var wantsAutoRotate = false { didSet { wantsAutoRotate ? startTour() : stopTour() } }
    /// Told whenever the tour starts or stops, so the caption can say so.
    var onAutoRotateChange: ((Bool) -> Void)?

    private var tour: Timer?
    private var tourStamp: CFTimeInterval = -1
    /// A stalled run loop hands back a huge delta; a jump is worse than a
    /// dropped frame.
    private static let maximumStepSeconds: CFTimeInterval = 0.1

    let renderer: MeshRenderer

    init(renderer: MeshRenderer) {
        self.renderer = renderer
        super.init(frame: .zero, device: renderer.device)
        delegate = renderer
        colorPixelFormat = .bgra8Unorm
        depthStencilPixelFormat = .depth32Float
        // Transparent over the media bed, exactly as the reference clears to
        // zero over `--mold-media-bed` rather than painting its own backdrop.
        clearColor = MTLClearColorMake(0, 0, 0, 0)
        layer?.isOpaque = false
        isPaused = true
        enableSetNeedsDisplay = true
    }

    @available(*, unavailable)
    required init(coder: NSCoder) { fatalError("not from a nib") }

    // MARK: - Lifecycle

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window == nil { stopTour() } else { startTour() }
        needsDisplay = true
    }

    override var acceptsFirstResponder: Bool { true }

    override func becomeFirstResponder() -> Bool { true }

    /// A click focuses the view, so the arrows reach it rather than the
    /// Library's previous/next.
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    // MARK: - The tour

    /// `0.25 rad/s` until the first interaction, off under Reduce Motion.
    private func startTour() {
        guard wantsAutoRotate, !interacted, !autoRotating, window != nil else { return }
        guard !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion else { return }
        autoRotating = true
        onAutoRotateChange?(true)
        tourStamp = -1
        let timer = Timer(timeInterval: 1.0 / 60, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.stepTour() }
        }
        RunLoop.main.add(timer, forMode: .common)
        tour = timer
    }

    func stopTour() {
        tour?.invalidate()
        tour = nil
        tourStamp = -1
        if autoRotating {
            autoRotating = false
            onAutoRotateChange?(false)
        }
    }

    private func stepTour() {
        guard autoRotating, let window, window.occlusionState.contains(.visible) else {
            // Occluded or scrolled away: park rather than burn a frame. The
            // next `viewDidMoveToWindow` or camera change starts it again.
            tourStamp = -1
            return
        }
        let now = CACurrentMediaTime()
        let elapsed = tourStamp < 0 ? 0 : Swift.min(now - tourStamp, Self.maximumStepSeconds)
        tourStamp = now
        var camera = renderer.camera
        camera.yaw = MeshViewerMath.advanceAutoRotate(yaw: camera.yaw,
                                                      elapsedMs: elapsed * 1000)
        renderer.setCamera(camera)
        needsDisplay = true
    }

    /// The first drag, key or wheel ends the tour for this view. An
    /// interaction is never taken back.
    func noteInteraction() {
        guard !interacted else { return }
        interacted = true
        stopTour()
    }
}
