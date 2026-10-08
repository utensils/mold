import MetalKit
import MoldClient
import MoldMesh
import SwiftUI

/// A 3-D print you can turn (DESIGN.md §5.2): the Mac's own renderer
/// (`MoldMesh`), driven by touch -- drag to turn, pinch to zoom, double-tap to
/// go back to the first view. It turns slowly on its own until touched, never
/// under Reduce Motion. A mesh that cannot be drawn shows its poster and says
/// why, in one line.
struct MeshPage: View {
    @Environment(HostStore.self) private var hosts
    let entry: LibraryEntry
    var isSelected = true
    @State private var retry = 0
    @State private var loaded: (MeshRenderer, MeshScene)?
    @State private var problem: String?

    var body: some View {
        ZStack {
            if let (renderer, scene) = loaded {
                MeshTouchView(renderer: renderer, scene: scene)
                    .accessibilityLabel("Interactive 3-D view of \(entry.print.displayName). Drag to turn it, pinch to zoom, double-tap to reset.")
            } else if let problem {
                VStack(spacing: 12) {
                    PrintThumbnail(entry: entry, points: 360).frame(width: 280, height: 280)
                        .clipShape(.rect(cornerRadius: 10))
                    Text(problem).foregroundStyle(.white).multilineTextAlignment(.center)
                    Button("Try Again") { retry += 1 }.buttonStyle(.borderedProminent)
                }
                .padding()
            } else {
                ProgressView().tint(.white)
            }
        }
        .task(id: identity) {
            guard isSelected else { loaded = nil; return }
            await load()
        }
    }

    private var identity: String {
        "\(entry.id)|\(hosts.host(entry.hostID)?.baseURL.absoluteString ?? "")|\(hosts.instanceID(of: entry.hostID) ?? "")|\(isSelected)|\(retry)"

    }

    private func load() async {
        guard let host = hosts.host(entry.hostID) else { return }
        loaded = nil
        problem = nil
        let requestIdentity = identity
        var downloaded = false
        do {
            let file = try await hosts.backend(for: host).mediaFile(entry.print.filename, trashed: false)
            defer { try? FileManager.default.removeItem(at: file) }
            try Task.checkCancellation()
            downloaded = true
            let payload = try await Task.detached {
                let data = try ResponseCeiling.readFile(file, ceiling: ResponseCeiling.media, what: "3-D print")
                return try MeshPayload.load(data)
            }.value
            try Task.checkCancellation()
            guard identity == requestIdentity else { return }
            let renderer = try MeshRenderer(pixelFormat: .bgra8Unorm, depthFormat: .depth32Float)
            guard let scene = MeshScene(payload, device: renderer.device) else {
                problem = String(localized: "This 3-D object can't be drawn here — showing its poster instead.")
                return
            }
            loaded = (renderer, scene)
        } catch is CancellationError {
            return
        } catch let failure as MeshViewFailure {
            guard identity == requestIdentity else { return }
            problem = failure.sentence
        } catch {
            guard !Task.isCancelled, identity == requestIdentity else { return }
            problem = downloaded
                ? String(localized: "This 3-D file couldn't be opened: \(error.sentence)")
                : String(localized: "This 3-D file couldn't be downloaded: \(error.reasonSentence)")
        }
    }
}

private struct MeshTouchView: UIViewRepresentable {
    let renderer: MeshRenderer
    let scene: MeshScene
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.device)
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColorMake(0, 0, 0, 0)
        view.isOpaque = false
        view.backgroundColor = .clear
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        renderer.install(scene)
        let coordinator = context.coordinator
        coordinator.view = view
        coordinator.renderer = renderer
        view.addGestureRecognizer(UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.pan(_:))))
        view.addGestureRecognizer(UIPinchGestureRecognizer(target: coordinator, action: #selector(Coordinator.pinch(_:))))
        let reset = UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.reset))
        reset.numberOfTapsRequired = 2
        view.addGestureRecognizer(reset)
        if !reduceMotion { coordinator.startTour() }
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        if reduceMotion { context.coordinator.stopTour() }
        view.setNeedsDisplay()
    }

    static func dismantleUIView(_ view: MTKView, coordinator: Coordinator) {
        coordinator.stopTour()
        view.delegate = nil
        coordinator.renderer?.release()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject {
        weak var view: MTKView?
        var renderer: MeshRenderer?
        private var link: CADisplayLink?
        private var last: CFTimeInterval = -1
        private var pinchBase: CGFloat = 1

        private func move(_ camera: ViewerCamera) {
            renderer?.setCamera(camera)
            view?.setNeedsDisplay()
        }

        @objc func pan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer else { return }
            stopTour()
            let delta = gesture.translation(in: gesture.view)
            gesture.setTranslation(.zero, in: gesture.view)
            move(MeshInteraction.orbit(renderer.camera,
                                       dx: Double(delta.x) * MeshInteraction.dragRadiansPerPixel,
                                       dy: Double(delta.y) * MeshInteraction.dragRadiansPerPixel))
        }

        @objc func pinch(_ gesture: UIPinchGestureRecognizer) {
            guard let renderer else { return }
            stopTour()
            if gesture.state == .began { pinchBase = 1 }
            move(MeshInteraction.zoom(renderer.camera,
                                      by: MeshInteraction.pinchFactor(previous: Double(pinchBase),
                                                                     current: Double(gesture.scale))))
            pinchBase = gesture.scale
        }

        @objc func reset() {
            stopTour()
            move(MeshViewerCamera.homeCamera())
        }

        func startTour() {
            guard link == nil else { return }
            let link = CADisplayLink(target: self, selector: #selector(step))
            link.add(to: .main, forMode: .common)
            self.link = link
        }

        func stopTour() {
            link?.invalidate()
            link = nil
            last = -1
        }

        @objc private func step(_ link: CADisplayLink) {
            guard let renderer else { return }
            let elapsed = last < 0 ? 0 : min(link.timestamp - last, 0.1)
            last = link.timestamp
            var camera = renderer.camera
            camera.yaw = MeshViewerMath.advanceAutoRotate(yaw: camera.yaw, elapsedMs: elapsed * 1000)
            move(camera)
        }
    }
}
