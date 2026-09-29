import Combine
import AVKit
import MoldClient
import SwiftUI

/// One page of the viewer: a still you can zoom, a clip that streams, or a
/// mesh you can turn. Anything that cannot be shown says so in a sentence.
struct PrintPage: View {
    let entry: LibraryEntry
    let trashed: Bool
    let isSelected: Bool

    var body: some View {
        switch entry.print.kind {
        case .clip: ClipPlayer(entry: entry, isSelected: isSelected)
        case .mesh: MeshPage(entry: entry)
        default: ZoomableStill(entry: entry, trashed: trashed)
        }
    }
}

/// A still, fitted, with pinch and double-tap zoom (a UIScrollView -- the
/// system's own zoom, bounce and centring, which SwiftUI does not yet match).
struct ZoomableStill: View {
    @Environment(ThumbnailLoader.self) private var loader
    let entry: LibraryEntry
    let trashed: Bool
    @State private var image: UIImage?
    @State private var isPreview = false

    var body: some View {
        ZStack(alignment: .bottom) {
            if let image {
                ZoomingImage(image: image)
                    .accessibilityLabel(entry.spokenDescription(showsHost: false))
            } else {
                ProgressView().tint(.white)
            }
            if isPreview {
                // Offline and never opened before: the saved thumbnail is all
                // there is, and it says so rather than pretending.
                Text("Offline — showing a preview")
                    .font(.footnote)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.black.opacity(0.6), in: .capsule)
                    .padding(.bottom, 24)
            }
        }
        .task(id: entry.id.filename) {
            // The grid's thumbnail at once; the print itself when it arrives
            // (from disk after the first view, so offline too).
            image = loader.cachedThumbnail(for: entry)
            if image == nil { image = await loader.image(for: entry, pixels: 512, trashed: trashed) }
            if let full = await loader.original(for: entry, trashed: trashed) {
                image = full
                isPreview = false
            } else {
                isPreview = image != nil
            }
        }
    }
}

private struct ZoomingImage: UIViewRepresentable {
    let image: UIImage

    func makeUIView(context: Context) -> UIScrollView {
        let scroll = UIScrollView()
        scroll.delegate = context.coordinator
        scroll.minimumZoomScale = 1
        scroll.maximumZoomScale = 6
        scroll.showsHorizontalScrollIndicator = false
        scroll.showsVerticalScrollIndicator = false
        scroll.contentInsetAdjustmentBehavior = .never
        let view = UIImageView(image: image)
        view.contentMode = .scaleAspectFit
        view.frame = scroll.bounds
        view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        scroll.addSubview(view)
        context.coordinator.imageView = view
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.doubleTap(_:)))
        tap.numberOfTapsRequired = 2
        scroll.addGestureRecognizer(tap)
        return scroll
    }

    func updateUIView(_ scroll: UIScrollView, context: Context) {
        context.coordinator.imageView?.image = image
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        weak var imageView: UIImageView?
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { imageView }

        @objc func doubleTap(_ gesture: UITapGestureRecognizer) {
            guard let scroll = gesture.view as? UIScrollView else { return }
            if scroll.zoomScale > 1 {
                scroll.setZoomScale(1, animated: true)
            } else {
                let point = gesture.location(in: imageView)
                let size = CGSize(width: scroll.bounds.width / 3, height: scroll.bounds.height / 3)
                scroll.zoom(to: CGRect(origin: CGPoint(x: point.x - size.width / 2, y: point.y - size.height / 2),
                                       size: size), animated: true)
            }
        }
    }
}

/// A clip, streamed through a short-lived media ticket -- never the key in a
/// URL, never the whole file buffered first (`playableURL`). A ticket that
/// expires mid-watch is re-minted once.
struct ClipPlayer: View {
    @Environment(HostStore.self) private var hosts
    let entry: LibraryEntry
    let isSelected: Bool
    @State private var player: AVPlayer?
    @State private var problem: String?

    var body: some View {
        ZStack {
            if let player {
                NativeVideoPlayer(player: player)
            } else if let problem {
                Text(problem).foregroundStyle(.white).padding()
            } else {
                ProgressView().tint(.white)
            }
        }
        // Page-style TabView prepares neighbouring pages. Only the selected
        // clip may fetch a stream or play audio; a swipe cancels that task.
        .task(id: isSelected) {
            if isSelected { await load() }
            else { ClipPlayback.sync(player, isSelected: false) }
        }
        .onDisappear { ClipPlayback.sync(player, isSelected: false) }
    }

    private func load() async {
        player?.pause()
        player = nil
        problem = nil
        await play(from: nil, reminted: false)
    }

    /// Plays from a fresh ticket. A ticket that expires mid-watch fails the
    /// item; that is re-minted ONCE, resuming where it stopped -- a second
    /// failure is a real one and says so.
    private func play(from time: CMTime?, reminted: Bool) async {
        guard let host = hosts.host(entry.hostID) else { return }
        let item: AVPlayerItem
        do {
            try PlaybackAudio.configure()
            let url = try await hosts.backend(for: host).playableURL(for: entry.print.filename)
            try Task.checkCancellation()
            item = AVPlayerItem(url: url)
        } catch is CancellationError {
            return
        } catch {
            problem = String(localized: "This clip can't be played here: \(error.reasonSentence)")
            return
        }
        if let player { player.replaceCurrentItem(with: item) } else { player = AVPlayer(playerItem: item) }
        if let time { await player?.seek(to: time) }
        guard !Task.isCancelled, isSelected else { player?.pause(); return }
        ClipPlayback.sync(player, isSelected: true)
        for await status in item.publisher(for: \.status).values where status == .failed {
            guard !Task.isCancelled else { return }
            guard !reminted else {
                player = nil
                problem = String(localized: "This clip stopped playing: \(item.error?.reasonSentence ?? String(localized: "the machine closed the stream."))")
                return
            }
            await play(from: player?.currentTime(), reminted: true)
            return
        }
    }
}

/// Keep page selection, rather than AVKit's neighbouring page preparation,
/// authoritative for playback. The native player keeps its transport controls.
enum ClipPlayback {
    static func sync(_ player: AVPlayer?, isSelected: Bool) {
        guard let player else { return }
        if isSelected { player.play() } else { player.pause() }
    }
}

/// Keep AVKit's controller and controls intact inside a paged gallery.
/// SwiftUI's VideoPlayer can suppress its controls inside a page-style TabView.
struct NativeVideoPlayer: UIViewControllerRepresentable {
    let player: AVPlayer

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.player = player
        controller.showsPlaybackControls = true
        return controller
    }

    func updateUIViewController(_ controller: AVPlayerViewController, context: Context) {
        if controller.player !== player { controller.player = player }
    }

    static func dismantleUIViewController(_ controller: AVPlayerViewController, coordinator: ()) {
        controller.player?.pause()
        controller.player = nil
    }
}
