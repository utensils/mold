import Combine
import AVKit
import MoldClient
import SwiftUI

/// One page of the viewer: a still you can zoom, a clip that streams, or a
/// mesh you can turn. Anything that cannot be shown says so in a sentence.
struct PrintPage: View {
    let entry: LibraryEntry
    let trashed: Bool

    var body: some View {
        switch entry.print.kind {
        case .clip: ClipPlayer(entry: entry)
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

    var body: some View {
        ZStack {
            if let image {
                ZoomingImage(image: image)
                    .accessibilityLabel(entry.spokenDescription(showsHost: false))
            } else {
                ProgressView().tint(.white)
            }
        }
        .task(id: entry.id.filename) {
            image = await loader.image(for: entry, pixels: 2048, trashed: trashed)
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
    @State private var player: AVPlayer?
    @State private var problem: String?

    var body: some View {
        ZStack {
            if let player {
                VideoPlayer(player: player)
            } else if let problem {
                Text(problem).foregroundStyle(.white).padding()
            } else {
                ProgressView().tint(.white)
            }
        }
        .task(id: entry.id.filename) { await load() }
        .onDisappear { player?.pause() }
    }

    private func load() async {
        await play(from: nil, reminted: false)
    }

    /// Plays from a fresh ticket. A ticket that expires mid-watch fails the
    /// item; that is re-minted ONCE, resuming where it stopped -- a second
    /// failure is a real one and says so.
    private func play(from time: CMTime?, reminted: Bool) async {
        guard let host = hosts.host(entry.hostID) else { return }
        let item: AVPlayerItem
        do {
            item = AVPlayerItem(url: try await hosts.backend(for: host).playableURL(for: entry.print.filename))
        } catch {
            problem = String(localized: "This clip can't be played here: \(error.reasonSentence)")
            return
        }
        if let player { player.replaceCurrentItem(with: item) } else { player = AVPlayer(playerItem: item) }
        if let time {
            await player?.seek(to: time)
            player?.play()
        }
        for await status in item.publisher(for: \.status).values where status == .failed {
            guard !reminted else {
                problem = String(localized: "This clip stopped playing: \(item.error?.reasonSentence ?? String(localized: "the machine closed the stream."))")
                return
            }
            await play(from: player?.currentTime(), reminted: true)
            return
        }
    }}
