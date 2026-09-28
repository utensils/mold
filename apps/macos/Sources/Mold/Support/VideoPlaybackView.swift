import AVKit
import SwiftUI

/// One player for Generate and Library. The video track's presentation size
/// supplies the ratio; AVKit's SwiftUI view has no useful intrinsic height.
enum VideoPlaybackLayout {
    static func aspectRatio(for size: CGSize) -> CGFloat {
        guard size.width > 0, size.height > 0 else { return 16.0 / 9.0 }
        return size.width / size.height
    }
}

enum VideoPlaybackPreference {
    static let key = "videoPlaybackMuted"

    static func isMuted(in defaults: UserDefaults = AppStorageSuite.defaults) -> Bool {
        defaults.bool(forKey: key)
    }

    static func setMuted(_ muted: Bool, in defaults: UserDefaults = AppStorageSuite.defaults) {
        defaults.set(muted, forKey: key)
    }
}

struct VideoPlaybackView: View {
    let player: AVPlayer

    @AppStorage(VideoPlaybackPreference.key, store: AppStorageSuite.defaults)
    private var isMuted = false
    @State private var presentationSize: CGSize = .zero
    @State private var isObserving = false

    var body: some View {
        Group {
            if let item = player.currentItem {
                VideoPlayer(player: player)
                    .aspectRatio(VideoPlaybackLayout.aspectRatio(for: presentationSize),
                                 contentMode: .fit)
                    .onReceive(item.publisher(for: \.presentationSize, options: [.initial, .new])) {
                        presentationSize = $0
                    }
            }
        }
        .overlay(alignment: .topTrailing) {
            Button {
                isMuted.toggle()
            } label: {
                Label(isMuted ? "Unmute" : "Mute",
                      systemImage: isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
            }
            .buttonStyle(.bordered)
            .padding(12)
            .help(isMuted ? "Unmute all videos" : "Mute all videos")
        }
        .onAppear {
            player.isMuted = isMuted
            isObserving = true
        }
        .onChange(of: isMuted) { _, newValue in player.isMuted = newValue }
        // The native transport may change mute too. Keep that choice in the
        // same preference as the visible button, after the stored value lands.
        .onReceive(player.publisher(for: \.isMuted, options: [.new])) { value in
            if isObserving, isMuted != value { isMuted = value }
        }
        .onDisappear { isObserving = false }
    }
}
