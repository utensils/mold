import AVKit
import MoldClient
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
    @AppStorage(VideoPlaybackPreferences.autoplayKey, store: AppStorageSuite.defaults)
    private var autoplay = VideoPlaybackPreferences.defaultAutoplay
    @AppStorage(VideoPlaybackPreferences.repeatKey, store: AppStorageSuite.defaults)
    private var repeats = VideoPlaybackPreferences.defaultRepeat
    @State private var presentationSize: CGSize = .zero
    @State private var isObserving = false
    @State private var displayMode = MediaDisplayMode.fit
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        VStack(spacing: 0) {
            if let item = player.currentItem {
                VideoViewport(player: player, pixels: presentationSize,
                              mode: displayMode, displayScale: displayScale)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onReceive(item.publisher(for: \.presentationSize, options: [.initial, .new])) {
                        presentationSize = $0
                    }
            }
            VideoTransportBar(player: player, isMuted: $isMuted, displayMode: $displayMode)
        }
        .onAppear {
            player.isMuted = isMuted
            isObserving = true
            if autoplay { player.play() }
        }
        .onChange(of: isMuted) { _, newValue in player.isMuted = newValue }
        // The native transport may change mute too. Keep that choice in the
        // same preference as the visible button, after the stored value lands.
        .onReceive(player.publisher(for: \.isMuted, options: [.new])) { value in
            if isObserving, isMuted != value { isMuted = value }
        }
        .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { notification in
            guard repeats, isObserving, NSApplication.shared.isActive, let item = notification.object as? AVPlayerItem,
                  item === player.currentItem else { return }
            Task { @MainActor in
                await player.seek(to: .zero)
                guard repeats, isObserving, NSApplication.shared.isActive, player.currentItem === item else { return }
                player.play()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in player.pause() }
        .onDisappear { isObserving = false; player.pause() }
    }
}
