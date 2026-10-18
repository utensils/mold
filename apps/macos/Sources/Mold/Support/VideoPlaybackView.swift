import AVKit
import MoldClient
import SwiftUI

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
    @State private var playback = VideoPlaybackSession()
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
            VideoTransportBar(player: player, playback: playback, isMuted: $isMuted, displayMode: $displayMode)
        }
        .onAppear {
            playback.attach(player)
            player.isMuted = isMuted
            isObserving = true
            if autoplay { playback.play() }
        }
        .onChange(of: isMuted) { _, newValue in player.isMuted = newValue }
        // The native transport may change mute too. Keep that choice in the
        // same preference as the visible button, after the stored value lands.
        .onReceive(player.publisher(for: \.isMuted, options: [.new])) { value in
            if isObserving, isMuted != value { isMuted = value }
        }
        .onReceive(NotificationCenter.default.publisher(for: .AVPlayerItemDidPlayToEndTime)) { notification in
            guard isObserving, NSApplication.shared.isActive, let item = notification.object as? AVPlayerItem,
                  item === player.currentItem else { return }
            playback.handleEnd(repeating: repeats)
        }
        .onReceive(player.publisher(for: \.rate, options: [.new])) { rate in
            if isObserving { playback.nativeRateChanged(rate) }
        }
        .onReceive(player.publisher(for: \.currentItem, options: [.initial, .new])) { _ in if isObserving { playback.attach(player) } }

        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didResignActiveNotification)) { _ in playback.suspend() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in if isObserving { playback.attach(player) } }
        .onDisappear { isObserving = false; playback.cancel() }
    }
}
