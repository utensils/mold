import AVKit
import SwiftUI
import Testing
@testable import MoldCompanion

@Suite(.serialized)
@MainActor
struct PlaybackAudioTests {
    @Test func videoPlaybackIgnoresTheSilentSwitch() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.soloAmbient)
        defer { try? session.setCategory(.soloAmbient) }
        try PlaybackAudio.configure(session)
        #expect(session.category == .playback)
        #expect(session.mode == .moviePlayback)
    }

    @Test func anAudioVideoAssetAdvancesWithAudioEnabled() async throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appending(path: "Fixtures/playback-tone.mp4")
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        #expect(tracks.count == 1)
        try PlaybackAudio.configure()
        let player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
        defer {
            player.pause()
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
        player.play()
        for _ in 0..<50 where player.currentTime().seconds < 0.25 {
            try await Task.sleep(for: .milliseconds(100))
        }
        #expect(player.currentItem?.status == .readyToPlay)
        #expect(player.currentTime().seconds >= 0.25)
        #expect(!player.isMuted && player.volume > 0)
        #expect(AVAudioSession.sharedInstance().category == .playback)
    }

    @Test func pagedVideoKeepsNativeTransportControls() async throws {
        let scene = try #require(UIApplication.shared.connectedScenes.first as? UIWindowScene)
        let previous = scene.keyWindow
        let window = UIWindow(windowScene: scene)
        let player = AVPlayer()
        let host = UIHostingController(rootView: TabView {
            NativeVideoPlayer(player: player).tag(0)
            Text("Next print").tag(1)
        }.tabViewStyle(.page(indexDisplayMode: .never)))
        window.rootViewController = host
        window.makeKeyAndVisible()
        defer {
            window.isHidden = true
            window.rootViewController = nil
            previous?.makeKey()
        }
        func findPlayer(in parent: UIViewController) -> AVPlayerViewController? {
            if let player = parent as? AVPlayerViewController { return player }
            return parent.children.lazy.compactMap { findPlayer(in: $0) }.first
        }
        for _ in 0..<20 where findPlayer(in: host) == nil {
            try await Task.sleep(for: .milliseconds(100))
        }
        let controller = try #require(findPlayer(in: host))
        #expect(controller.player === player)
        #expect(controller.showsPlaybackControls)
        NativeVideoPlayer.dismantleUIViewController(controller, coordinator: ())
        #expect(controller.player == nil)
    }

}
