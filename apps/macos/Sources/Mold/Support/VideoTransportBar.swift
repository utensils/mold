import AVKit
import SwiftUI

/// Transport stays outside the media; hovering never changes picture opacity.
struct VideoTransportBar: View {
    let player: AVPlayer
    @Binding var isMuted: Bool
    @Binding var displayMode: MediaDisplayMode
    @State private var position = 0.0
    @State private var duration = 0.0
    @State private var isPlaying = false
    @State private var isScrubbing = false
    private let clock = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                Button {
                    if player.rate == 0 {
                        if Self.shouldRestart(position: player.currentTime().seconds,
                                              duration: player.currentItem?.duration.seconds ?? 0) {
                            player.seek(to: .zero)
                        }
                        player.play()
                    } else {
                        player.pause()
                    }
                } label: {
                    Label(isPlaying ? "Pause" : "Play", systemImage: isPlaying ? "pause.fill" : "play.fill")
                }
                .labelStyle(.iconOnly)
                .help(isPlaying ? "Pause video" : "Play video")
                // The binding receives mouse, keyboard and accessibility
                // changes. Timer updates only change local state, never seek.
                Slider(value: Binding(get: { position }, set: { value in
                    position = value
                    player.seek(to: CMTime(seconds: value, preferredTimescale: 600),
                                    toleranceBefore: .zero, toleranceAfter: .zero)
                }), in: 0...max(duration, 1), onEditingChanged: { isScrubbing = $0 })
                .disabled(duration <= 0)
                .accessibilityLabel("Playback position")
                Text("\(Self.timestamp(position)) / \(Self.timestamp(duration))")
                    .font(.caption.monospacedDigit())
                    .fixedSize()
            }
            HStack {
                Button { isMuted.toggle() } label: {
                    Label(isMuted ? "Unmute" : "Mute",
                          systemImage: isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                }
                .help(isMuted ? "Unmute all videos" : "Mute all videos")
                Button {
                    NSApp.sendAction(#selector(NSWindow.toggleFullScreen(_:)), to: nil, from: nil)
                } label: {
                    Label("Full Screen", systemImage: "arrow.up.left.and.arrow.down.right")
                }
                .labelStyle(.iconOnly)
                .help("Toggle full-screen window")
                Spacer()
                MediaSizeControls { displayMode = $0 }
            }
        }
        .buttonStyle(.bordered)
        .padding(8)
        .onReceive(clock) { _ in
            let seconds = player.currentItem?.duration.seconds ?? 0
            duration = seconds.isFinite && seconds > 0 ? seconds : 0
            if !isScrubbing {
                let current = player.currentTime().seconds
                position = current.isFinite ? min(max(0, current), duration) : 0
            }
        }
        .onReceive(player.publisher(for: \.rate, options: [.initial, .new])) { isPlaying = $0 != 0 }
    }

    static func timestamp(_ seconds: Double) -> String {
        let value = seconds.isFinite && seconds > 0 ? Int(seconds) : 0
        return String(format: "%d:%02d", value / 60, value % 60)
    }

    static func shouldRestart(position: Double, duration: Double) -> Bool {
        duration.isFinite && duration > 0 && position.isFinite && position >= duration
    }
}
