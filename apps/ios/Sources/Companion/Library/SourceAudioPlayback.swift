import AVKit
import SwiftUI

/// Audio uses visible transport instead of a video canvas with no picture.
struct SourceAudioPlayback: View {
    let player: AVPlayer
    @State private var duration = 0.0
    @Environment(\.dynamicTypeSize) private var textSize
    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in
            let current = player.currentTime().seconds
            let position = current.isFinite ? max(0, min(current, duration)) : 0
            VStack(spacing: 16) {
                Image(systemName: "waveform").font(.largeTitle).foregroundStyle(.secondaryText).accessibilityHidden(true)
                let layout = textSize.isAccessibilitySize ? AnyLayout(VStackLayout(spacing: 12)) : AnyLayout(HStackLayout(spacing: 16))
                layout {
                    Button {
                        if player.timeControlStatus == .playing { player.pause() }
                        else { try? PlaybackAudio.configure(); player.play() }
                    } label: {
                        Image(systemName: player.timeControlStatus == .playing ? "pause.fill" : "play.fill")
                            .frame(minWidth: 44, minHeight: 44)
                    }.accessibilityLabel(player.timeControlStatus == .playing ? "Pause audio" : "Play audio")
                    Slider(value: Binding(get: { position }, set: { player.seek(to: CMTime(seconds: $0, preferredTimescale: 600)) }),
                           in: 0...max(duration, 0.01)).disabled(duration <= 0).accessibilityLabel("Playback position")
                    Text("\(Int(position)) / \(Int(duration)) s").font(.caption).monospacedDigit()
                }
            }.padding()
        }
        .task(id: ObjectIdentifier(player)) {
            duration = 0
            if let value = try? await player.currentItem?.asset.load(.duration), value.seconds.isFinite {
                duration = max(0, value.seconds)
            }
        }
    }
}
