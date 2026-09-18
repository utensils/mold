import AVKit
import MoldClient
import SwiftUI

// Playing a finished clip, and noticing when it cannot be played. Split from
// `RunCanvas+Result` purely for size.
extension RunCanvas {
    /// Mints a ticket, plays, and then WATCHES: `AVPlayer` reports a container
    /// it cannot open, or a media ticket that has lapsed under a long-lived
    /// canvas, only through its item's `status`. Without this a clip that
    /// fails to decode is an inert black rectangle -- the same "spins for
    /// ever" complaint the parity fix was about, one step later.
    func playClip(
        _ filename: String, backend: any MoldBackend, remintsLeft: Int
    ) async {
        let player: AVPlayer
        do {
            // `AVPlayer` builds its own requests and cannot carry `X-Api-Key`,
            // so the URL is minted with a media ticket on a keyed host -- the
            // same shared verb the Library plays through.
            player = AVPlayer(url: try await backend.playableURL(for: filename))
        } catch {
            show(.unavailable(error.failureSentence))
            return
        }
        player.play()
        show(.clip(player))

        guard let item = player.currentItem else { return }
        while !Task.isCancelled {
            switch item.status {
            case .readyToPlay:
                return
            case .failed:
                player.pause()
                // A lapsed ticket looks exactly like a broken file from here,
                // so the honest move is to mint a fresh one and try once.
                guard remintsLeft > 0 else {
                    show(.unavailable(item.error?.failureSentence
                        ?? "This clip couldn't be played on this Mac. It is in the Library."))
                    return
                }
                await playClip(filename, backend: backend, remintsLeft: remintsLeft - 1)
                return
            default:
                try? await Task.sleep(for: .milliseconds(120))
            }
        }
    }
}
