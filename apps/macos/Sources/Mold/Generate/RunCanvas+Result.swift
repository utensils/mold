import AVKit
import AppKit
import MoldClient
import SwiftUI

// The finished view: what the batch made, the strip of the others beneath it
// when there is more than one, and the bar of actions. Split from
// `RunCanvas.swift` purely for size.
extension RunCanvas {
    /// The chrome is OUTSIDE the media, because it is about the batch rather
    /// than about one picture: a clip that cannot be decoded as a still, a
    /// mesh, or a fetch that failed used to take `ResultStrip`, `ResultBar`
    /// and even `failureSummary` down with it (parity report §5.1). Nothing
    /// here waits on bytes.
    @ViewBuilder func finishedView(_ outcome: BatchOutcome) -> some View {
        VStack(spacing: 12) {
            media(of: outcome)
            // Absent for a batch of one, so the ordinary case is
            // byte-identical to before a batch could be more than one child.
            if outcome.results.count > 1 {
                ResultStrip(results: outcome.results, host: host,
                            actions: actions, selected: $selected)
            }
            if let current = selectedResult(in: outcome) {
                ResultBar(result: current, host: host, actions: actions)
            }
            if let summary = outcome.failureSummary {
                Text(summary).font(.callout).foregroundStyle(.secondary)
            }
        }
        .padding(24)
    }

    @ViewBuilder private func media(of outcome: BatchOutcome) -> some View {
        switch result {
        case .loading:
            ProgressView("Fetching what you made…")
        case let .picture(image):
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
                .onTapGesture(perform: togglePrompt)
                .accessibilityAddTraits(.isButton)
                .accessibilityHint("Hides the prompt so the picture fills the pane")
                // The big picture offers the same menu as its strip tile.
                .modifier(OptionalResultMenu(result: selectedResult(in: outcome), actions: actions))
        case let .clip(player):
            // Streamed, not downloaded, exactly as `LibraryViewer` plays one:
            // a clip can be hundreds of megabytes and waiting for all of it
            // before the first frame is not playback.
            VideoPlayer(player: player)
                .aspectRatio(contentMode: .fit)
                .onDisappear {
                    player.pause()
                    player.replaceCurrentItem(with: nil)
                }
                .modifier(OptionalResultMenu(result: selectedResult(in: outcome), actions: actions))
        case .mesh:
            // SEAM for Wave 3 · F1: the interactive `MeshView` replaces this
            // whole arm. Until then the canvas names what it made and the bar
            // below still saves it, copies it and opens it in the Library.
            ContentUnavailableView {
                Label("A 3-D model", systemImage: "cube.transparent")
            } description: {
                Text("Open it in the Library to look at it, or export it from there.")
            }
        case let .unavailable(sentence):
            ContentUnavailableView("That didn't arrive", systemImage: "exclamationmark.triangle",
                                   description: Text(sentence))
        }
    }

    var resultFilename: String? {
        guard case let .finished(outcome, _) = state else { return nil }
        return selectedResult(in: outcome)?.filename
    }

    func loadResult() async {
        // A player left running behind a new result keeps playing its audio.
        if case let .clip(previous) = result {
            previous.pause()
            previous.replaceCurrentItem(with: nil)
        }
        result = .loading
        guard let filename = resultFilename, let host else { return }
        let backend = hosts.backend(for: host)
        switch PrintKind(playbackOf: filename) {
        case .clip:
            await playClip(filename, backend: backend, remintsLeft: 1)
        case .mesh:
            result = .mesh
            onResultShown()
        case .picture:
            do {
                let data = try await backend.media(filename, trashed: false)
                guard let image = NSImage(data: data) else {
                    show(.unavailable("The machine sent this back in a form this Mac "
                        + "cannot show. It is in the Library."))
                    return
                }
                show(.picture(image))
            } catch {
                show(.unavailable(error.reasonSentence))
            }
        }
    }

    /// Every terminal media state goes through here, because the beat the
    /// queue waits for is about the PICTURE being drawn -- acknowledging when
    /// the container appeared meant acknowledging while it still read
    /// "Fetching what you made…" (finding 02#9, second half).
    func show(_ media: RunResultMedia) {
        result = media
        onResultShown()
    }

    func selectedResult(in outcome: BatchOutcome) -> BatchResult? {
        outcome.results.indices.contains(selected) ? outcome.results[selected] : nil
    }
}

/// What the finished canvas is showing. A clip, a mesh and a picture are three
/// different things to draw, and a result that cannot be drawn at all is a
/// sentence -- never an endless spinner.
enum RunResultMedia {
    case loading
    case picture(NSImage)
    case clip(AVPlayer)
    case mesh
    case unavailable(String)
}
