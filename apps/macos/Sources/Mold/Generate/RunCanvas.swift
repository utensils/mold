import AppKit
import MoldClient
import SwiftUI

/// What the Generate pane shows while and after a render.
struct RunCanvas: View {
    let state: RunState
    let host: MoldHost?
    /// What a finished result can do -- the same closures the result bar's
    /// buttons and the contextual menu beside them both perform.
    let actions: ResultActions
    /// Clicking the picture -- and only the picture, never the empty canvas or
    /// the buttons under it -- tucks the prompt away and brings it back.
    let togglePrompt: () -> Void
    /// Called once the settled outcome is really on screen, which is what
    /// releases the next queued batch onto the canvas (`ResultHandoff`).
    let onResultShown: () -> Void

    /// Not `private`: `RunCanvas+Result` reads all three from its own
    /// extension methods, and `private` does not cross files for the same
    /// type.
    @Environment(HostStore.self) var hosts
    @State private var preview: NSImage?
    @State var result: RunResultMedia = .loading
    @State var selected = 0

    var body: some View {
        ZStack {
            switch state {
            case .idle, .submitting:
                idle
            case .running, .runningChain:
                running
            case let .finished(outcome, _):
                // The queue is released by the MEDIA arm reaching a drawn
                // state (`RunCanvas+Result.show`), not by this container
                // appearing: at that instant `result` is still `.loading`.
                finishedView(outcome)
            case let .failed(message):
                ContentUnavailableView("That didn't finish", systemImage: "exclamationmark.triangle",
                                       description: Text(message))
                    // A failure sentence is a drawn outcome too.
                    .task(id: message) { onResultShown() }
            }
        }
        // Decoded on change, never in `body`: `body` re-runs on every progress
        // tick and decoding a PNG there would burn a decode per tick.
        .onChange(of: state.previewData, initial: true) { _, data in
            preview = data.flatMap(NSImage.init(data:))
        }
        // A fresh submission starts the selection over -- otherwise a batch
        // of one after a batch of four could restore a stale index 2 with
        // nothing at position 2 to show.
        .onChange(of: state.isBusy) { _, busy in if busy { selected = 0 } }
        .task(id: resultFilename) { await loadResult() }
    }

    @ViewBuilder private var idle: some View {
        ContentUnavailableView {
            Label("Nothing rendered yet", systemImage: "wand.and.sparkles")
        } description: {
            Text(state.isBusy ? "Sending…" : "Your picture appears here.")
        }
    }

    @ViewBuilder private var running: some View {
        VStack(spacing: 14) {
            if let preview {
                Image(nsImage: preview)
                    .resizable()
                    .interpolation(.medium)
                    .aspectRatio(contentMode: .fit)
                    .padding(40)
                    .onTapGesture(perform: togglePrompt)
            } else {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(state.stage ?? "Getting ready…")
                        .foregroundStyle(.secondary)
                    Text("A preview appears after the first step.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}
