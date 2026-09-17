import AppKit
import MoldClient
import SwiftUI

// The finished view: the selected picture, the strip of the others beneath
// it when there is more than one, and the bar of actions. Split from
// `RunCanvas.swift` purely for size.
extension RunCanvas {
    @ViewBuilder func finishedView(_ outcome: BatchOutcome) -> some View {
        if let result {
            VStack(spacing: 12) {
                Image(nsImage: result)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .onTapGesture(perform: togglePrompt)
                    .accessibilityAddTraits(.isButton)
                    .accessibilityHint("Hides the prompt so the picture fills the pane")
                // Absent for a batch of one, so the ordinary case is
                // byte-identical to before a batch could be more than one
                // child.
                if outcome.results.count > 1 {
                    ResultStrip(results: outcome.results, host: host, selected: $selected)
                }
                if let current = selectedResult(in: outcome) {
                    ResultBar(result: current, host: host, showInLibrary: showInLibrary)
                }
                if let summary = outcome.failureSummary {
                    Text(summary)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(24)
        } else {
            ProgressView("Fetching your picture…")
        }
    }

    var resultFilename: String? {
        guard case let .finished(outcome, _) = state else { return nil }
        return selectedResult(in: outcome)?.filename
    }

    func loadResult() async {
        guard let filename = resultFilename, let host else { result = nil; return }
        guard let data = try? await hosts.backend(for: host).media(filename, trashed: false)
        else { return }
        result = NSImage(data: data)
    }

    private func selectedResult(in outcome: BatchOutcome) -> BatchResult? {
        outcome.results.indices.contains(selected) ? outcome.results[selected] : nil
    }
}
