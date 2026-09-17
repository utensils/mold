import MoldClient
import SwiftUI

/// What the host says about a render before it is asked for.
struct PlacementHint: View {
    let placement: PlacementPreview?
    let error: String?

    var body: some View {
        Group {
            if let error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(error)
            } else if let candidate = placement?.candidate,
                      let duration = candidate.predictedDuration {
                // A low-confidence estimate is stated as approximate. Showing
                // a guess as a measurement is how a progress bar starts lying.
                Label(
                    "about \(duration.formatted(.units(allowed: [.minutes, .seconds])))",
                    systemImage: candidate.setupKind == "cold" ? "snowflake" : "bolt"
                )
                .help(candidate.estimateConfidence == "low"
                      ? "A rough estimate — this model hasn't run here recently."
                      : "Estimated from recent runs on this machine.")
            } else if let reason = placement?.reason {
                Label(reason, systemImage: "exclamationmark.triangle")
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(reason)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}
