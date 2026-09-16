import MoldClient
import SwiftUI

/// How a print was made. Every value is selectable, because the whole point of
/// showing a seed is that somebody copies it.
struct ProvenanceGrid: View {
    let entry: LibraryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let prompt = entry.print.metadata.prompt, !prompt.isEmpty {
                Text(prompt)
                    .font(.callout)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 6) {
                row("Machine", entry.hostName)
                row("Model", meta.model)
                row("Seed", meta.seed.map(String.init))
                row("Steps", meta.steps.map(String.init))
                row("Guidance", meta.guidance.map { $0.formatted(.number.precision(.fractionLength(1))) })
                row("Size", size)
                row("Made", entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                row("File", entry.print.filename)
            }
            .font(.caption)
        }
    }

    private var meta: OutputMetadata { entry.print.metadata }

    private var size: String? {
        guard let width = meta.width, let height = meta.height else { return nil }
        guard let frames = meta.frames else { return "\(width) × \(height)" }
        return "\(width) × \(height) · \(frames) frames"
    }

    @ViewBuilder private func row(_ label: String, _ value: String?) -> some View {
        if let value {
            GridRow {
                Text(label).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text(value).textSelection(.enabled).monospacedDigit().lineLimit(3)
            }
        }
    }
}
