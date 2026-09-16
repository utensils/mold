import MoldClient
import SwiftUI

/// What a selected print is made of.
struct LibraryInspector: View {
    let item: LibraryEntry?
    let host: MoldHost?

    var body: some View {
        Group {
            if let item, let host {
                details(item, host)
            } else {
                ContentUnavailableView("Nothing selected", systemImage: "sidebar.right")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func details(_ item: LibraryEntry, _ host: MoldHost) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                LibraryThumbnail(item: item, host: host, edge: 320)
                    .frame(maxWidth: .infinity)

                if let prompt = item.print.metadata.prompt, !prompt.isEmpty {
                    Text(prompt)
                        .font(.callout)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }

                facts(item)
            }
            .padding(16)
        }
    }

    private func facts(_ item: LibraryEntry) -> some View {
        let meta = item.print.metadata
        return Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 6) {
            row("Machine", item.hostName)
            row("Model", meta.model)
            row("Seed", meta.seed.map(String.init))
            row("Steps", meta.steps.map(String.init))
            row("Guidance", meta.guidance.map { $0.formatted(.number.precision(.fractionLength(1))) })
            row("Size", size(meta))
            row("Made", item.createdAt.formatted(date: .abbreviated, time: .shortened))
            row("File", item.print.filename)
        }
        .font(.caption)
    }

    private func size(_ meta: OutputMetadata) -> String? {
        guard let width = meta.width, let height = meta.height else { return nil }
        guard let frames = meta.frames else { return "\(width) × \(height)" }
        return "\(width) × \(height) · \(frames) frames"
    }

    @ViewBuilder private func row(_ label: String, _ value: String?) -> some View {
        if let value {
            GridRow {
                Text(label)
                    .foregroundStyle(.secondary)
                    .gridColumnAlignment(.trailing)
                Text(value)
                    .textSelection(.enabled)
                    // Numbers and identifiers should not reflow as they change.
                    .monospacedDigit()
                    .lineLimit(3)
            }
        }
    }
}
